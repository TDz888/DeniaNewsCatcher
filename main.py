import asyncio
import html
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiosqlite
import discord
import feedparser
import httpx
from aiolimiter import AsyncLimiter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from mistralai.client import Mistral

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
CHANNEL_NEWS_ID = int(os.getenv("CHANNEL_NEWS_ID", "0"))
CHANNEL_PAPERS_ID = int(os.getenv("CHANNEL_PAPERS_ID", "0"))
CHANNEL_MODELS_ID = int(os.getenv("CHANNEL_MODELS_ID", "0"))
CHANNEL_LOGS_ID = int(os.getenv("CHANNEL_LOGS_ID", "0"))

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")

DB_PATH = os.getenv("DB_PATH", "/data/denia.db")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
HN_MIN_SCORE = int(os.getenv("HN_MIN_SCORE", "100"))
HN_MIN_COMMENTS = int(os.getenv("HN_MIN_COMMENTS", "20"))
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "5"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
LLM_RATE_INTERVAL = float(os.getenv("LLM_RATE_INTERVAL", "1.2"))

logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    canonical_url TEXT,
    url TEXT NOT NULL,
    title TEXT,
    channel_id INTEGER,
    message_id INTEGER,
    status TEXT DEFAULT 'sent',
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS seen_urls (
    canonical_url TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_created ON seen_articles(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_source ON seen_articles(source);
CREATE INDEX IF NOT EXISTS idx_canonical ON seen_articles(canonical_url);
"""


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    query = parse_qs(parsed.query)
    tracking = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source", "_hs")
    for key in list(query.keys()):
        lk = key.lower()
        if any(lk.startswith(t) for t in tracking):
            del query[key]
    new_query = urlencode(query, doseq=True)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path, "", new_query, ""))


def truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def validate_config() -> list[str]:
    errors: list[str] = []
    if not DISCORD_TOKEN:
        errors.append("DISCORD_TOKEN is required")
    if not MISTRAL_API_KEY:
        errors.append("MISTRAL_API_KEY is required")
    if CHANNEL_NEWS_ID == 0:
        errors.append("CHANNEL_NEWS_ID is required")
    if CHANNEL_PAPERS_ID == 0:
        errors.append("CHANNEL_PAPERS_ID is required")
    if CHANNEL_MODELS_ID == 0:
        errors.append("CHANNEL_MODELS_ID is required")
    if CHANNEL_LOGS_ID == 0:
        errors.append("CHANNEL_LOGS_ID is required")
    return errors


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.execute("PRAGMA cache_size=-64000;")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info(f"Database connected at {self.db_path}")

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    async def is_seen(self, source: str, external_id: str, canonical_url: str) -> bool:
        if self._conn is None:
            return False
        async with self._conn.execute(
            "SELECT 1 FROM seen_articles WHERE source = ? AND external_id = ? LIMIT 1",
            (source, external_id),
        ) as cursor:
            if await cursor.fetchone():
                return True
        if canonical_url:
            async with self._conn.execute(
                "SELECT 1 FROM seen_urls WHERE canonical_url = ? LIMIT 1",
                (canonical_url,),
            ) as cursor:
                if await cursor.fetchone():
                    return True
        return False

    async def mark_sent(
        self,
        source: str,
        external_id: str,
        canonical_url: str,
        url: str,
        title: str,
        channel_id: int,
        message_id: int,
    ) -> None:
        if self._conn is None:
            return
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_articles
            (source, external_id, canonical_url, url, title, channel_id, message_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'sent')
            """,
            (source, external_id, canonical_url, url, title, channel_id, message_id),
        )
        if canonical_url:
            await self._conn.execute(
                "INSERT OR IGNORE INTO seen_urls (canonical_url) VALUES (?)",
                (canonical_url,),
            )
        await self._conn.commit()

    async def mark_failed(
        self,
        source: str,
        external_id: str,
        canonical_url: str,
        url: str,
        title: str,
        error: str,
    ) -> None:
        if self._conn is None:
            return
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_articles
            (source, external_id, canonical_url, url, title, status, error)
            VALUES (?, ?, ?, ?, ?, 'failed', ?)
            """,
            (source, external_id, canonical_url, url, title, error[:500]),
        )
        await self._conn.commit()

    async def get_stats(self) -> dict:
        if self._conn is None:
            return {"total_sent": 0, "by_source": {}}
        async with self._conn.execute(
            "SELECT COUNT(*) FROM seen_articles WHERE status='sent'"
        ) as cur:
            sent = (await cur.fetchone())[0]
        async with self._conn.execute(
            "SELECT source, COUNT(*) FROM seen_articles WHERE status='sent' GROUP BY source"
        ) as cur:
            by_source = dict(await cur.fetchall())
        return {"total_sent": sent, "by_source": by_source}


@dataclass
class Article:
    source: str
    external_id: str
    url: str
    title: str
    content: str
    author: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    category: str = "news"
    comments: list[str] = field(default_factory=list)

    @property
    def canonical_url(self) -> str:
        return canonicalize_url(self.url)


class BaseFetcher:
    name: str = "base"
    category: str = "news"

    async def fetch(self) -> list[Article]:
        raise NotImplementedError


AI_KEYWORDS = [
    "ai", "llm", "gpt", "machine learning", "deep learning",
    "neural", "transformer", "diffusion", "rag", "agent",
    "openai", "anthropic", "mistral", "huggingface", "pytorch",
    "tensorflow", "gemini", "claude", "llama", "langchain",
    "embedding", "fine-tuning", "inference", "generative",
    "chatbot", "prompt", "reasoning", "multimodal",
]


def is_ai_related(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for kw in AI_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lowered):
            return True
    return False


async def extract_article_content(url: str) -> str | None:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            raw_html = resp.text
    except Exception as e:
        logger.debug(f"[Extract] HTTP failed for {url[:80]}: {e}")
        return None

    try:
        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer", "header",
            "aside", "form", "noscript", "iframe", "svg",
        ]):
            tag.decompose()

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(class_=re.compile(
                r"(article|post|content|entry|story|body)", re.I
            ))
            or soup.body
            or soup
        )

        paragraphs = main.find_all(["p", "h2", "h3"])
        text_parts: list[str] = []
        seen: set[str] = set()
        for p in paragraphs:
            t = p.get_text(" ", strip=True)
            t = re.sub(r"\s+", " ", t)
            if len(t) > 40 and t not in seen:
                seen.add(t)
                text_parts.append(t)

        text = "\n\n".join(text_parts)
        if len(text) > 150:
            return text

    except Exception as e:
        logger.debug(f"[Extract] Parse failed for {url[:80]}: {e}")

    return None


async def fetch_hn_comments(story_id: str, max_comments: int = 6) -> list[str]:
    url = f"https://hn.algolia.com/api/v1/items/{story_id}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.debug(f"[HN Comments] Fetch failed for {story_id}: {e}")
        return []

    comments: list[str] = []

    def walk(node: dict, depth: int = 0) -> None:
        if depth > 3 or len(comments) >= max_comments:
            return
        for child in node.get("children", []) or []:
            if len(comments) >= max_comments:
                return
            text = child.get("text")
            if text:
                clean = re.sub(r"<[^>]+>", "", text)
                clean = html.unescape(clean).strip()
                clean = re.sub(r"\s+", " ", clean)
                if len(clean) > 80:
                    comments.append(truncate(clean, 400))
            walk(child, depth + 1)

    walk(data)
    return comments[:max_comments]


class HackerNewsFetcher(BaseFetcher):
    name = "hackernews"
    category = "news"
    ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"

    async def fetch(self) -> list[Article]:
        params = {
            "tags": "story",
            "numericFilters": f"points>{HN_MIN_SCORE}",
            "hitsPerPage": MAX_ARTICLES_PER_SOURCE * 3,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.ALGOLIA_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"[HN] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for hit in data.get("hits", []):
            if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                break

            title = hit.get("title") or ""
            external_url = hit.get("url")
            story_text = (hit.get("story_text") or "").strip()
            hn_url = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            url = external_url or hn_url

            if not is_ai_related(f"{title} {story_text}"):
                continue

            content = ""
            if external_url:
                content = await extract_article_content(external_url) or ""

            if not content and story_text:
                content = story_text

            comments: list[str] = []
            num_comments = hit.get("num_comments", 0)
            if num_comments >= HN_MIN_COMMENTS:
                comments = await fetch_hn_comments(hit["objectID"])

            if not content and comments:
                content = (
                    "Bài viết là link ngoài không truy cập được. "
                    "Dưới đây là thảo luận cộng đồng:\n\n"
                    + "\n\n".join(comments)
                )

            if len(content) < 150:
                logger.debug(f"[HN] Skip '{title[:60]}' - content too short ({len(content)} chars)")
                continue

            try:
                published = datetime.fromisoformat(hit["created_at"].replace("Z", "+00:00"))
            except Exception:
                published = None

            articles.append(
                Article(
                    source=self.name,
                    external_id=hit["objectID"],
                    url=url,
                    title=title,
                    content=content[:8000],
                    author=hit.get("author"),
                    published_at=published,
                    metadata={
                        "points": hit.get("points", 0),
                        "comments": num_comments,
                        "hn_url": hn_url,
                        "external_url": external_url,
                    },
                    category=self.category,
                    comments=comments,
                )
            )

            await asyncio.sleep(0.3)

        logger.info(f"[HN] Fetched {len(articles)} AI articles")
        return articles


class HuggingFacePapersFetcher(BaseFetcher):
    name = "hf_papers"
    category = "papers"
    URL = "https://huggingface.co/api/daily_papers"

    async def fetch(self) -> list[Article]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.URL)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"[HF Papers] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for item in data[: MAX_ARTICLES_PER_SOURCE * 2]:
            if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                break

            paper = item.get("paper", {})
            arxiv_id = paper.get("id", "")
            title = (paper.get("title") or "").strip()
            summary = (paper.get("summary") or "").strip()

            if not title or not summary or len(summary) < 200:
                continue

            published = None
            if paper.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        paper["publishedAt"].replace("Z", "+00:00")
                    )
                except Exception:
                    published = None

            authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])

            articles.append(
                Article(
                    source=self.name,
                    external_id=arxiv_id,
                    url=f"https://huggingface.co/papers/{arxiv_id}",
                    title=title,
                    content=summary,
                    author=authors,
                    published_at=published,
                    metadata={
                        "upvotes": paper.get("upvotes", 0),
                        "arxiv_id": arxiv_id,
                        "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
                        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    },
                    category=self.category,
                )
            )

        logger.info(f"[HF Papers] Fetched {len(articles)} papers")
        return articles


class HuggingFaceModelsFetcher(BaseFetcher):
    name = "hf_models"
    category = "models"
    URL = "https://huggingface.co/api/models"

    async def fetch(self) -> list[Article]:
        params = {
            "sort": "trendingScore",
            "direction": -1,
            "limit": MAX_ARTICLES_PER_SOURCE,
            "full": "true",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"[HF Models] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for m in data:
            model_id = m.get("modelId") or m.get("id", "")
            if not model_id:
                continue

            created = m.get("createdAt")
            dt = None
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    dt = None

            tags = m.get("tags", [])
            pipeline = m.get("pipeline_tag", "unknown")
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)
            library = m.get("library_name", "unknown")

            content = (
                f"Model ID: {model_id}\n"
                f"Author: {m.get('author', 'unknown')}\n"
                f"Task: {pipeline}\n"
                f"Library: {library}\n"
                f"Downloads: {downloads}\n"
                f"Likes: {likes}\n"
                f"Tags: {', '.join(tags[:15])}"
            )

            articles.append(
                Article(
                    source=self.name,
                    external_id=model_id,
                    url=f"https://huggingface.co/{model_id}",
                    title=model_id,
                    content=content,
                    author=m.get("author"),
                    published_at=dt,
                    metadata={
                        "pipeline": pipeline,
                        "downloads": downloads,
                        "likes": likes,
                        "tags": tags[:10],
                        "library": library,
                        "model_id": model_id,
                    },
                    category=self.category,
                )
            )

        logger.info(f"[HF Models] Fetched {len(articles)} models")
        return articles


class ArxivFetcher(BaseFetcher):
    name = "arxiv"
    category = "papers"
    URL = "http://export.arxiv.org/api/query"
    CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]

    async def fetch(self) -> list[Article]:
        query = " OR ".join(f"cat:{c}" for c in self.CATEGORIES)
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": MAX_ARTICLES_PER_SOURCE,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.URL, params=params)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
        except Exception as e:
            logger.error(f"[arXiv] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for entry in feed.entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            title = entry.title.strip().replace("\n", " ")
            summary = entry.summary.strip().replace("\n", " ")

            if len(summary) < 200:
                continue

            try:
                published = datetime.fromisoformat(entry.published.replace("Z", "+00:00"))
            except Exception:
                published = None

            authors = ", ".join(a.name for a in entry.get("authors", [])[:3])

            articles.append(
                Article(
                    source=self.name,
                    external_id=arxiv_id,
                    url=entry.id,
                    title=title,
                    content=summary,
                    author=authors,
                    published_at=published,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "arxiv_url": entry.id,
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                    },
                    category=self.category,
                )
            )

        logger.info(f"[arXiv] Fetched {len(articles)} papers")
        return articles


class RSSFetcher(BaseFetcher):
    def __init__(self, name: str, rss_url: str, category: str = "news"):
        self.name = name
        self.rss_url = rss_url
        self.category = category

    async def fetch(self) -> list[Article]:
        try:
            async with httpx.AsyncClient(
                timeout=30, headers={"User-Agent": USER_AGENT}
            ) as client:
                resp = await client.get(self.rss_url)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for entry in feed.entries[: MAX_ARTICLES_PER_SOURCE * 2]:
            if len(articles) >= MAX_ARTICLES_PER_SOURCE:
                break

            title = (entry.get("title") or "").strip()
            url = entry.get("link", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            summary = re.sub(r"<[^>]+>", "", summary)
            summary = html.unescape(summary).strip()

            if not title or not url:
                continue

            if len(summary) < 200:
                extracted = await extract_article_content(url)
                if extracted:
                    summary = extracted

            if len(summary) < 150:
                continue

            published = None
            try:
                if entry.get("published_parsed"):
                    published = datetime(*entry.published_parsed[:6])
            except Exception:
                published = None

            articles.append(
                Article(
                    source=self.name,
                    external_id=entry.get("id", url),
                    url=url,
                    title=title,
                    content=summary[:8000],
                    published_at=published,
                    category=self.category,
                )
            )

        logger.info(f"[{self.name}] Fetched {len(articles)} items")
        return articles


SYSTEM_PROMPT = """Bạn là DeniaNewsCatcher - một chuyên gia phân tích AI, đọc hiểu tin tức, paper, và model với trình độ cao.

QUY TẮC BẮT BUỘC:
1. CHỈ dùng thông tin có trong nội dung được cung cấp. KHÔNG bịa, KHÔNG thêm kiến thức ngoài.
2. TL;DR PHẢI KHÁC với tiêu đề. Không được lặp lại tiêu đề. Phải chứa thông tin cụ thể: con số, tên riêng, sự kiện.
3. Mọi điểm chính phải có thông tin cụ thể, không được chung chung kiểu "công ty đang phát triển AI".
4. Nếu một phần không có thông tin, ghi rõ "Không có thông tin trong bài".
5. Trả về DUY NHẤT một JSON object hợp lệ, không có text nào khác ngoài JSON.
6. Tất cả nội dung bằng tiếng Việt, giữ nguyên tên riêng, tên model, tên công ty bằng tiếng Anh."""

NEWS_PROMPT = """Phân tích bài tin tức AI sau và trả về JSON theo schema:

{{
  "tldr": "3-4 câu tóm tắt cốt lõi. Phải có thông tin cụ thể: ai, làm gì, kết quả ra sao, con số nếu có. KHÔNG được lặp lại tiêu đề.",
  "key_points": [
    "Điểm chính 1 với thông tin cụ thể (tên, số liệu, hành động)",
    "Điểm chính 2",
    "Điểm chính 3",
    "Điểm chính 4 nếu có"
  ],
  "context": "2-3 câu về bối cảnh: tại sao sự việc này xảy ra, có liên quan gì đến xu hướng AI hiện tại.",
  "impact": "2-3 câu về tác động: ai bị ảnh hưởng (công ty, developer, người dùng), lĩnh vực nào.",
  "tags": ["tag1", "tag2", "tag3", "tag4"]
}}

Tiêu đề: {title}
Nguồn: {source}
Nội dung bài viết:
{content}

{comments_section}"""

PAPER_PROMPT = """Phân tích paper AI sau và trả về JSON theo schema:

{{
  "tldr": "3-4 câu tóm tắt toàn bộ paper. Nêu rõ: giải quyết vấn đề gì, dùng phương pháp gì, đạt kết quả ra sao. Có số liệu cụ thể.",
  "problem": "2-3 câu mô tả vấn đề paper giải quyết. Tại sao vấn đề này khó? Cách tiếp cận hiện tại có hạn chế gì?",
  "method": "3-4 câu về phương pháp chính. Kiến trúc, thuật toán, dataset sử dụng, cách huấn luyện nếu có.",
  "results": [
    "Kết quả cụ thể 1: tên benchmark + số liệu (accuracy, F1, BLEU...) + so với baseline nào",
    "Kết quả cụ thể 2",
    "Kết quả cụ thể 3"
  ],
  "limitations": "2 câu về hạn chế của phương pháp. Paper có đề cập hoặc có thể suy ra từ kết quả.",
  "applications": "2 câu về ứng dụng thực tế của nghiên cứu này.",
  "tags": ["tag1", "tag2", "tag3"]
}}

Tiêu đề: {title}
Tác giả: {author}
Nội dung (abstract):
{content}"""

MODEL_PROMPT = """Phân tích model AI sau và trả về JSON theo schema:

{{
  "tldr": "3-4 câu giới thiệu model. Model làm được gì, được huấn luyện thế nào, khác biệt gì so với model khác.",
  "introduction": "2-3 câu về model này: do ai phát triển, mục đích, điểm đặc biệt.",
  "specs": {{
    "task": "Task chính của model",
    "library": "Thư viện sử dụng",
    "license": "Giấy phép nếu có",
    "params": "Số tham số nếu biết, nếu không ghi 'không rõ'",
    "context": "Context length nếu có, nếu không ghi 'không rõ'"
  }},
  "strengths": [
    "Điểm mạnh cụ thể 1",
    "Điểm mạnh cụ thể 2"
  ],
  "weaknesses": [
    "Hạn chế 1",
    "Hạn chế 2"
  ],
  "use_cases": "2 câu về khi nào nên dùng model này, ứng dụng phù hợp.",
  "code_snippet": "3-5 dòng code Python để load model bằng transformers hoặc thư viện tương ứng.",
  "tags": ["tag1", "tag2", "tag3"]
}}

Model ID: {title}
Thông tin model:
{content}"""


def get_prompt(category: str) -> str:
    if category == "papers":
        return PAPER_PROMPT
    if category == "models":
        return MODEL_PROMPT
    return NEWS_PROMPT


class LLMRateLimiter:
    def __init__(self, min_interval: float = 1.2):
        self._limiter = AsyncLimiter(1, min_interval)

    async def __aenter__(self):
        await self._limiter.acquire()
        return self

    async def __aexit__(self, *args):
        pass


llm_limiter = LLMRateLimiter(min_interval=LLM_RATE_INTERVAL)


async def summarize(client: Mistral, article: Article) -> dict | None:
    content = article.content[:7000]
    prompt_template = get_prompt(article.category)

    if article.category == "news":
        if article.comments:
            comments_text = "\n\n".join(f"- {c}" for c in article.comments[:5])
            comments_section = "Thảo luận cộng đồng (từ Hacker News):\n" + comments_text
        else:
            comments_section = ""
        prompt = prompt_template.format(
            title=article.title,
            source=article.source,
            content=content,
            comments_section=comments_section,
        )
    elif article.category == "papers":
        prompt = prompt_template.format(
            title=article.title,
            author=article.author or "không rõ",
            content=content,
        )
    else:
        prompt = prompt_template.format(
            title=article.title,
            content=content,
        )

    for attempt in range(LLM_MAX_RETRIES):
        try:
            async with llm_limiter:
                async with asyncio.timeout(LLM_TIMEOUT):
                    resp = await client.chat.complete_async(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3,
                        max_tokens=LLM_MAX_TOKENS,
                    )

            raw = resp.choices[0].message.content
            data = json.loads(raw)

            if not isinstance(data, dict):
                logger.warning(f"[LLM] Response is not dict for '{article.title[:60]}'")
                return None

            tldr = data.get("tldr", "")
            if not tldr or tldr.strip().lower() == article.title.strip().lower():
                logger.warning(f"[LLM] TL;DR identical to title for '{article.title[:60]}'")
                return None

            logger.info(f"[LLM] OK: '{article.title[:50]}'")
            return data

        except json.JSONDecodeError as e:
            logger.warning(
                f"[LLM] Invalid JSON for '{article.title[:60]}' "
                f"(attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
            )
            if attempt < LLM_MAX_RETRIES - 1:
                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                await asyncio.sleep(wait)

        except asyncio.TimeoutError:
            logger.warning(
                f"[LLM] Timeout for '{article.title[:60]}' "
                f"(attempt {attempt + 1}/{LLM_MAX_RETRIES})"
            )
            if attempt < LLM_MAX_RETRIES - 1:
                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                await asyncio.sleep(wait)

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "rate" in error_str.lower():
                wait = min(120, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(
                    f"[LLM] Rate limited for '{article.title[:60]}' "
                    f"(attempt {attempt + 1}/{LLM_MAX_RETRIES}), waiting {wait:.1f}s"
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    f"[LLM] Failed for '{article.title[:60]}' "
                    f"(attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
                )
                if attempt < LLM_MAX_RETRIES - 1:
                    wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                    await asyncio.sleep(wait)

    logger.error(f"[LLM] Gave up on '{article.title[:60]}' after {LLM_MAX_RETRIES} attempts")
    return None


async def filter_new(db: DatabaseManager, articles: list[Article]) -> list[Article]:
    new_articles: list[Article] = []
    for art in articles:
        if await db.is_seen(art.source, art.external_id, art.canonical_url):
            continue
        new_articles.append(art)

    skipped = len(articles) - len(new_articles)
    if skipped:
        logger.info(f"[Dedup] Skipped {skipped} seen articles")
    return new_articles


async def summarize_articles(
    client: Mistral, articles: list[Article]
) -> list[tuple[Article, dict]]:
    semaphore = asyncio.Semaphore(1)

    async def _one(art: Article):
        async with semaphore:
            summary = await summarize(client, art)
            if summary is None:
                return None
            return (art, summary)

    results = await asyncio.gather(
        *[_one(a) for a in articles], return_exceptions=True
    )

    valid: list[tuple[Article, dict]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"[Summarize] Task exception: {r}")
            continue
        if r is None:
            continue
        valid.append(r)

    logger.info(f"[Summarize] Summarized {len(valid)}/{len(articles)} articles")
    return valid


CATEGORY_COLORS = {
    "news": 0x3498DB,
    "papers": 0x9B59B6,
    "models": 0xE67E22,
}

SOURCE_ICONS = {
    "hackernews": "🟠 Hacker News",
    "hf_papers": "🤗 HuggingFace Papers",
    "hf_models": "🤗 HuggingFace Models",
    "arxiv": "📄 arXiv",
    "mit_tech": "📰 MIT Tech Review",
    "techcrunch": "📰 TechCrunch",
}


def build_news_embed(article: Article, summary: dict) -> discord.Embed:
    color = CATEGORY_COLORS["news"]
    icon = SOURCE_ICONS.get(article.source, article.source)

    embed = discord.Embed(
        title=truncate(article.title, 250),
        url=article.url,
        color=color,
    )

    tldr = summary.get("tldr", "")
    if tldr:
        embed.description = f"**📌 TÓM TẮT**\n{truncate(tldr, 800)}"

    key_points = summary.get("key_points", [])
    if isinstance(key_points, list) and key_points:
        value = "\n".join(f"• {truncate(str(p), 200)}" for p in key_points[:5])
        embed.add_field(name="🔑 ĐIỂM CHÍNH", value=truncate(value, 1000), inline=False)

    context = summary.get("context", "")
    if context and context.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="💡 BỐI CẢNH", value=truncate(context, 900), inline=False)

    impact = summary.get("impact", "")
    if impact and impact.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="🎯 TÁC ĐỘNG", value=truncate(impact, 900), inline=False)

    if article.comments:
        comment_text = "\n".join(f"• {truncate(c, 180)}" for c in article.comments[:4])
        embed.add_field(
            name=f"💬 THẢO LUẬN ({article.metadata.get('comments', 0)} comments)",
            value=truncate(comment_text, 900),
            inline=False,
        )

    tags = summary.get("tags", [])
    if isinstance(tags, list) and tags:
        tag_str = " · ".join(f"`{t}`" for t in tags[:6])
        embed.add_field(name="🏷️ Tags", value=truncate(tag_str, 300), inline=False)

    footer_parts = [icon]
    if article.metadata.get("points"):
        footer_parts.append(f"👍 {article.metadata['points']}")
    if article.metadata.get("comments"):
        footer_parts.append(f"💬 {article.metadata['comments']}")
    if article.published_at:
        footer_parts.append(article.published_at.strftime("%d/%m %H:%M"))

    embed.set_footer(text=" · ".join(footer_parts))
    return embed


def build_paper_embed(article: Article, summary: dict) -> discord.Embed:
    color = CATEGORY_COLORS["papers"]
    icon = SOURCE_ICONS.get(article.source, article.source)

    embed = discord.Embed(
        title=truncate(article.title, 250),
        url=article.url,
        color=color,
    )

    if article.author:
        embed.set_author(name=truncate(article.author, 200))

    tldr = summary.get("tldr", "")
    if tldr:
        embed.description = f"**📌 TÓM TẮT**\n{truncate(tldr, 800)}"

    problem = summary.get("problem", "")
    if problem and problem.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="❓ VẤN ĐỀ", value=truncate(problem, 900), inline=False)

    method = summary.get("method", "")
    if method and method.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="🔬 PHƯƠNG PHÁP", value=truncate(method, 900), inline=False)

    results = summary.get("results", [])
    if isinstance(results, list) and results:
        value = "\n".join(f"• {truncate(str(r), 250)}" for r in results[:5])
        embed.add_field(name="📊 KẾT QUẢ", value=truncate(value, 1000), inline=False)
    elif isinstance(results, str) and results:
        embed.add_field(name="📊 KẾT QUẢ", value=truncate(results, 900), inline=False)

    limitations = summary.get("limitations", "")
    if limitations and limitations.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="⚠️ HẠN CHẾ", value=truncate(limitations, 800), inline=False)

    applications = summary.get("applications", "")
    if applications and applications.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="💼 ỨNG DỤNG", value=truncate(applications, 800), inline=False)

    tags = summary.get("tags", [])
    if isinstance(tags, list) and tags:
        tag_str = " · ".join(f"`{t}`" for t in tags[:6])
        embed.add_field(name="🏷️ Tags", value=truncate(tag_str, 300), inline=False)

    links: list[str] = []
    if article.metadata.get("arxiv_url"):
        links.append(f"[arXiv]({article.metadata['arxiv_url']})")
    if article.metadata.get("pdf_url"):
        links.append(f"[PDF]({article.metadata['pdf_url']})")
    if article.metadata.get("hf_url"):
        links.append(f"[HF]({article.metadata['hf_url']})")
    if links:
        embed.add_field(name="🔗 Links", value=" · ".join(links), inline=False)

    footer_parts = [icon]
    if article.metadata.get("upvotes"):
        footer_parts.append(f"⬆️ {article.metadata['upvotes']}")
    if article.published_at:
        footer_parts.append(article.published_at.strftime("%d/%m %H:%M"))

    embed.set_footer(text=" · ".join(footer_parts))
    return embed


def build_model_embed(article: Article, summary: dict) -> discord.Embed:
    color = CATEGORY_COLORS["models"]
    icon = SOURCE_ICONS.get(article.source, article.source)

    embed = discord.Embed(
        title=truncate(article.title, 250),
        url=article.url,
        color=color,
    )

    tldr = summary.get("tldr", "")
    if tldr:
        embed.description = f"**📌 GIỚI THIỆU**\n{truncate(tldr, 800)}"

    specs = summary.get("specs", {})
    if isinstance(specs, dict) and specs:
        spec_parts = []
        if specs.get("task"):
            spec_parts.append(f"**Task:** {truncate(str(specs['task']), 80)}")
        if specs.get("params") and str(specs["params"]).lower() != "không rõ":
            spec_parts.append(f"**Params:** {truncate(str(specs['params']), 50)}")
        if specs.get("context") and str(specs["context"]).lower() != "không rõ":
            spec_parts.append(f"**Context:** {truncate(str(specs['context']), 50)}")
        if specs.get("license"):
            spec_parts.append(f"**License:** {truncate(str(specs['license']), 50)}")
        if specs.get("library"):
            spec_parts.append(f"**Library:** {truncate(str(specs['library']), 50)}")
        if spec_parts:
            embed.add_field(
                name="📋 THÔNG SỐ",
                value=truncate("\n".join(spec_parts), 1000),
                inline=False,
            )

    strengths = summary.get("strengths", [])
    if isinstance(strengths, list) and strengths:
        value = "\n".join(f"✅ {truncate(str(s), 200)}" for s in strengths[:4])
        embed.add_field(name="ĐIỂM MẠNH", value=truncate(value, 900), inline=False)

    weaknesses = summary.get("weaknesses", [])
    if isinstance(weaknesses, list) and weaknesses:
        value = "\n".join(f"⚠️ {truncate(str(w), 200)}" for w in weaknesses[:4])
        embed.add_field(name="HẠN CHẾ", value=truncate(value, 900), inline=False)

    use_cases = summary.get("use_cases", "")
    if use_cases and use_cases.lower() not in ("không có thông tin trong bài", "không rõ"):
        embed.add_field(name="💼 USE CASE", value=truncate(use_cases, 800), inline=False)

    code = summary.get("code_snippet", "")
    if code and len(code) > 10:
        code_block = f"```python\n{truncate(code, 800)}\n```"
        embed.add_field(name="💻 CÁCH DÙNG", value=code_block, inline=False)

    tags = summary.get("tags", [])
    if isinstance(tags, list) and tags:
        tag_str = " · ".join(f"`{t}`" for t in tags[:6])
        embed.add_field(name="🏷️ Tags", value=truncate(tag_str, 300), inline=False)

    footer_parts = [icon]
    if article.metadata.get("downloads"):
        footer_parts.append(f"⬇️ {article.metadata['downloads']:,}")
    if article.metadata.get("likes"):
        footer_parts.append(f"❤️ {article.metadata['likes']}")
    if article.published_at:
        footer_parts.append(article.published_at.strftime("%d/%m %H:%M"))

    embed.set_footer(text=" · ".join(footer_parts))
    return embed


def build_embed(article: Article, summary: dict) -> discord.Embed:
    if article.category == "papers":
        return build_paper_embed(article, summary)
    if article.category == "models":
        return build_model_embed(article, summary)
    return build_news_embed(article, summary)


def get_channel_id(category: str) -> int:
    if category == "papers":
        return CHANNEL_PAPERS_ID
    if category == "models":
        return CHANNEL_MODELS_ID
    return CHANNEL_NEWS_ID


async def send_log(bot: discord.Client, message: str):
    if not CHANNEL_LOGS_ID:
        return
    try:
        channel = bot.get_channel(CHANNEL_LOGS_ID)
        if channel is None:
            channel = await bot.fetch_channel(CHANNEL_LOGS_ID)
        if channel:
            await channel.send(f"```{truncate(message, 1900)}```")
    except Exception as e:
        logger.error(f"[Log] Failed to send log: {e}")


async def push_article(
    bot: discord.Client,
    db: DatabaseManager,
    article: Article,
    summary: dict,
    locks: dict[int, asyncio.Lock],
) -> None:
    channel_id = get_channel_id(article.category)
    channel = bot.get_channel(channel_id)

    if channel is None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception as e:
            logger.error(f"[Push] Channel {channel_id} not found: {e}")
            await db.mark_failed(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                "channel not found",
            )
            return

    lock = locks.get(channel_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[channel_id] = lock

    async with lock:
        try:
            embed = build_embed(article, summary)
            msg = await channel.send(embed=embed)
            await db.mark_sent(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                channel_id,
                msg.id,
            )
            await asyncio.sleep(1.5)

        except discord.Forbidden:
            error_msg = "Bot lacks permission to send messages in this channel"
            logger.error(f"[Push] {error_msg} ({channel_id})")
            await send_log(bot, f"[Push] {error_msg}")
            await db.mark_failed(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                error_msg,
            )

        except discord.NotFound:
            error_msg = "Channel not found or deleted"
            logger.error(f"[Push] {error_msg} ({channel_id})")
            await db.mark_failed(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                error_msg,
            )

        except discord.HTTPException as e:
            logger.error(f"[Push] Discord HTTP error for {article.title[:60]}: {e}")
            await db.mark_failed(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                str(e),
            )

        except Exception as e:
            logger.exception(f"[Push] Unexpected error for {article.title[:60]}: {e}")
            await db.mark_failed(
                article.source,
                article.external_id,
                article.canonical_url,
                article.url,
                article.title,
                str(e),
            )


class DeniaBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")
        self._lock = asyncio.Lock()
        self.fetchers: list[BaseFetcher] = []
        self.db: DatabaseManager | None = None
        self.mistral: Mistral | None = None
        self.channel_locks: dict[int, asyncio.Lock] = {}

    async def setup_hook(self):
        self.db = DatabaseManager(DB_PATH)
        await self.db.connect()
        self.mistral = Mistral(api_key=MISTRAL_API_KEY)

        self.fetchers = [
            HackerNewsFetcher(),
            HuggingFacePapersFetcher(),
            ArxivFetcher(),
            HuggingFaceModelsFetcher(),
            RSSFetcher(
                name="mit_tech",
                rss_url="https://www.technologyreview.com/topic/artificial-intelligence/feed/",
                category="news",
            ),
            RSSFetcher(
                name="techcrunch",
                rss_url="https://techcrunch.com/category/artificial-intelligence/feed/",
                category="news",
            ),
        ]

        self.scheduler.add_job(
            self.run_pipeline,
            "interval",
            minutes=15,
            id="pipeline",
            max_instances=1,
            next_run_time=datetime.now() + timedelta(seconds=30),
        )
        self.scheduler.start()
        logger.info("🐱 DeniaNewsCatcher ready!")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")

    async def on_disconnect(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        if self.mistral:
            try:
                await self.mistral.close()
            except Exception as e:
                logger.error(f"[Shutdown] Mistral close failed: {e}")
            self.mistral = None
        if self.db:
            await self.db.close()
        logger.info("DeniaNewsCatcher disconnected")

    async def run_pipeline(self):
        if self._lock.locked():
            logger.warning("[Pipeline] Previous run still in progress, skipping")
            return

        async with self._lock:
            logger.info("=" * 60)
            logger.info("[Pipeline] Starting cycle")

            total_sent = 0
            for fetcher in self.fetchers:
                try:
                    logger.info(f"[Pipeline] Fetching from {fetcher.name}")
                    articles = await fetcher.fetch()
                    if not articles:
                        continue

                    new_articles = await filter_new(self.db, articles)
                    if not new_articles:
                        continue

                    logger.info(
                        f"[Pipeline] Summarizing {len(new_articles)} new articles "
                        f"from {fetcher.name}"
                    )
                    summarized = await summarize_articles(self.mistral, new_articles)

                    for article, summary in summarized:
                        await push_article(
                            self, self.db, article, summary, self.channel_locks
                        )
                        total_sent += 1

                except Exception as e:
                    logger.exception(f"[Pipeline] Fetcher {fetcher.name} error: {e}")
                    await send_log(self, f"[Pipeline] {fetcher.name} error: {e}")

            stats = await self.db.get_stats()
            logger.info(
                f"[Pipeline] Done. Sent {total_sent} this cycle. "
                f"Total: {stats['total_sent']}"
            )
            logger.info("=" * 60)


def main():
    errors = validate_config()
    if errors:
        logger.error("Configuration errors:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    bot = DeniaBot()
    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
