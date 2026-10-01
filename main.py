import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import discord
import feedparser
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from loguru import logger
from mistralai import Mistral

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
HN_MIN_SCORE = int(os.getenv("HN_MIN_SCORE", "50"))
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "20"))

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

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    channel_id INTEGER,
    message_id INTEGER,
    status TEXT DEFAULT 'sent',
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_created ON seen_articles(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_source ON seen_articles(source);
"""


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
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info(f"Database connected at {self.db_path}")

    async def close(self):
        if self._conn:
            await self._conn.close()
            logger.info("Database connection closed")

    async def is_seen(self, source: str, external_id: str) -> bool:
        async with self._conn.execute(
            "SELECT 1 FROM seen_articles WHERE source = ? AND external_id = ? LIMIT 1",
            (source, external_id),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def mark_sent(
        self,
        source: str,
        external_id: str,
        url: str,
        title: str,
        channel_id: int,
        message_id: int,
    ) -> None:
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_articles
            (source, external_id, url, title, channel_id, message_id, status)
            VALUES (?, ?, ?, ?, ?, ?, 'sent')
            """,
            (source, external_id, url, title, channel_id, message_id),
        )
        await self._conn.commit()

    async def mark_failed(
        self,
        source: str,
        external_id: str,
        url: str,
        title: str,
        error: str,
    ) -> None:
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_articles
            (source, external_id, url, title, status, error)
            VALUES (?, ?, ?, ?, 'failed', ?)
            """,
            (source, external_id, url, title, error[:500]),
        )
        await self._conn.commit()

    async def get_stats(self) -> dict:
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


class BaseFetcher:
    name: str = "base"
    category: str = "news"

    async def fetch(self) -> list[Article]:
        raise NotImplementedError


AI_KEYWORDS = [
    "ai", "llm", "gpt", "machine learning", "deep learning",
    "neural", "transformer", "model", "dataset", "benchmark",
    "openai", "anthropic", "google ai", "meta ai", "huggingface",
    "pytorch", "tensorflow", "diffusion", "rag", "agent",
]


class HackerNewsFetcher(BaseFetcher):
    name = "hackernews"
    category = "news"
    ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"

    async def fetch(self) -> list[Article]:
        params = {
            "tags": "story",
            "numericFilters": f"points>{HN_MIN_SCORE}",
            "hitsPerPage": MAX_ARTICLES_PER_SOURCE,
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
            title = hit.get("title") or ""
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
            text = (hit.get("story_text") or "").strip()

            haystack = f"{title} {text}".lower()
            if not any(kw in haystack for kw in AI_KEYWORDS):
                continue

            articles.append(
                Article(
                    source=self.name,
                    external_id=hit["objectID"],
                    url=url,
                    title=title,
                    content=text or title,
                    author=hit.get("author"),
                    published_at=datetime.fromisoformat(
                        hit["created_at"].replace("Z", "+00:00")
                    ),
                    metadata={
                        "points": hit.get("points", 0),
                        "comments": hit.get("num_comments", 0),
                        "hn_url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    },
                    category=self.category,
                )
            )

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
        for item in data[:MAX_ARTICLES_PER_SOURCE]:
            paper = item.get("paper", {})
            arxiv_id = paper.get("id", "")
            title = paper.get("title", "").strip()
            summary = paper.get("summary", "").strip()

            if not title or not summary:
                continue

            published = None
            if paper.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        paper["publishedAt"].replace("Z", "+00:00")
                    )
                except Exception:
                    published = None

            articles.append(
                Article(
                    source=self.name,
                    external_id=arxiv_id,
                    url=f"https://huggingface.co/papers/{arxiv_id}",
                    title=title,
                    content=summary,
                    author=", ".join(a.get("name", "") for a in paper.get("authors", [])[:3]),
                    published_at=published,
                    metadata={
                        "upvotes": paper.get("upvotes", 0),
                        "arxiv_id": arxiv_id,
                        "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
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
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    dt = None
            else:
                dt = None

            tags = m.get("tags", [])
            pipeline = m.get("pipeline_tag", "unknown")
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)

            content = (
                f"Model: {model_id}\n"
                f"Task: {pipeline}\n"
                f"Tags: {', '.join(tags[:10])}\n"
                f"Downloads: {downloads}\n"
                f"Likes: {likes}\n"
                f"Library: {m.get('library_name', 'unknown')}"
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

            try:
                published = datetime.fromisoformat(entry.published.replace("Z", "+00:00"))
            except Exception:
                published = None

            articles.append(
                Article(
                    source=self.name,
                    external_id=arxiv_id,
                    url=entry.id,
                    title=title,
                    content=summary,
                    author=", ".join(a.name for a in entry.get("authors", [])[:3]),
                    published_at=published,
                    metadata={"arxiv_id": arxiv_id},
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
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.rss_url)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return []

        articles: list[Article] = []
        for entry in feed.entries[:15]:
            title = entry.get("title", "").strip()
            url = entry.get("link", "")
            summary = entry.get("summary", "") or entry.get("description", "")

            summary = re.sub(r"<[^>]+>", "", summary).strip()

            if not title or not url:
                continue

            try:
                published = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
            except Exception:
                published = None

            articles.append(
                Article(
                    source=self.name,
                    external_id=entry.get("id", url),
                    url=url,
                    title=title,
                    content=summary or title,
                    published_at=published,
                    category=self.category,
                )
            )

        logger.info(f"[{self.name}] Fetched {len(articles)} items")
        return articles


SYSTEM_PROMPT = """Bạn là DeniaNewsCatcher - chuyên gia AI tóm tắt tin tức, paper, và model.
Nhiệm vụ: tóm tắt nội dung bằng tiếng Việt, CHỈ dùng thông tin từ nội dung gốc.
Nếu không có thông tin, ghi "không rõ". KHÔNG bịa. Trả về JSON hợp lệ, không thêm text ngoài JSON."""

PAPER_PROMPT = """Tóm tắt paper AI sau, trả về JSON:

{{
  "tldr": "2-3 câu tóm tắt cốt lõi",
  "problem": "Vấn đề paper giải quyết",
  "method": "Phương pháp chính (ngắn gọn)",
  "results": "Kết quả nổi bật, có số liệu nếu có",
  "why_matters": "Vì sao đáng quan tâm",
  "tags": ["tag1", "tag2", "tag3"]
}}

Tiêu đề: {title}
Nội dung: {content}"""

MODEL_PROMPT = """Tóm tắt model HuggingFace sau, trả về JSON:

{{
  "tldr": "2-3 câu giới thiệu model",
  "task": "Task chính",
  "strengths": ["điểm mạnh 1", "điểm mạnh 2"],
  "weaknesses": ["điểm yếu nếu có"],
  "use_case": "Dùng khi nào",
  "tags": ["tag1", "tag2"]
}}

Tên model: {title}
Thông tin: {content}"""

NEWS_PROMPT = """Tóm tắt tin tức AI sau, trả về JSON:

{{
  "tldr": "1-2 câu tóm tắt",
  "key_points": ["điểm 1", "điểm 2", "điểm 3"],
  "why_matters": "Vì sao quan trọng",
  "tags": ["tag1", "tag2"]
}}

Tiêu đề: {title}
Nội dung: {content}"""


def get_prompt(category: str) -> str:
    if category == "papers":
        return PAPER_PROMPT
    if category == "models":
        return MODEL_PROMPT
    return NEWS_PROMPT


async def summarize(client: Mistral, title: str, content: str, category: str) -> dict | None:
    content = content[:6000]
    prompt = get_prompt(category).format(title=title, content=content)

    try:
        resp = await client.chat.complete_async(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] Invalid JSON for '{title[:60]}': {e}")
        return None
    except KeyError as e:
        logger.error(f"[LLM] Missing key in response for '{title[:60]}': {e}")
        return None
    except Exception as e:
        logger.error(f"[LLM] Summarize failed for '{title[:60]}': {e}")
        return None


async def filter_new(db: DatabaseManager, articles: list[Article]) -> list[Article]:
    new_articles: list[Article] = []
    for art in articles:
        if await db.is_seen(art.source, art.external_id):
            continue
        new_articles.append(art)

    skipped = len(articles) - len(new_articles)
    if skipped:
        logger.info(f"[Dedup] Skipped {skipped} seen articles")
    return new_articles


async def summarize_articles(
    client: Mistral, articles: list[Article]
) -> list[tuple[Article, dict]]:
    semaphore = asyncio.Semaphore(5)

    async def _one(art: Article):
        async with semaphore:
            if len(art.content) < 100:
                return art, {
                    "tldr": art.title,
                    "key_points": [],
                    "why_matters": "",
                    "tags": [],
                }
            summary = await summarize(client, art.title, art.content, art.category)
            return art, summary

    results = await asyncio.gather(*[_one(a) for a in articles])
    valid = [(a, s) for a, s in results if s is not None]
    logger.info(f"[Summarize] Summarized {len(valid)}/{len(articles)} articles")
    return valid


CATEGORY_COLORS = {
    "news": 0x3498DB,
    "papers": 0x9B59B6,
    "models": 0xE67E22,
}

SOURCE_ICONS = {
    "hackernews": "🟠 Hacker News",
    "hf_papers": "🤗 HF Papers",
    "hf_models": "🤗 HF Models",
    "arxiv": "📄 arXiv",
    "mit_tech": "📰 MIT Tech Review",
    "techcrunch": "📰 TechCrunch",
}


def build_embed(article: Article, summary: dict) -> discord.Embed:
    icon = SOURCE_ICONS.get(article.source, article.source)
    color = CATEGORY_COLORS.get(article.category, 0x95A5A6)

    embed = discord.Embed(
        title=article.title[:250],
        url=article.url,
        color=color,
    )

    tldr = summary.get("tldr", "")
    if tldr:
        embed.description = f"**TL;DR:** {tldr[:400]}"

    if article.category == "papers":
        if summary.get("problem"):
            embed.add_field(name="🎯 Vấn đề", value=summary["problem"][:1000], inline=False)
        if summary.get("method"):
            embed.add_field(name="🔬 Phương pháp", value=summary["method"][:1000], inline=False)
        if summary.get("results"):
            embed.add_field(name="📊 Kết quả", value=summary["results"][:1000], inline=False)
        if summary.get("why_matters"):
            embed.add_field(name="⭐ Vì sao quan trọng", value=summary["why_matters"][:1000], inline=False)

    elif article.category == "models":
        if summary.get("task"):
            embed.add_field(name="📋 Task", value=summary["task"][:200], inline=True)
        if summary.get("strengths"):
            val = "\n".join(f"• {s}" for s in summary["strengths"][:4])
            embed.add_field(name="✅ Điểm mạnh", value=val[:1000], inline=False)
        if summary.get("weaknesses"):
            val = "\n".join(f"• {s}" for s in summary["weaknesses"][:4])
            embed.add_field(name="⚠️ Điểm yếu", value=val[:1000], inline=False)

    else:
        points = summary.get("key_points", [])
        if points:
            val = "\n".join(f"• {p}" for p in points[:5])
            embed.add_field(name="📌 Điểm chính", value=val[:1000], inline=False)
        if summary.get("why_matters"):
            embed.add_field(name="⭐ Vì sao quan trọng", value=summary["why_matters"][:1000], inline=False)

    tags = summary.get("tags", [])
    if tags:
        embed.add_field(name="🏷️ Tags", value=" · ".join(tags[:6]), inline=False)

    footer_parts = [icon]
    if article.metadata.get("points"):
        footer_parts.append(f"👍 {article.metadata['points']}")
    if article.metadata.get("comments"):
        footer_parts.append(f"💬 {article.metadata['comments']}")
    if article.metadata.get("upvotes"):
        footer_parts.append(f"⬆️ {article.metadata['upvotes']}")
    if article.metadata.get("downloads"):
        footer_parts.append(f"⬇️ {article.metadata['downloads']}")
    if article.published_at:
        footer_parts.append(article.published_at.strftime("%d/%m %H:%M"))

    embed.set_footer(text=" · ".join(footer_parts))
    return embed


def get_channel_id(category: str) -> int:
    if category == "papers":
        return CHANNEL_PAPERS_ID
    if category == "models":
        return CHANNEL_MODELS_ID
    return CHANNEL_NEWS_ID


async def send_log(bot: discord.Client, message: str):
    try:
        channel = bot.get_channel(CHANNEL_LOGS_ID)
        if channel:
            await channel.send(f"```{message}```")
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
            await send_log(bot, f"[Push] Channel {channel_id} not found: {e}")
            await db.mark_failed(article.source, article.external_id, article.url, article.title, "channel not found")
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
                article.url,
                article.title,
                channel_id,
                msg.id,
            )
            await asyncio.sleep(1.2)
        except Exception as e:
            logger.error(f"[Push] Failed to send {article.title[:60]}: {e}")
            await send_log(bot, f"[Push] Failed to send {article.title[:60]}: {e}")
            await db.mark_failed(article.source, article.external_id, article.url, article.title, str(e))


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
            HuggingFaceModelsFetcher(),
            ArxivFetcher(),
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
        )
        self.scheduler.start()
        logger.info("🐱 DeniaNewsCatcher ready!")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        asyncio.create_task(self.run_pipeline())

    async def on_disconnect(self):
        if self.db:
            await self.db.close()
        if self.mistral:
            await self.mistral.close()
        logger.info("DeniaNewsCatcher disconnected")

    async def run_pipeline(self):
        if self._lock.locked():
            logger.warning("[Pipeline] Previous run still in progress, skipping")
            return

        async with self._lock:
            logger.info("=" * 50)
            logger.info("[Pipeline] Starting cycle")

            total_sent = 0
            for fetcher in self.fetchers:
                try:
                    articles = await fetcher.fetch()
                    if not articles:
                        continue

                    new_articles = await filter_new(self.db, articles)
                    if not new_articles:
                        continue

                    summarized = await summarize_articles(self.mistral, new_articles)

                    for article, summary in summarized:
                        await push_article(self, self.db, article, summary, self.channel_locks)
                        total_sent += 1

                except Exception as e:
                    logger.exception(f"[Pipeline] Fetcher {fetcher.name} error: {e}")
                    await send_log(self, f"[Pipeline] Fetcher {fetcher.name} error: {e}")

            stats = await self.db.get_stats()
            logger.info(f"[Pipeline] Done. Sent {total_sent} this cycle. Total: {stats['total_sent']}")


def main():
    bot = DeniaBot()
    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
