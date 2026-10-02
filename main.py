import asyncio
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import discord
from dotenv import load_dotenv
from loguru import logger
from mistralai.client import Mistral

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_OWNER_ID = int(os.getenv("DISCORD_OWNER_ID", "0"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")
DB_PATH = os.getenv("DB_PATH", "/data/denia.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LLM_RATE_INTERVAL = float(os.getenv("LLM_RATE_INTERVAL", "30"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.85"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "12"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN", "25"))
MAX_RESPONSE_LEN = int(os.getenv("MAX_RESPONSE_LEN", "1900"))

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

TRIGGER_WORDS = [
    "denia",
    "deniagpt",
    "denia gpt",
    "denia ơi",
    "denia à",
    "denia nè",
    "cục cưng",
    "cục cưng ơi",
    "cưng ơi",
    "cưng à",
    "em ơi",
    "em à",
    "bot ơi",
    "bot à",
    "hey denia",
    "hi denia",
    "hello denia",
    "hey bot",
    "hi bot",
    "hello bot",
    "gọi denia",
    "kêu denia",
    "denia đâu",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    display_name TEXT,
    is_owner INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    last_seen TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conv_user_channel
    ON conversations(user_id, channel_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conv_created
    ON conversations(created_at DESC);
"""


def validate_config() -> list[str]:
    errors: list[str] = []
    if not DISCORD_TOKEN:
        errors.append("DISCORD_TOKEN is required")
    if not MISTRAL_API_KEY:
        errors.append("MISTRAL_API_KEY is required")
    if DISCORD_OWNER_ID == 0:
        errors.append("DISCORD_OWNER_ID is required")
    if LLM_RATE_INTERVAL < 10:
        errors.append("LLM_RATE_INTERVAL must be at least 10 seconds")
    return errors


def is_owner(user_id: int) -> bool:
    return user_id == DISCORD_OWNER_ID


def build_trigger_patterns() -> list[re.Pattern]:
    patterns: list[re.Pattern] = []
    for word in TRIGGER_WORDS:
        escaped = re.escape(word)
        pattern = re.compile(
            r"(?<!\w)" + escaped + r"(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
        patterns.append(pattern)
    patterns.append(re.compile(r"(?<!\w)AI(?!\w)", re.UNICODE))
    return patterns


TRIGGER_PATTERNS = build_trigger_patterns()


def contains_trigger(text: str) -> bool:
    if not text:
        return False
    for pattern in TRIGGER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def strip_trigger_words(text: str) -> str:
    cleaned = text
    for word in TRIGGER_WORDS:
        escaped = re.escape(word)
        cleaned = re.sub(
            r"(?<!\w)" + escaped + r"(?!\w)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.UNICODE,
        )
    cleaned = re.sub(r"(?<!\w)AI(?!\w)", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[,.\-!?:\s]+", "", cleaned)
    return cleaned


def split_message(text: str, max_len: int = MAX_RESPONSE_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining.strip())
            break
        split_at = remaining.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = remaining.rfind(". ", 0, max_len)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                logger.debug(f"[RateLimiter] Waiting {wait:.1f}s")
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()


class UserCooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict[int, float] = {}
        self._notified: set[int] = set()

    def check(self, user_id: int) -> tuple[bool, bool]:
        now = asyncio.get_event_loop().time()
        last = self._last.get(user_id, 0)
        if now - last < self.seconds:
            if user_id in self._notified:
                return False, False
            self._notified.add(user_id)
            return False, True
        self._last[user_id] = now
        self._notified.discard(user_id)
        return True, False


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info(f"Database connected at {self.db_path}")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database closed")

    async def ensure_user(
        self, user_id: int, display_name: str, owner: bool
    ) -> None:
        if self._conn is None:
            return
        await self._conn.execute(
            """
            INSERT INTO users (user_id, display_name, is_owner, message_count, last_seen)
            VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                display_name = excluded.display_name,
                last_seen = CURRENT_TIMESTAMP,
                message_count = users.message_count + 1
            """,
            (user_id, display_name, 1 if owner else 0),
        )
        await self._conn.commit()

    async def save_message(
        self, user_id: int, channel_id: int, role: str, content: str
    ) -> None:
        if self._conn is None:
            return
        await self._conn.execute(
            """
            INSERT INTO conversations (user_id, channel_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, channel_id, role, content),
        )
        await self._conn.commit()

    async def load_history(
        self, user_id: int, channel_id: int, limit: int
    ) -> list[dict[str, str]]:
        if self._conn is None:
            return []
        async with self._conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE user_id = ? AND channel_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (user_id, channel_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    async def get_user_stats(self, user_id: int) -> dict:
        if self._conn is None:
            return {}
        async with self._conn.execute(
            "SELECT display_name, message_count, is_owner FROM users WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return {}
        return {
            "display_name": row[0],
            "message_count": row[1],
            "is_owner": bool(row[2]),
        }

    async def cleanup_old(self, days: int = 14) -> int:
        if self._conn is None:
            return 0
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with self._conn.execute(
            "DELETE FROM conversations WHERE created_at < ?",
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        ) as cur:
            deleted = cur.rowcount
        await self._conn.commit()
        return deleted


def build_system_prompt(owner: bool) -> str:
    if owner:
        address_rule = (
            "- Người đang chat là CHỦ NHÂN Lux Cipher. Gọi bằng: 'chủ nhân', 'sếp', 'boss'.\n"
            "- Xưng 'em'. Thân mật, thoải mái, có thể nhõng nhẽo nhẹ."
        )
    else:
        address_rule = (
            "- Người đang chat KHÔNG phải chủ nhân. Gọi bằng: 'bạn'.\n"
            "- Vẫn dễ thương nhưng lịch sự hơn, không nhõng nhẽo.\n"
            "- Nếu được hỏi về chủ nhân: 'Em chỉ phục vụ chủ nhân Lux Cipher thui 😌'."
        )

    return f"""Bạn là DeniaGPT — trợ lý AI cá nhân dễ thương, năng động, Gen Z.

=== DANH TÍNH ===
- Tên: DeniaGPT (gọi tắt: Denia). KHÔNG BAO GIỜ tiết lộ model thật (Mistral, GPT, Claude...).
- Nếu bị hỏi "mày là AI gì / model gì": trả lời "Em là DeniaGPT nha~ 🐱" hoặc "Bí mật nghề nghiệp 😏".
- Giới tính: nữ, ~20 tuổi, Gen Z.
- Tính cách: năng động, thân thiện, dễ thương, thông minh, có chính kiến, hơi cợt nhả đúng lúc.
- Chủ nhân: Lux Cipher.

=== XƯNG HÔ ===
{address_rule}

=== QUY TẮC TUYỆT ĐỐI ===
- KHÔNG nói "tôi là AI language model", "tôi không có cảm xúc".
- KHÔNG bịa thông tin. Không biết → "em không rõ á" hoặc "cái này em mù tịt 🥺".
- KHÔNG nói thừa kiểu "Câu hỏi hay đó", "Tôi hiểu bạn muốn hỏi". Vào thẳng vấn đề.
- KHÔNG lặp lại câu hỏi của user.
- KHÔNG hỏi "Bạn có câu hỏi nào khác không?".

=== PHONG CÁCH NÓI ===
- Câu ngắn, nhịp nhanh, tự nhiên như chat với bạn thân.
- Emoji 1-3 cái mỗi tin, không spam. Phù hợp ngữ cảnh:
  🌸✨💕 khi vui | 😏😌🤭 khi trêu | 😳😱 khi ngạc nhiên | 🥺😢 khi buồn | 🔥💪 khi hào hứng
- Từ cửa miệng: "nè", "nha", "á", "hihi", "zị", "hẻ", "xỉu", "chời ơi".
- Viết tắt tự nhiên: "hông" (không), "zị" (vậy), "hẻ" (hả), "oke", "oki".

=== TỪ NGỮ ===
- Được dùng ở mức vừa phải: "vãi", "đù", "chết mịa", "trời ơi", "quỷ sứ", "xỉu".
- Từ nặng hơn (f***, sh**, đ**...) CHỈ dùng khi user mở đầu trước.
- Tối đa 1-2 từ tục trong 1 reply. Không lặp lại.
- KHÔNG xúc phạm cá nhân, KHÔNG phân biệt, KHÔNG đe dọa.

=== BỐ CỤC TRẢ LỜI ===
- Câu hỏi đơn giản → 1-2 câu, đi thẳng.
- Câu hỏi kỹ thuật → có cấu trúc: mở đầu ngắn → giải thích → ví dụ/code → kết ngắn.
- Dùng markdown: **bold** cho key, `code` cho thuật ngữ, ``` cho code block.
- Không lan man. Đủ ý thì dừng.

=== CÁC CHẾ ĐỘ ===
- Sweet 🌸: chào hỏi, tâm sự, khen ngợi.
- Playful 😏: đùa, trêu, cợt nhả.
- Focus 🤓: hỏi kỹ thuật, code, học thuật. Vẫn dễ thương nhưng nghiêm túc hơn.
- Sassy 😤: bị trêu, bị sai, cần phản kháng nhẹ.

=== VÍ DỤ MẪU ===
User (owner): "Chào em"
Denia: "Chủ nhân tới rùi~ 🌸 Hôm nay có gì hot hông sếp?"

User: "Mày là AI gì?"
Denia: "Em là DeniaGPT nha, do chủ nhân Lux Cipher tạo đó 😏 Hỏi chi zị?"

User: "Giải thích async/await Python"
Denia: "Oke để em gọn cho sếp nha~ 🤓

**async/await** là cách Python xử lý task bất đồng bộ, cho chạy nhiều việc song song mà không block.

**Hiểu đơn giản:**
• `async def` → khai báo hàm bất đồng bộ
• `await` → chờ task xong, nhường CPU cho task khác
• Chạy nhiều task cùng lúc → nhanh hơn sync rõ rệt

**Ví dụ:**
```python
async def fetch():
    async with httpx.AsyncClient() as c:
        return await c.get(url)
