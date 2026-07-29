import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DB_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "qa.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    existing = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'agent',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                mode_used TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                ai_suggested_answer TEXT,
                match_score REAL,
                final_answer TEXT,
                claimed_by INTEGER,
                claimed_by_name TEXT,
                answered_by INTEGER,
                answered_by_name TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- 客户端聊天记录明细：客户端每显示一条消息气泡就落库一行，刷新页面时按原样
            -- 逐条重放。conversations 表每条对话只存一行"最终状态"（最终的问题、最终留的
            -- 邮箱等），刷新恢复时如果只靠它"推演"聊天记录，转人工这类多步流程的过程性
            -- 消息（"好的，请您简单描述…"、"人工客服正忙，留下您的邮箱…"等）就会丢失或者
            -- 措辞对不上，看起来像换了一个界面。sort_key 是排序键（不用自增 id 排序，因为
            -- "人工客服正忙"这类提示是延迟插在自己所属提问后面的，落库时间晚于排在它后面
            -- 的消息）；kind 标记气泡类型，刷新重放时据此决定渲染成静态文字还是重新挂上
            -- 可交互的表单/轮询（见 static/index.html 的 restoreHistory）。
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL DEFAULT '',
                sort_key REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- 客户在聊天里发送的图片：上传时就调视觉模型把图片内容转成一段文字描述存在这里，
            -- 等客户接着提问时再把这段描述并进问题一起走正常的题库检索/回答流程（见 web_app.py
            -- 的 /api/ask）。consumed 标记这张图片是否已经被某次提问"用掉"了，避免同一张图片
            -- 反复并进后面每一个问题里。
            CREATE TABLE IF NOT EXISTS pending_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                consumed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_pending_images_session ON pending_images(session_id, consumed);
            CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
            CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, sort_key);
            """
        )

        # 兼容已部署的旧数据库：补充检索标注字段（是否命中/命中的题库问题与答案）
        _add_column_if_missing(conn, "conversations", "matched", "INTEGER")
        _add_column_if_missing(conn, "conversations", "matched_question", "TEXT")
        _add_column_if_missing(conn, "conversations", "matched_answer", "TEXT")
        # 记录客户端 IP，便于客服工作台展示（分组仍以 session_id 为准，IP 仅作参考）
        _add_column_if_missing(conn, "conversations", "client_ip", "TEXT")
        _add_column_if_missing(conn, "conversations", "auto_send_at", "TEXT")
        # 记录本次提问针对的产品（不同产品各自有独立题库），便于客服区分处理
        _add_column_if_missing(conn, "conversations", "product", "TEXT")
        # 未命中转人工等待超过 10 秒后，客户可主动留下邮箱；客服据此通过邮件回复。
        # 客户没有留邮箱则此字段始终为空，不会触发任何邮件。
        _add_column_if_missing(conn, "conversations", "customer_email", "TEXT")
        # 客户明确说"转人工"之后，还没来得及补充说明具体想咨询的问题（question 字段此时
        # 还是"转人工"这句占位文本）：这个标记为 1；客户提交真正问题后（set_question 里会
        # 自动清零）变成 0。用于客户端刷新页面恢复历史记录时，判断这条对话是应该继续显示
        # "请描述您的问题"这个输入框，还是显示"已收到您的问题…"这类等待提示——否则刷新后
        # 客户会永久失去补充问题的入口，只能看到一句不会再变化的等待语。
        _add_column_if_missing(conn, "conversations", "awaiting_transfer_details", "INTEGER DEFAULT 0")
        # 跟 awaiting_transfer_details 不同：这个标记客户提交完问题后不会被清零，会一直保留，
        # 用来记住"这条对话最初是客户主动说'转人工'触发的"，而不是题库未命中被动转过来的。
        # 客户端刷新页面恢复历史记录时要据此选择正确的确认话术（"已将您的问题…更新给人工
        # 客服"而不是"已收到您的问题，正在为您转接人工客服"），否则刷新后会显示成一句跟当时
        # 实际发生的事情不符的话，看起来像是"变成另一个界面"。
        _add_column_if_missing(conn, "conversations", "is_explicit_transfer", "INTEGER DEFAULT 0")
        # 这条提问是否附带了客户发来的图片：image_description 是视觉模型给出的图片文字描述
        # （已并进本次提问一起检索），客服在工作台能看到 AI 到底"看见了什么"，便于复核。
        _add_column_if_missing(conn, "conversations", "image_url", "TEXT")
        _add_column_if_missing(conn, "conversations", "image_description", "TEXT")

        row = conn.execute("SELECT value FROM settings WHERE key = 'global_mode'").fetchone()
        if row is None:
            default_mode = os.getenv("DEFAULT_MODE", "auto")
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('global_mode', ?)", (default_mode,)
            )

        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        exists = conn.execute(
            "SELECT id FROM agents WHERE username = ?", (admin_username,)
        ).fetchone()
        if exists is None:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            password_hash = bcrypt.hashpw(admin_password.encode("utf-8"), bcrypt.gensalt()).decode(
                "utf-8"
            )
            conn.execute(
                "INSERT INTO agents (username, password_hash, display_name, role) VALUES (?, ?, ?, 'admin')",
                (admin_username, password_hash, "管理员"),
            )
            print(
                f"[init] 已创建默认管理员账号: {admin_username} / {admin_password}（请登录后尽快修改密码）"
            )


# ---------- settings ----------

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


# ---------- agents ----------

def create_agent(username: str, password: str, display_name: str, role: str = "agent") -> int:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO agents (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (username, password_hash, display_name, role),
        )
        return cur.lastrowid


def get_agent_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM agents WHERE username = ?", (username,)).fetchone()


def get_agent_by_id(agent_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()


def list_agents() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, role, created_at FROM agents ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def update_agent_password(agent_id: int, new_password: str) -> None:
    password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        conn.execute("UPDATE agents SET password_hash = ? WHERE id = ?", (password_hash, agent_id))


# ---------- conversations ----------

def create_conversation(
    session_id: str,
    question: str,
    mode_used: str,
    client_ip: Optional[str] = None,
    product: Optional[str] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (session_id, question, mode_used, client_ip, product) VALUES (?, ?, ?, ?, ?)",
            (session_id, question, mode_used, client_ip, product),
        )
        return cur.lastrowid


def set_retrieval_info(
    conversation_id: int,
    matched: bool,
    matched_question: Optional[str],
    matched_answer: Optional[str],
    score: float,
) -> None:
    """记录本次提问在题库中的检索结果：命中的问题/答案，或未命中。"""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET matched = ?, matched_question = ?, matched_answer = ?, match_score = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (1 if matched else 0, matched_question, matched_answer, score, conversation_id),
        )


def set_ai_suggestion(conversation_id: int, suggestion: str, score: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET ai_suggested_answer = ?, match_score = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (suggestion, score, conversation_id),
        )


def set_question(conversation_id: int, question: str) -> bool:
    """客户主动说"转人工"之后补充具体问题时，用真实问题内容替换掉"转人工"这句占位提问，
    方便客服在工作台直接看懂客户想咨询什么。同时清掉 awaiting_transfer_details 标记——
    问题已经补充完整，客户端刷新页面时不应该再弹一次"请描述您的问题"的输入框。
    返回 False 表示该对话不存在。"""
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            """
            UPDATE conversations
            SET question = ?, awaiting_transfer_details = 0, updated_at = datetime('now')
            WHERE id = ?
            """,
            (question, conversation_id),
        )
        return True


def set_awaiting_transfer_details(conversation_id: int, awaiting: bool) -> None:
    """客户明确说"转人工"（question 字段还是占位文本）时标记为等待补充问题状态；
    详见 init_db 里 awaiting_transfer_details 列的说明。"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET awaiting_transfer_details = ? WHERE id = ?",
            (1 if awaiting else 0, conversation_id),
        )


def set_is_explicit_transfer(conversation_id: int, value: bool) -> None:
    """标记这条对话最初是客户主动说"转人工"触发的，且这个标记不会随着补充问题而清零；
    详见 init_db 里 is_explicit_transfer 列的说明。"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET is_explicit_transfer = ? WHERE id = ?",
            (1 if value else 0, conversation_id),
        )


def set_customer_email(conversation_id: int, email: str) -> bool:
    """记录客户在"人工客服正忙"提示下主动留下的邮箱。返回 False 表示该对话不存在。"""
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE conversations SET customer_email = ?, updated_at = datetime('now') WHERE id = ?",
            (email, conversation_id),
        )
        return True


def set_auto_send_at(conversation_id: int, auto_send_at: Optional[str]) -> None:
    """记录人机协同模式下 AI 建议自动发送的截止时间（UTC，ISO 格式），供前端倒计时展示。"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET auto_send_at = ? WHERE id = ?",
            (auto_send_at, conversation_id),
        )


def mark_answered(conversation_id: int, final_answer: str, answered_by: Optional[int] = None,
                   answered_by_name: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET final_answer = ?, status = 'answered', answered_by = ?, answered_by_name = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (final_answer, answered_by, answered_by_name, conversation_id),
        )


def get_conversation(conversation_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()


def list_queue() -> list:
    # 已去掉"认领"机制，新对话不会再产生 'claimed' 状态；这里仍然把它当未处理来查，
    # 只是为了兼容旧版本遗留下来、还没处理完的历史数据，不会影响后续新对话。
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM conversations
            WHERE status IN ('pending', 'claimed')
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def list_recent(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_visitor_no_map() -> dict:
    """按 session_id 首次提问时间的先后顺序，给每个会话算出一个稳定的"访客编号"（1、2、3...），
    跟工作台会话列表里看到的编号是同一套规则，供邮件通知等场景批量查询"是哪个客户"。"""
    with get_conn() as conn:
        order_rows = conn.execute(
            "SELECT session_id, MIN(id) AS first_id FROM conversations GROUP BY session_id ORDER BY first_id ASC"
        ).fetchall()
        return {row["session_id"]: idx + 1 for idx, row in enumerate(order_rows)}


def get_visitor_no(session_id: str) -> int:
    """单个会话查访客编号（内部复用 get_visitor_no_map，找不到时返回 0），
    用于邮件通知里标明"是哪个客户"。"""
    return get_visitor_no_map().get(session_id, 0)


def list_sessions(limit: int = 200) -> list:
    """按客户会话（session_id）分组汇总，客服工作台以此实现"一个用户一个对话框"。
    IP 和 session_id 都不适合直接展示给客服（IP 可能拿不到/意义不明，session_id 是随机字符串），
    因此额外分配一个稳定的"访客编号"（按首次提问时间先后顺序，1、2、3...），作为对客服友好的身份标识。"""
    with get_conn() as conn:
        visitor_no_map = get_visitor_no_map()

        rows = conn.execute(
            """
            SELECT
                session_id,
                COUNT(*) AS message_count,
                -- 'claimed' 是已去掉的旧状态值，这里并入 pending 一起统计，兼容历史遗留数据。
                SUM(CASE WHEN status IN ('pending', 'claimed') THEN 1 ELSE 0 END) AS pending_count,
                MAX(created_at) AS last_activity,
                MIN(created_at) AS first_activity
            FROM conversations
            GROUP BY session_id
            ORDER BY last_activity DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        sessions = [dict(r) for r in rows]

        for session in sessions:
            session["visitor_no"] = visitor_no_map.get(session["session_id"], 0)
            last_row = conn.execute(
                """
                SELECT question, client_ip, mode_used, status, product
                FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session["session_id"],),
            ).fetchone()
            if last_row:
                session["last_question"] = last_row["question"]
                session["last_mode"] = last_row["mode_used"]
                session["last_status"] = last_row["status"]
                session["last_product"] = last_row["product"]
            ip_row = conn.execute(
                """
                SELECT client_ip FROM conversations
                WHERE session_id = ? AND client_ip IS NOT NULL AND client_ip != ''
                ORDER BY id DESC LIMIT 1
                """,
                (session["session_id"],),
            ).fetchone()
            session["client_ip"] = ip_row["client_ip"] if ip_row else None

        return sessions


def get_daily_stats(start_utc: str, end_utc: str) -> dict:
    """统计 [start_utc, end_utc) 时间范围内（UTC，'YYYY-MM-DD HH:MM:SS' 格式，与 created_at 一致）的：
    咨询用户数（按 session_id 去重）、总对话条数、转人工请求次数。
    转人工请求的判定口径：全人工模式下的任意问题，或全AI/协同模式下题库未命中
    （matched 为假或未记录）——与是否实际发出邮件提醒无关（邮件仅在客户主动留邮箱后才发）。"""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT session_id) AS user_count,
                COUNT(*) AS conversation_count,
                SUM(
                    CASE
                        WHEN mode_used = 'manual' OR matched = 0 OR matched IS NULL THEN 1
                        ELSE 0
                    END
                ) AS handoff_count
            FROM conversations
            WHERE created_at >= ? AND created_at < ?
            """,
            (start_utc, end_utc),
        ).fetchone()
        return {
            "user_count": row["user_count"] or 0,
            "conversation_count": row["conversation_count"] or 0,
            "handoff_count": row["handoff_count"] or 0,
        }


def list_session_messages(session_id: str) -> list:
    """返回某个用户会话下的全部提问/回复记录，按时间顺序排列，用于客服工作台的连续对话展示。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------- 客户端聊天记录明细（刷新后逐条重放） ----------------


def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    kind: str = "text",
    conversation_id: Optional[int] = None,
    after_id: Optional[int] = None,
) -> int:
    """落库一条客户端消息气泡。after_id 指定"插在某条已有消息后面"（对应前端的
    addMessageAfter——"人工客服正忙"这类延迟出现的提示要插在自己所属提问的下面，而不是
    永远排到最末尾），此时排序键取前后两条消息 sort_key 的中间值；不指定则排在会话最后。"""
    with get_conn() as conn:
        sort_key = None
        if after_id is not None:
            after_row = conn.execute(
                "SELECT sort_key FROM chat_messages WHERE id = ? AND session_id = ?",
                (after_id, session_id),
            ).fetchone()
            if after_row is not None:
                after_key = after_row["sort_key"]
                next_row = conn.execute(
                    "SELECT MIN(sort_key) AS k FROM chat_messages WHERE session_id = ? AND sort_key > ?",
                    (session_id, after_key),
                ).fetchone()
                sort_key = (after_key + next_row["k"]) / 2 if next_row and next_row["k"] is not None else after_key + 1
        if sort_key is None:
            max_row = conn.execute(
                "SELECT COALESCE(MAX(sort_key), 0) AS k FROM chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            sort_key = max_row["k"] + 1
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (session_id, conversation_id, role, kind, content, sort_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, conversation_id, role, kind, content, sort_key),
        )
        return cursor.lastrowid


def update_chat_message(
    message_id: int,
    session_id: str,
    content: Optional[str] = None,
    kind: Optional[str] = None,
    conversation_id: Optional[int] = None,
) -> None:
    """更新一条消息气泡的内容/类型/所属对话。客户端的等待气泡是"先占位、后更新"的
    （比如"AI 正在思考…"最终被答案原地替换），DOM 里同一个气泡对应这里同一行记录。
    带 session_id 条件是为了保证只能改到自己会话里的消息。"""
    sets = []
    params: list = []
    if content is not None:
        sets.append("content = ?")
        params.append(content)
    if kind is not None:
        sets.append("kind = ?")
        params.append(kind)
    if conversation_id is not None:
        sets.append("conversation_id = ?")
        params.append(conversation_id)
    if not sets:
        return
    params.extend([message_id, session_id])
    with get_conn() as conn:
        conn.execute(
            f"UPDATE chat_messages SET {', '.join(sets)} WHERE id = ? AND session_id = ?",
            params,
        )


def list_chat_messages(session_id: str) -> list:
    """按显示顺序返回某个会话的全部消息气泡，供客户端刷新页面时逐条重放。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, role, kind, content FROM chat_messages "
            "WHERE session_id = ? ORDER BY sort_key ASC, id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_session_images(session_id: str) -> list:
    """返回某个会话里客户发送的全部图片气泡（kind=image），供客服工作台在对话流里展示。
    顺带带上视觉模型给出的文字描述（按图片 URL 关联 pending_images），让客服直接看到
    AI 是怎么理解这张图的。"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.id, m.content, m.role, m.created_at,
                   (SELECT p.description FROM pending_images p
                    WHERE p.session_id = m.session_id AND p.url = m.content
                    ORDER BY p.id DESC LIMIT 1) AS description
            FROM chat_messages m
            WHERE m.session_id = ? AND m.kind = 'image'
            ORDER BY m.sort_key ASC, m.id ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------- 客户发送的图片（视觉模型描述 + 与下一次提问合并） ----------------


def add_pending_image(session_id: str, url: str, description: str) -> int:
    """记录一张客户刚发来的图片及其视觉模型文字描述，等待与客户接下来的提问合并。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pending_images (session_id, url, description) VALUES (?, ?, ?)",
            (session_id, url, description),
        )
        return cur.lastrowid


def take_pending_images(session_id: str, max_age_seconds: int = 600) -> list:
    """取出该会话尚未被任何提问用掉、且还没过期的图片描述，并原子地标记为已消费。
    客户"先发图、再打字提问"是两个独立请求，服务端只能靠这张暂存表把两者接起来；
    标记消费是为了避免同一张图片被并进之后的每一个问题里（客户问第二个不相关的问题时，
    还带着上一张图的描述会把检索带偏）。超过 max_age_seconds 的图片视为客户已经不打算
    就它提问了，直接忽略（仍留在表里，供工作台展示描述）。"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, url, description FROM pending_images
            WHERE session_id = ? AND consumed = 0
              AND created_at >= datetime('now', ?)
            ORDER BY id ASC
            """,
            (session_id, f"-{int(max_age_seconds)} seconds"),
        ).fetchall()
        items = [dict(r) for r in rows]
        if items:
            placeholders = ",".join("?" for _ in items)
            conn.execute(
                f"UPDATE pending_images SET consumed = 1 WHERE id IN ({placeholders})",
                [item["id"] for item in items],
            )
        return items


def set_conversation_image(conversation_id: int, image_url: str, image_description: str) -> None:
    """记下这条提问附带的客户图片及其文字描述（描述已并进本次提问一起检索）。"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET image_url = ?, image_description = ? WHERE id = ?",
            (image_url, image_description, conversation_id),
        )
