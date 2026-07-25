import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from auth import create_token, decode_token, get_current_agent, require_admin
from doc_rag_chatbot import DocRagBot
from email_utils import DEFAULT_NOTIFY_EMAIL_TO, send_email, send_test_email
from excel_rag_chatbot import AnswerResult, ExcelFaqRagBot
from ws_manager import manager

DEFAULT_EXCEL = "2026.01.26_肤润康-常见咨询问题_v2(1).xls"
UREA_DOC = "urea_hand_cream_info.md"
DEFAULT_MODEL = "qwen3.6-flash"
# 所有邮件通知的主题统一带上品牌名，方便客服在收件箱里一眼认出是哪个客服系统发的。
BRAND_NAME = "YUNI"
VALID_MODES = {"auto", "manual", "collab"}
MODE_LABELS = {"auto": "全AI模式", "manual": "全人工模式", "collab": "人机协同模式"}
DEFAULT_COLLAB_AUTO_SEND_SECONDS = 5
AUTO_SEND_AGENT_NAME = "AI自动发送"
DEFAULT_REMINDER_INTERVAL_MINUTES = 30
REMINDER_TICK_SECONDS = 30

# 每日数据日报：默认每天香港时间 09:00 推送前一个香港日历日（00:00~24:00）的统计数据。
HK_TZ = ZoneInfo("Asia/Hong_Kong")
DEFAULT_DAILY_REPORT_TIME = "09:00"
DAILY_REPORT_TICK_SECONDS = 30

# 两款产品用了两种不同的知识来源：
# - 杜鹃花酸乳霜：现成的"问题-答案"题库（Excel），专业内容原文照搬，逐字不改写。
# - 10%尿素护手霜：暂无题库，只有一份产品说明文档（doc），没有固定问答对，
#   AI 需要现场组织语言回答，但内容必须严格限定在文档范围内——文档没提到的内容
#   （如孕妇能否使用等）一律视为未命中，与题库场景未命中时的转人工规则完全一致。
PRODUCTS: dict = {
    "azelaic_cream": {"label": "YUNI 杜鹃花酸乳霜", "excel": DEFAULT_EXCEL},
    "urea_hand_cream": {"label": "YUNI 10%尿素护手霜", "doc": UREA_DOC},
}
DEFAULT_PRODUCT = "azelaic_cream"
NO_KB_TEXT = "亲，这款产品的常见问题库还在整理中，已为您转接人工客服，请稍候~"


class AskRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    product: Optional[str] = None


class AskResponse(BaseModel):
    conversation_id: int
    status: str  # 'answered' | 'pending'
    answer: Optional[str] = None
    mode: str
    # 仅在 status='pending' 且为人机协同模式时可能为 True：
    # 表示题库已命中、AI 已生成建议并已安排自动发送倒计时，客户此时应看到"AI 思考中"而不是"转人工"提示。
    matched: bool = False
    # 仅在 status='pending' 时可能为 True：客户直接说了"转人工"之类明确要求转接真人，
    # 客户端应立即展示"请描述具体问题 + 留邮箱（选填）"的入口，不必再经历"AI 思考/等待 10 秒"这段过程。
    need_transfer_details: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class AnswerRequest(BaseModel):
    answer: str


class ModeRequest(BaseModel):
    mode: str


class CollabTimeoutRequest(BaseModel):
    seconds: float


class ReminderSettingsRequest(BaseModel):
    enabled: bool
    interval_minutes: float


class DailyReportSettingsRequest(BaseModel):
    enabled: bool
    # 格式 "HH:MM"，按香港时间（Asia/Hong_Kong）计算
    time: str


class NotifyEmailRequest(BaseModel):
    email: str


class LeaveEmailRequest(BaseModel):
    email: str


class TransferQuestionRequest(BaseModel):
    # 客户主动说"转人工"之后，需要客户再描述一次具体想咨询的问题（"转人工"本身不是一句
    # 有实际内容的提问，客服光看这几个字不知道要处理什么）；邮箱选填，留下真实格式的邮箱
    # 才会真正发邮件通知客服，不留邮箱也不影响问题内容的更新和客服在工作台实时看到。
    question: str
    email: Optional[str] = None


class IrrelevantFilterSettingsRequest(BaseModel):
    enabled: bool


class ChatMessageCreateRequest(BaseModel):
    # 客户端每显示一条消息气泡就调一次这个接口落库，刷新页面时按原样逐条重放（见
    # /api/chat-messages 两个端点的说明）。after_id 对应前端的 addMessageAfter：延迟出现
    # 的提示（如 10 秒后的"人工客服正忙"）要插在自己所属提问后面，不是排到最末尾。
    session_id: str
    role: str
    content: str = ""
    kind: str = "text"
    conversation_id: Optional[int] = None
    after_id: Optional[int] = None


class ChatMessageUpdateRequest(BaseModel):
    # 等待类气泡是"先占位、后原地更新"的（"AI 正在思考…"最终被答案替换），DOM 里同一个
    # 气泡始终对应同一行记录；session_id 用来保证只能改到自己会话里的消息。
    session_id: str
    content: Optional[str] = None
    kind: Optional[str] = None
    conversation_id: Optional[int] = None


class SmtpSettingsRequest(BaseModel):
    host: str
    port: int = 587
    username: Optional[str] = None
    # 密码留空表示"不修改已保存的密码"，避免每次改其他字段都要重新输入一遍密码。
    password: Optional[str] = None
    sender: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False


class SmtpTestRequest(BaseModel):
    to: Optional[str] = None


class CreateAgentRequest(BaseModel):
    username: str
    password: str
    display_name: str
    role: Optional[str] = "agent"


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


app = FastAPI(title="Excel FAQ AI 客服机器人")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# 每个有题库的产品各自一个 bot 实例；没有题库的产品不在此字典中出现。
bots: dict = {}


@app.on_event("startup")
async def startup() -> None:
    database.init_db()

    top_k = int(os.getenv("FAQ_TOP_K", "8"))
    min_score = float(os.getenv("FAQ_MIN_SCORE", "0.1"))

    for product_id, meta in PRODUCTS.items():
        excel_name = meta.get("excel")
        doc_name = meta.get("doc")

        if excel_name:
            default_path = str(BASE_DIR / excel_name)
            # 兼容旧的 FAQ_EXCEL_PATH 环境变量，仅对默认产品（杜鹃花酸乳霜）生效
            excel_path = os.getenv("FAQ_EXCEL_PATH", default_path) if product_id == DEFAULT_PRODUCT else default_path
            if not Path(excel_path).exists():
                print(f"[warn] 产品「{meta['label']}」配置的题库文件不存在，跳过: {excel_path}", file=sys.stderr)
                continue
            excel_bot = ExcelFaqRagBot(excel_path=excel_path, top_k=top_k, min_score=min_score)
            excel_bot.build_index()
            bots[product_id] = excel_bot
        elif doc_name:
            doc_path = str(BASE_DIR / doc_name)
            if not Path(doc_path).exists():
                print(f"[warn] 产品「{meta['label']}」配置的说明文档不存在，跳过: {doc_path}", file=sys.stderr)
                continue
            doc_bot = DocRagBot(doc_path=doc_path, top_k=4, min_score=float(os.getenv("DOC_MIN_SCORE", "0.15")))
            doc_bot.build_index()
            bots[product_id] = doc_bot

    # 品牌类问题（如"是澳洲品牌？"）会被标记为 shared=True，属于跨产品共用问题：
    # 无论客户当前选的是哪款产品，都应该能命中——即便该产品自己还没有专属题库。
    shared_items = [it for b in bots.values() for it in b.items if it.shared]
    if shared_items:
        for product_id in PRODUCTS:
            existing_bot = bots.get(product_id)
            if existing_bot is not None:
                existing_bot.load_extra_items(shared_items)
            else:
                bots[product_id] = ExcelFaqRagBot.from_items(shared_items, top_k=top_k, min_score=min_score)

    asyncio.create_task(_reminder_loop())
    asyncio.create_task(_daily_report_loop())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent")
def agent_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "agent.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "products": {pid: len(b.items) for pid, b in bots.items()},
    }


@app.get("/api/products")
def get_products() -> dict:
    return {
        "items": [
            {"id": pid, "label": meta["label"], "has_kb": pid in bots}
            for pid, meta in PRODUCTS.items()
        ],
        "default": DEFAULT_PRODUCT,
    }


def _normalize_product(product: Optional[str]) -> str:
    return product if product in PRODUCTS else DEFAULT_PRODUCT


@app.get("/api/irrelevant-filter-settings")
def get_irrelevant_filter_settings() -> dict:
    enabled = database.get_setting("skip_irrelevant_enabled", "true") == "true"
    return {"enabled": enabled}


@app.post("/api/irrelevant-filter-settings")
def set_irrelevant_filter_settings(
    req: IrrelevantFilterSettingsRequest, agent: dict = Depends(require_admin)
) -> dict:
    database.set_setting("skip_irrelevant_enabled", "true" if req.enabled else "false")
    return {"enabled": req.enabled}


# 客户提问前的单纯寒暄/打招呼（"你好""在吗"之类），不是真正的咨询问题，不应该被判定为
# "题库未命中"而转人工/发提醒邮件——直接由 AI 打个招呼、引导客户说出具体问题即可。
# 仅匹配"整条消息都是寒暄用语"的情况；只要寒暄后面还带了具体问题（比如"你好，能天天用吗"），
# 就不会命中这里，会正常进入各模式原本的题库检索流程。
_GREETING_ONLY_PATTERN = re.compile(
    r"^[\s，,。.！!？?~～]*"
    r"(你好|您好|哈喽|哈啰|hi|hello|hey|在吗|在么|在不在|有人吗|有人在吗|"
    r"有客服吗|客服在吗|请问有人吗|早上好|上午好|中午好|下午好|晚上好)"
    r"[\s，,。.！!？?~～]*$",
    re.IGNORECASE,
)


def _is_pure_greeting(text: str) -> bool:
    return bool(_GREETING_ONLY_PATTERN.match(text.strip()))


# 客户直接说"转人工""人工客服"之类，是明确要求转接真人、不是一句需要检索/AI 回答的正常问题——
# 不应该走题库检索，更不应该被"无关闲聊"AI 判断误判掉（比如被当成与产品无关而回一句引导语）。
# 命中后直接进入"转人工需要留邮箱"流程：不立即发邮件提醒客服，只有客户真的填了有效邮箱才通知，
# 与题库未命中等太久后的留邮箱入口共用同一条规则（不留邮箱就不会触发任何邮件）。
_EXPLICIT_TRANSFER_PATTERN = re.compile(
    r"(转人工|转接人工|转真人|人工客服|真人客服|找人工|找客服|人工坐席|接入人工|人工服务|人工帮我|"
    r"human agent|talk to (a )?human|real person)",
    re.IGNORECASE,
)


def _is_explicit_transfer_request(text: str) -> bool:
    return bool(_EXPLICIT_TRANSFER_PATTERN.search(text.strip()))


# 骂人/人身攻击这类内容，字数往往很短（"你是傻子"这种四五个字），单靠 AI 语义判断不够稳：
# 真实发生过被误判成"命中题库、匹配度100%"直接当成正常问题回复的 bug（短文本的语义检索容易
# 凑巧跟某条无关的题库内容算出很高的相似度）。这里用关键词硬规则兜底，命中就直接判定为无关，
# 不依赖 AI 调用是否成功、判断是否到位；只有关键词没命中的模糊情况，才交给 AI 做语义判断。
_ABUSIVE_PATTERN = re.compile(
    r"(傻[子瓜逼比]|笨蛋|蠢货|蠢[比货]|智障|脑残|白痴|废物|滚开|你妈|尼玛|贱人|婊子|"
    r"操你|去死|fuck|shit|bitch|asshole|f\*ck|stfu)",
    re.IGNORECASE,
)


def _is_abusive_language(text: str) -> bool:
    return bool(_ABUSIVE_PATTERN.search(text.strip()))


# 判断客户填写的邮箱是否是"看起来真的邮箱"（而不是随便打几个字符），只有格式合法才会真正
# 发邮件通知客服；宁可拒绝一个格式有问题的邮箱，也不要发一封没法送达/客服没法回复的邮件。
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _is_valid_email(text: str) -> bool:
    return bool(_EMAIL_PATTERN.match(text.strip()))


def _greeting_reply(product: str) -> str:
    label = PRODUCTS.get(product, {}).get("label", "")
    if label:
        return f"您好，我是「{label}」的 AI 客服，请问有什么想了解的呢？您可以直接告诉我想咨询的问题，我马上为您查询～"
    return f"您好，我是 {BRAND_NAME} AI 客服，请问有什么想了解的呢？您可以直接告诉我想咨询的问题，我马上为您查询～"


# 与产品咨询完全无关的闲聊/荒谬提问（"吃饭了吗""这产品对国家安全有危害吗"之类），不应该像
# 正常问题一样转人工——但这类说法千变万化，没法像打招呼一样靠关键词穷举，需要 AI 做语义判断。
# 出于风险控制，只有 AI 非常确信"完全无关"时才会判定为 True；任何模糊情况或调用异常都返回
# False，退回到原有的检索/转人工流程，避免真实的客户问题被误判成"无关"而悄悄漏单。
# 管理员可在工作台通过 skip_irrelevant_enabled 设置随时关闭这个判断，一键退回"全部按正常问题处理"。
def _classify_irrelevant(question: str, product: str, model: Optional[str]) -> bool:
    try:
        from openai import OpenAI
    except ImportError:
        return False

    product_label = PRODUCTS.get(product, {}).get("label", "该产品")
    system_prompt = (
        f"你是电商客服的预处理模块，只做一件事：判断客户这句话是否与「{product_label}」的产品咨询完全无关，"
        "或者是明显不构成真实客服需求的无聊/挑逗性/不当提问。\n"
        "包括四类：\n"
        "1. 纯粹的日常闲聊/寒暄，不构成真正的问题，例如：吃饭了吗、天气怎么样、讲个笑话、你多大了。\n"
        "2. 与产品/护肤/使用场景毫无关系的荒谬、挑衅性、无意义提问，例如：这个产品对国家安全有危害吗、你支持谁当总统。\n"
        "3. 字面上就不是在向官方客服寻求正常帮助的挑逗性/恶作剧式提问，例如：向官方客服问哪里能买到假货/仿品"
        "（注意这和'怎么辨别真伪''在哪买正品才不会买到假货'这类关于防伪、正品渠道的正常疑虑完全不同，"
        "后者必须判定为相关，只有字面上就是在问'哪里能买到假货本身'才算这一类）。\n"
        "4. 内容低俗/色情/脏话骂人/人身攻击、明显违法违规、或涉及政治敏感/国家安全等违禁话题的提问或言论，"
        "例如：你是傻子、你们客服都是废物、骂人的话、色情低俗内容。"
        "不管是否提到产品，只要字面内容本身带有这类不当性质就算这一类。\n"
        "只有在你非常确信客户这句话完全不构成真实客服需求时，才判定为无关；只要哪怕有一点点可能是在问"
        "产品本身、成分、功效、使用方法、适用人群、购买渠道、防伪辨别、售后等内容，就必须判定为相关，"
        "不确定的一律判定为相关——宁可放过，不可错判（第 4 类不当内容除外，只要命中就必须判定为无关，"
        "不能因为顺带提到了产品就放过）。\n"
        '只输出如下 JSON，不要输出任何其他文字：{"irrelevant": true 或 false}'
    )
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=15.0,
            max_retries=1,
        )
        resp = client.chat.completions.create(
            model=_ai_model(model),
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return False
        payload = json.loads(match.group())
        return bool(payload.get("irrelevant", False))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 无关提问识别调用失败，按正常问题处理: {exc}", file=sys.stderr)
        return False


def _is_irrelevant_question(question: str, product: str, model: Optional[str]) -> bool:
    """判断是否该按"无关闲聊"处理：先用关键词硬规则拦截明显的骂人/人身攻击（见
    _is_abusive_language 的说明），命中就直接判定为无关，不必等 AI 判断；关键词没命中的
    模糊情况，才交给 AI 做语义判断（原有的三类闲聊/荒谬提问/恶作剧式提问逻辑不变）。"""
    if _is_abusive_language(question):
        return True
    return _classify_irrelevant(question, product, model)


def _irrelevant_reply(product: str) -> str:
    label = PRODUCTS.get(product, {}).get("label", "")
    if label:
        return f"这个问题好像和「{label}」的产品咨询没有太大关系呢，如果您有产品使用、成分、购买等相关问题，欢迎随时告诉我，我马上为您查询～"
    return "这个问题好像和产品咨询没有太大关系呢，如果您有产品使用、成分、购买等相关问题，欢迎随时告诉我，我马上为您查询～"


def _ai_model(model: Optional[str]) -> str:
    return model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _generate_ai_reply(product: str, question: str, model: Optional[str]) -> AnswerResult:
    product_bot = bots.get(product)
    if product_bot is None:
        # 该产品还没有题库，直接判定未命中，不消耗 AI 调用
        return AnswerResult(text=NO_KB_TEXT, matched=False, score=0.0)
    return product_bot.answer(
        question,
        model=_ai_model(model),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _client_ip(request: Request) -> Optional[str]:
    """获取访客真实 IP：优先取反向代理头（若未来接入 nginx/CDN），否则取连接的源地址。
    仅用于客服工作台展示参考，不作为区分用户的依据（同一 IP 下可能有多个真实客户）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None


async def _auto_send_after_timeout(conversation_id: int, ai_answer: str, timeout: float) -> None:
    """人机协同模式下，若客服在超时时间内还没有认领处理，则自动把 AI 建议发送给客户，
    避免客户长时间等待；一旦客服已认领（状态不再是 pending），则尊重人工处理，不再自动发送。"""
    try:
        await asyncio.sleep(timeout)
        conversation = database.get_conversation(conversation_id)
        if conversation is None or conversation["status"] != "pending":
            return
        database.mark_answered(conversation_id, ai_answer, answered_by=None, answered_by_name=AUTO_SEND_AGENT_NAME)
        await manager.broadcast({"type": "answered", "id": conversation_id})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 协同模式自动发送失败: {exc}", file=sys.stderr)


def _notify_recipient() -> Optional[str]:
    """收件邮箱以后台「工作台设置」里配置的为准；管理员还没在后台配置过时，
    返回 None，交给 email_utils.send_email 自己回退到环境变量 NOTIFY_EMAIL_TO 的默认值。"""
    email = database.get_setting("notify_email_to")
    return email.strip() if email and email.strip() else None


def _smtp_overrides() -> dict:
    """SMTP 发信配置以后台「工作台设置」里保存的为准；管理员还没在后台配置过（没填服务器地址）时，
    返回空字典，交给 email_utils 自己回退到环境变量 SMTP_* 的默认值——这样服务器换了新环境、
    还没来得及在后台配置之前，.env 里的旧配置依然可以继续工作，不会突然失效。"""
    host = database.get_setting("smtp_host")
    if not host:
        return {}
    return {
        "host": host,
        "port": int(database.get_setting("smtp_port", "587") or "587"),
        "username": database.get_setting("smtp_username") or None,
        "password": database.get_setting("smtp_password") or None,
        "sender": database.get_setting("smtp_from") or None,
        "use_tls": database.get_setting("smtp_use_tls", "true") == "true",
        "use_ssl": database.get_setting("smtp_use_ssl", "false") == "true",
    }


async def _notify_customer_email_left(
    conversation_id: int, product: str, question: str, mode: str, customer_email: str, visitor_no: int = 0
) -> None:
    """客户在"人工客服正忙"提示下主动留下了邮箱：额外发一封邮件告知客服，
    客服可直接通过该邮箱回复客户，而不必等客户重新打开网页查看。
    客户不留邮箱则此函数完全不会被调用，不会触发任何邮件。"""
    product_label = PRODUCTS.get(product, {}).get("label", product or "未知产品")
    mode_label = MODE_LABELS.get(mode, mode)
    visitor_label = f"访客{visitor_no}" if visitor_no else "未知访客"
    subject = f"【{BRAND_NAME} 客服提醒】{visitor_label}留下邮箱待人工回复（对话 #{conversation_id}）"
    body = (
        f"客户：{visitor_label}\n"
        f"产品：{product_label}\n"
        f"工作模式：{mode_label}\n"
        f"客户提问：{question}\n"
        f"客户邮箱：{customer_email}\n"
        f"对话编号：#{conversation_id}\n\n"
        f"客户因等待较久，主动留下了邮箱，请客服直接通过邮件回复客户。\n"
    )
    try:
        await asyncio.to_thread(send_email, subject, body, _notify_recipient(), **_smtp_overrides())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 客户留邮箱提醒邮件发送失败: {exc}", file=sys.stderr)


async def _reminder_loop() -> None:
    """后台定时提醒：管理员可在工作台开启/关闭，并设置提醒间隔（分钟）。
    开启后，每到设定的间隔就统计一次当前待处理队列（按客户/session_id 分组），
    通过邮件汇总提醒有多少客户、多少条问题待处理；队列为空时不打扰，只更新计时。"""
    while True:
        try:
            await asyncio.sleep(REMINDER_TICK_SECONDS)
            enabled = database.get_setting("reminder_enabled", "false") == "true"
            if not enabled:
                continue

            interval_minutes = float(
                database.get_setting("reminder_interval_minutes", str(DEFAULT_REMINDER_INTERVAL_MINUTES))
            )
            last_sent_raw = database.get_setting("reminder_last_sent_at")
            now = datetime.utcnow()
            if last_sent_raw:
                try:
                    last_sent = datetime.strptime(last_sent_raw, "%Y-%m-%dT%H:%M:%SZ")
                    if (now - last_sent).total_seconds() < interval_minutes * 60:
                        continue
                except ValueError:
                    pass

            queue = database.list_queue()
            if queue:
                customer_count = len({item["session_id"] for item in queue})
                question_count = len(queue)
                visitor_no_map = database.get_visitor_no_map()
                subject = f"【{BRAND_NAME} 客服定时提醒】当前有 {customer_count} 位客户、{question_count} 个问题待处理"
                lines = [subject, ""]
                for item in queue:
                    label = PRODUCTS.get(item["product"], {}).get("label", item["product"] or "未知产品")
                    visitor_no = visitor_no_map.get(item["session_id"], 0)
                    visitor_label = f"访客{visitor_no}" if visitor_no else "未知访客"
                    lines.append(f"- 对话 #{item['id']}（{visitor_label} · {label}）：{item['question']}")
                await asyncio.to_thread(
                    send_email, subject, "\n".join(lines), _notify_recipient(), **_smtp_overrides()
                )

            database.set_setting("reminder_last_sent_at", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 定时提醒任务失败: {exc}", file=sys.stderr)


def _daily_report_range_utc(now_hk: datetime) -> Tuple[str, str, str]:
    """给定当前香港时间，返回"前一个香港日历日"（00:00~24:00）对应的 UTC 起止时间字符串
    （格式与 conversations.created_at 一致，便于直接用于 SQL 区间查询），以及该日历日的日期标签。"""
    today_hk_midnight = now_hk.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_hk_midnight = today_hk_midnight - timedelta(days=1)
    fmt_sql = "%Y-%m-%d %H:%M:%S"
    start_utc = yesterday_hk_midnight.astimezone(timezone.utc).strftime(fmt_sql)
    end_utc = today_hk_midnight.astimezone(timezone.utc).strftime(fmt_sql)
    return start_utc, end_utc, yesterday_hk_midnight.strftime("%Y-%m-%d")


async def _daily_report_loop() -> None:
    """每日数据日报：管理员可在工作台开启/关闭，并设置每天推送的时间点（香港时间，HH:MM）。
    到点后统计"前一个香港日历日"的咨询用户数、总对话条数、转人工请求次数，通过邮件推送；
    用一个"今天是否已发送"的日期标记去重，避免同一天到点后被重复触发或重启后重复发送。"""
    while True:
        try:
            await asyncio.sleep(DAILY_REPORT_TICK_SECONDS)
            enabled = database.get_setting("daily_report_enabled", "true") == "true"
            if not enabled:
                continue

            report_time = database.get_setting("daily_report_time", DEFAULT_DAILY_REPORT_TIME)
            now_hk = datetime.now(HK_TZ)
            today_hk_str = now_hk.strftime("%Y-%m-%d")

            if database.get_setting("daily_report_last_sent_date") == today_hk_str:
                continue
            if now_hk.strftime("%H:%M") < report_time:
                continue

            start_utc, end_utc, report_date_label = _daily_report_range_utc(now_hk)
            stats = database.get_daily_stats(start_utc, end_utc)
            subject = f"【{BRAND_NAME} AI 客服数据日报】{report_date_label}"
            body = (
                f"报表日期：{report_date_label}（香港时间 00:00-24:00）\n\n"
                f"咨询用户数：{stats['user_count']} 人\n"
                f"总对话条数：{stats['conversation_count']} 条\n"
                f"转人工请求：{stats['handoff_count']} 次\n"
            )
            await asyncio.to_thread(send_email, subject, body, _notify_recipient(), **_smtp_overrides())
            database.set_setting("daily_report_last_sent_date", today_hk_str)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 每日数据日报任务失败: {exc}", file=sys.stderr)


def _log_retrieval_only(conversation_id: int, product: str, question: str) -> None:
    """全人工模式下不调用 AI 生成回复，但仍在后台记录题库检索结果，便于客服复核。"""
    product_bot = bots.get(product)
    if product_bot is None:
        database.set_retrieval_info(conversation_id, False, None, None, 0.0)
        return
    try:
        ranked = product_bot.retrieve(question)
        if ranked and ranked[0][0] >= product_bot.min_score:
            score, item = ranked[0]
            database.set_retrieval_info(conversation_id, True, item.question, item.answer, score)
        else:
            database.set_retrieval_info(conversation_id, False, None, None, ranked[0][0] if ranked else 0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 检索标注失败: {exc}", file=sys.stderr)


# ---------------- 模式设置 ----------------

@app.get("/api/mode")
def get_mode() -> dict:
    mode = database.get_setting("global_mode", "auto")
    return {"mode": mode}


@app.post("/api/mode")
def set_mode(req: ModeRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="模式必须是 auto / manual / collab 之一")
    database.set_setting("global_mode", req.mode)
    return {"mode": req.mode}


@app.get("/api/collab-timeout")
def get_collab_timeout() -> dict:
    seconds = float(database.get_setting("collab_auto_send_seconds", str(DEFAULT_COLLAB_AUTO_SEND_SECONDS)))
    return {"seconds": seconds}


@app.post("/api/collab-timeout")
def set_collab_timeout(req: CollabTimeoutRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.seconds < 1 or req.seconds > 300:
        raise HTTPException(status_code=400, detail="超时时间需在 1-300 秒之间")
    database.set_setting("collab_auto_send_seconds", str(req.seconds))
    return {"seconds": req.seconds}


# ---------------- SMTP 发信配置（后台可配置，优先于 .env 里的 SMTP_*） ----------------

@app.get("/api/smtp-settings")
def get_smtp_settings(agent: dict = Depends(require_admin)) -> dict:
    """出于安全考虑，密码不会原样返回，只返回是否已配置过（password_set），
    管理员保存新配置时若留空密码，则视为沿用已保存的旧密码。"""
    db_host = database.get_setting("smtp_host")
    db_password = database.get_setting("smtp_password")
    return {
        "host": db_host or os.getenv("SMTP_HOST", ""),
        "port": int(database.get_setting("smtp_port", os.getenv("SMTP_PORT", "587")) or "587"),
        "username": database.get_setting("smtp_username") or os.getenv("SMTP_USERNAME", ""),
        "sender": database.get_setting("smtp_from") or os.getenv("SMTP_FROM", ""),
        "use_tls": (database.get_setting("smtp_use_tls") or os.getenv("SMTP_USE_TLS", "true")) == "true",
        "use_ssl": (database.get_setting("smtp_use_ssl") or os.getenv("SMTP_USE_SSL", "false")) == "true",
        "password_set": bool(db_password) or bool(os.getenv("SMTP_PASSWORD")),
        "source": "database" if db_host else ("env" if os.getenv("SMTP_HOST") else "none"),
    }


@app.post("/api/smtp-settings")
def set_smtp_settings(req: SmtpSettingsRequest, agent: dict = Depends(require_admin)) -> dict:
    if not req.host.strip():
        raise HTTPException(status_code=400, detail="请填写发信服务器地址")
    if req.port < 1 or req.port > 65535:
        raise HTTPException(status_code=400, detail="端口号不合法")

    database.set_setting("smtp_host", req.host.strip())
    database.set_setting("smtp_port", str(req.port))
    database.set_setting("smtp_username", (req.username or "").strip())
    if req.password:  # 留空表示不修改已保存的密码
        database.set_setting("smtp_password", req.password)
    database.set_setting("smtp_from", (req.sender or req.username or "").strip())
    database.set_setting("smtp_use_tls", "true" if req.use_tls else "false")
    database.set_setting("smtp_use_ssl", "true" if req.use_ssl else "false")
    return {"success": True}


@app.post("/api/smtp-settings/test")
async def test_smtp_settings(req: SmtpTestRequest, agent: dict = Depends(require_admin)) -> dict:
    to = (req.to or _notify_recipient() or os.getenv("NOTIFY_EMAIL_TO", DEFAULT_NOTIFY_EMAIL_TO)).strip()
    success, detail = await asyncio.to_thread(send_test_email, to, **_smtp_overrides())
    if not success:
        raise HTTPException(status_code=400, detail=f"发送失败：{detail}")
    return {"success": True, "detail": detail}


# ---------------- 邮件提醒收件邮箱（即时提醒 + 定时提醒共用） ----------------

@app.get("/api/notify-email")
def get_notify_email() -> dict:
    email = database.get_setting("notify_email_to") or os.getenv("NOTIFY_EMAIL_TO", DEFAULT_NOTIFY_EMAIL_TO)
    return {"email": email}


@app.post("/api/notify-email")
def set_notify_email(req: NotifyEmailRequest, agent: dict = Depends(require_admin)) -> dict:
    addresses = [addr.strip() for addr in req.email.split(",") if addr.strip()]
    if not addresses or any("@" not in addr for addr in addresses):
        raise HTTPException(status_code=400, detail="请输入有效的邮箱地址，多个邮箱用英文逗号分隔")
    normalized = ", ".join(addresses)
    database.set_setting("notify_email_to", normalized)
    return {"email": normalized}


# ---------------- 待处理队列定时提醒（邮件） ----------------

@app.get("/api/reminder-settings")
def get_reminder_settings() -> dict:
    enabled = database.get_setting("reminder_enabled", "false") == "true"
    interval_minutes = float(
        database.get_setting("reminder_interval_minutes", str(DEFAULT_REMINDER_INTERVAL_MINUTES))
    )
    return {"enabled": enabled, "interval_minutes": interval_minutes}


@app.post("/api/reminder-settings")
def set_reminder_settings(req: ReminderSettingsRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.interval_minutes < 1 or req.interval_minutes > 1440:
        raise HTTPException(status_code=400, detail="提醒间隔需在 1-1440 分钟之间")
    database.set_setting("reminder_enabled", "true" if req.enabled else "false")
    database.set_setting("reminder_interval_minutes", str(req.interval_minutes))
    # 重新开启或修改间隔时，把"上次发送时间"重置为现在，避免用刚关闭前的旧计时立刻触发一次意外提醒。
    database.set_setting("reminder_last_sent_at", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    return {"enabled": req.enabled, "interval_minutes": req.interval_minutes}


# ---------------- 每日数据日报（邮件） ----------------

@app.get("/api/daily-report-settings")
def get_daily_report_settings() -> dict:
    enabled = database.get_setting("daily_report_enabled", "true") == "true"
    report_time = database.get_setting("daily_report_time", DEFAULT_DAILY_REPORT_TIME)
    return {"enabled": enabled, "time": report_time}


@app.post("/api/daily-report-settings")
def set_daily_report_settings(
    req: DailyReportSettingsRequest, agent: dict = Depends(require_admin)
) -> dict:
    if not re.fullmatch(r"[0-2]\d:[0-5]\d", req.time):
        raise HTTPException(status_code=400, detail="推送时间格式需为 HH:MM，例如 09:00")
    hour, minute = (int(part) for part in req.time.split(":"))
    if hour > 23:
        raise HTTPException(status_code=400, detail="推送时间格式需为 HH:MM，例如 09:00")
    database.set_setting("daily_report_enabled", "true" if req.enabled else "false")
    database.set_setting("daily_report_time", f"{hour:02d}:{minute:02d}")
    # 修改设置后重置"今天是否已发送"标记，避免用旧设置遗留的标记误判为今天已发过
    database.set_setting("daily_report_last_sent_date", "")
    return {"enabled": req.enabled, "time": f"{hour:02d}:{minute:02d}"}


# ---------------- 客户端提问 ----------------

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    product = _normalize_product(req.product)
    mode = database.get_setting("global_mode", "auto")
    is_greeting = _is_pure_greeting(question)
    # 客户直接说"转人工""人工客服"之类，是明确的转接要求，不是需要检索/AI 判断的正常问题——
    # 要在"无关闲聊"判断之前拦截，否则可能被误判成"与产品无关"而回一句引导语（真实发生过的 bug）。
    is_explicit_transfer = not is_greeting and _is_explicit_transfer_request(question)

    conversation_id = database.create_conversation(session_id, question, mode, _client_ip(request), product)

    # 与产品咨询完全无关的闲聊/荒谬提问（"吃饭了吗""这产品对国家安全有危害吗"之类）：不进人工
    # 队列、不发邮件提醒，只在客户端展示一句引导语；但仍然完整落库（标记成 matched=True、附上
    # "无关闲聊/非常规提问"这个说明，跟问候语的记录方式一致），方便管理员事后能在对话记录/刷新
    # 恢复的聊天记录里都看到这条消息，同时也不会被误计入"转人工请求次数"这类统计。管理员可在
    # 工作台随时关闭这个判断（一键退回"全部按正常问题处理"）。打招呼、明确要求转人工的都已在
    # 上面单独处理，这里跳过，避免重复消耗一次 AI 调用、也避免被误判。
    if not is_greeting and not is_explicit_transfer and database.get_setting("skip_irrelevant_enabled", "true") == "true":
        if _is_irrelevant_question(question, product, req.model):
            irrelevant_reply = _irrelevant_reply(product)
            database.set_retrieval_info(conversation_id, True, "无关闲聊/非常规提问", irrelevant_reply, 1.0)
            database.mark_answered(conversation_id, irrelevant_reply)
            return AskResponse(conversation_id=conversation_id, status="answered", answer=irrelevant_reply, mode=mode)

    # 单纯打招呼（"你好""在吗"等，不带具体问题）：任何模式下都直接由 AI 回一句问候语并结束，
    # 不算"题库未命中"，不转人工、不发邮件提醒——避免客户每次只是打个招呼就惊动客服。
    if is_greeting:
        greeting_text = _greeting_reply(product)
        database.set_retrieval_info(conversation_id, True, "问候语", greeting_text, 1.0)
        database.mark_answered(conversation_id, greeting_text)
        return AskResponse(conversation_id=conversation_id, status="answered", answer=greeting_text, mode=mode)

    # 客户明确要求转人工：跳过题库检索/AI 生成，直接进入"转人工"状态，客户端立即展示留邮箱入口。
    # 转人工的必要条件是客户填写（真实格式的）邮箱——这里不会立即发邮件提醒客服，只有客户在留
    # 邮箱入口提交了合法邮箱后，才会真正发邮件通知（见 /leave-email 接口），避免每次客户单纯说一句
    # "转人工"就触发一封没有联系方式、客服也没法回复的邮件；客服在工作台仍能实时看到这条待处理对话。
    if is_explicit_transfer:
        database.set_retrieval_info(conversation_id, False, None, None, 0.0)
        # 客户此时还没补充说明具体想咨询的问题（question 字段还是"转人工"这句占位文本），
        # 标记一下：客户端刷新页面恢复历史记录时要据此重新展示"请描述您的问题"输入框，
        # 而不是误当成已经转人工成功、只需要安静等待的状态。
        database.set_awaiting_transfer_details(conversation_id, True)
        database.set_is_explicit_transfer(conversation_id, True)
        conversation = database.get_conversation(conversation_id)
        await manager.broadcast({"type": "new_question", "conversation": dict(conversation)})
        return AskResponse(
            conversation_id=conversation_id,
            status="pending",
            answer=None,
            mode=mode,
            matched=False,
            need_transfer_details=True,
        )

    # 未命中题库（包括该产品尚未建立题库的情况）时，任何模式都不允许 AI 直接回复或编造答案，
    # 统一转人工处理；只有确认命中题库时，才允许由 AI 直接回复（全AI模式）或生成建议（协同模式）。
    # 转人工不会立刻发邮件提醒客服：只有客户在"人工客服正忙"提示下主动留下邮箱后才会发邮件，
    # 避免客户还没留联系方式时就打扰客服；客服仍能在工作台的待处理队列里实时看到这条对话。
    # pending_matched 仅人机协同模式下可能为 True：题库已命中、AI 已生成建议并安排好自动发送倒计时，
    # 客户此时应看到"AI 思考中"而不是"转人工"提示；其余情况（未命中/全人工）都应显示转人工提示。
    pending_matched = False

    if mode == "auto":
        result: Optional[AnswerResult] = None
        try:
            result = _generate_ai_reply(product, question, req.model)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 全AI模式生成回复失败: {exc}", file=sys.stderr)

        if result is not None and result.matched:
            database.set_retrieval_info(
                conversation_id, True, result.matched_question, result.matched_answer, result.score
            )
            database.mark_answered(conversation_id, result.text)
            return AskResponse(conversation_id=conversation_id, status="answered", answer=result.text, mode=mode)

        database.set_retrieval_info(
            conversation_id, False, None, None, result.score if result is not None else 0.0
        )

    elif mode == "collab":
        try:
            result = _generate_ai_reply(product, question, req.model)
            database.set_retrieval_info(
                conversation_id, result.matched, result.matched_question, result.matched_answer, result.score
            )
            if result.matched:
                database.set_ai_suggestion(conversation_id, result.text, result.score)
                timeout = float(
                    database.get_setting("collab_auto_send_seconds", str(DEFAULT_COLLAB_AUTO_SEND_SECONDS))
                )
                auto_send_at = (datetime.utcnow() + timedelta(seconds=timeout)).strftime("%Y-%m-%dT%H:%M:%SZ")
                database.set_auto_send_at(conversation_id, auto_send_at)
                asyncio.create_task(_auto_send_after_timeout(conversation_id, result.text, timeout))
                pending_matched = True
            # 未命中：不生成 AI 建议、不安排自动发送，完全交给客服人工处理
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 协同模式生成AI建议失败: {exc}", file=sys.stderr)
            database.set_retrieval_info(conversation_id, False, None, None, 0.0)
    elif mode == "manual":
        _log_retrieval_only(conversation_id, product, question)

    # 走到这里说明本次提问需要人工处理（全人工模式 / 协同或全AI模式下未命中题库）；
    # 只广播给工作台实时展示，不发邮件——邮件仅在客户主动留下邮箱后才发送。
    conversation = database.get_conversation(conversation_id)
    await manager.broadcast({"type": "new_question", "conversation": dict(conversation)})

    return AskResponse(
        conversation_id=conversation_id, status="pending", answer=None, mode=mode, matched=pending_matched
    )


@app.get("/api/conversations/by-session/{session_id}")
def list_customer_session_history(session_id: str) -> dict:
    """客户端刷新页面时用来恢复聊天记录：session_id 相当于客户浏览器里的私有令牌（存
    在 sessionStorage，随标签页关闭而清除，刷新页面时还在），知道它才能查到对应记录，
    安全性与 /api/ask 现有的会话机制一致。这里只返回客户自己该看到的字段（问题、最终
    回复、状态、模式、题库是否命中、客户自己留的邮箱），不暴露 AI 建议草稿、题库匹配
    详情、客户 IP 等客服工作台专用信息。客户本来就知道自己留的邮箱是什么，直接把邮箱
    原文返回给前端，方便刷新页面后继续显示"客服稍后通过邮件 xxx 回复您"这类带具体邮箱
    的提示文案，不算泄露额外信息。"""
    items = database.list_session_messages(session_id)
    return {
        "items": [
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["final_answer"],
                "status": item["status"],
                "mode_used": item["mode_used"],
                "matched": bool(item["matched"]),
                "has_email": bool(item["customer_email"]),
                "email": item["customer_email"] or None,
                "awaiting_transfer_details": bool(item["awaiting_transfer_details"]),
                # 客户端刷新恢复历史记录时用来选择正确的确认话术："已将您的问题…更新给
                # 人工客服"（主动转人工场景）而不是"已收到您的问题，正在为您转接人工客服"
                # （题库未命中被动转人工场景）——这个标记不会随补充问题而清零，永久记住
                # 这条对话最初是怎么进入转人工状态的。
                "is_explicit_transfer": bool(item["is_explicit_transfer"]),
                # 提问时间（UTC，不带时区后缀，前端按 UTC 解析）：客户端刷新恢复历史记录时，
                # 用它算出这条对话已经等了多久，避免每次刷新都从 0 重新数 10 秒——不然客户
                # 明明已经等过、已经看到过"人工客服正忙"提示，刷新一次就"倒退"回"AI 思考中"，
                # 还要重新等满 10 秒才能看到本该早就出现的提示。
                "created_at": item["created_at"],
            }
            for item in items
        ]
    }


@app.post("/api/chat-messages")
def create_chat_message(req: ChatMessageCreateRequest) -> dict:
    """客户端聊天记录逐条落库：客户端每显示一条消息气泡（客户的提问、AI/客服的回复、
    各种过程性提示）就调一次这个接口。conversations 表每条对话只存一行最终状态，光靠它
    没法在刷新后原样还原出转人工这类多步流程的全部过程消息（引导语、"人工客服正忙"提示、
    第一次确认语等都会丢失或措辞对不上）；有了这份明细，刷新页面就是简单的逐条重放，
    跟刷新前看到的完全一致。session_id 是客户浏览器里的私有令牌，安全模型与
    /api/conversations/by-session 一致。"""
    role = req.role if req.role in ("user", "bot") else "bot"
    kind = req.kind if req.kind in ("text", "waiting", "transfer_form", "email_form", "busy_note") else "text"
    message_id = database.add_chat_message(
        session_id=req.session_id,
        role=role,
        content=(req.content or "")[:4000],
        kind=kind,
        conversation_id=req.conversation_id,
        after_id=req.after_id,
    )
    return {"id": message_id}


@app.post("/api/chat-messages/{message_id}/update")
def update_chat_message(message_id: int, req: ChatMessageUpdateRequest) -> dict:
    """更新一条已落库的消息气泡（内容/类型/所属对话）。等待类气泡"先占位、后原地更新"：
    "AI 正在思考…"最终会被答案原地替换，DOM 里同一个气泡对应明细表里同一行。"""
    kind = req.kind if req.kind in (None, "text", "waiting", "transfer_form", "email_form", "busy_note") else None
    database.update_chat_message(
        message_id=message_id,
        session_id=req.session_id,
        content=(req.content or "")[:4000] if req.content is not None else None,
        kind=kind,
        conversation_id=req.conversation_id,
    )
    return {"success": True}


@app.get("/api/chat-messages/by-session/{session_id}")
def list_chat_messages_by_session(session_id: str) -> dict:
    """客户端刷新页面时拉取本会话的全部消息明细，按显示顺序逐条重放。"""
    return {"items": database.list_chat_messages(session_id)}


@app.get("/api/conversations/{conversation_id}")
def get_conversation_status(conversation_id: int) -> dict:
    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {
        "id": conversation["id"],
        "status": conversation["status"],
        "answer": conversation["final_answer"],
    }


@app.post("/api/conversations/{conversation_id}/leave-email")
async def leave_customer_email(conversation_id: int, req: LeaveEmailRequest) -> dict:
    """客户在"人工客服正忙"提示下主动留下邮箱：仅在客户填写并提交时才会调用此接口，
    客户不填邮箱则完全不会触发这条邮件通知，与其他转人工场景各自独立。"""
    email = req.email.strip()
    if not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="请输入有效的邮箱地址")

    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    saved = database.set_customer_email(conversation_id, email)
    if not saved:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 全人工模式下客服本就要盯着工作台处理每一条对话，不需要额外邮件提醒；
    # 客户留下的邮箱依然会保存、在工作台可见，只是不再触发邮件。
    if conversation["mode_used"] != "manual":
        asyncio.create_task(
            _notify_customer_email_left(
                conversation_id,
                conversation["product"],
                conversation["question"],
                conversation["mode_used"],
                email,
                database.get_visitor_no(conversation["session_id"]),
            )
        )
    return {"success": True}


@app.post("/api/conversations/{conversation_id}/transfer-question")
async def submit_transfer_question(conversation_id: int, req: TransferQuestionRequest) -> dict:
    """客户主动说"转人工"之后，补充说明具体想咨询的问题（必填，替换掉"转人工"这句没有实际
    内容的占位提问，方便客服知道要处理什么）；邮箱选填，留下真实格式的邮箱才会真正发邮件通知
    客服，不留邮箱则只更新问题内容、不触发邮件，客服仍能在工作台实时看到更新后的问题。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="请输入您想咨询的问题")

    email = (req.email or "").strip()
    if email and not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="请输入有效的邮箱地址，或留空")

    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    updated = database.set_question(conversation_id, question)
    if not updated:
        raise HTTPException(status_code=404, detail="对话不存在")

    # 客户明确说"转人工"之后补充的这个具体问题，同样要过一遍"无关闲聊/低俗/违法/涉政等不当
    # 内容"判断——否则客户可以直接靠说一句"转人工"绕开题库检索这道判断，把无关或不当内容
    # 硬推给人工客服处理。命中的话直接按无关闲聊处理并结束：不进入转人工等待状态、不发邮件
    # 通知客服、不广播给工作台，客户端直接看到引导语，整个转人工流程到此终止。
    if database.get_setting("skip_irrelevant_enabled", "true") == "true":
        if _is_irrelevant_question(question, conversation["product"], None):
            irrelevant_reply = _irrelevant_reply(conversation["product"])
            database.set_retrieval_info(conversation_id, True, "无关闲聊/非常规提问", irrelevant_reply, 1.0)
            database.mark_answered(conversation_id, irrelevant_reply)
            return {"success": True, "irrelevant": True, "answer": irrelevant_reply}

    if email:
        database.set_customer_email(conversation_id, email)

    conversation = database.get_conversation(conversation_id)
    await manager.broadcast({"type": "conversation_updated", "conversation": dict(conversation)})

    # 全人工模式不需要邮件提醒（同上，客服已在工作台盯着处理）。
    if email and conversation["mode_used"] != "manual":
        asyncio.create_task(
            _notify_customer_email_left(
                conversation_id,
                conversation["product"],
                question,
                conversation["mode_used"],
                email,
                database.get_visitor_no(conversation["session_id"]),
            )
        )
    return {"success": True, "irrelevant": False}


# ---------------- 客服端 ----------------

@app.post("/api/agent/login")
def agent_login(req: LoginRequest) -> dict:
    agent_row = database.get_agent_by_username(req.username)
    if agent_row is None or not database.verify_password(req.password, agent_row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(agent_row)
    return {
        "token": token,
        "agent": {
            "id": agent_row["id"],
            "username": agent_row["username"],
            "displayName": agent_row["display_name"],
            "role": agent_row["role"],
        },
    }


@app.get("/api/agent/me")
def agent_me(agent: dict = Depends(get_current_agent)) -> dict:
    return {"agent": agent}


@app.post("/api/agent/change-password")
def agent_change_password(
    req: ChangePasswordRequest, agent: dict = Depends(get_current_agent)
) -> dict:
    agent_row = database.get_agent_by_id(agent["id"])
    if not database.verify_password(req.old_password, agent_row["password_hash"]):
        raise HTTPException(status_code=401, detail="旧密码不正确")
    database.update_agent_password(agent["id"], req.new_password)
    return {"success": True}


@app.get("/api/agent/queue")
def agent_queue(agent: dict = Depends(get_current_agent)) -> dict:
    return {"items": database.list_queue()}


@app.get("/api/agent/history")
def agent_history(agent: dict = Depends(get_current_agent)) -> dict:
    return {"items": database.list_recent(50)}


@app.get("/api/agent/sessions")
def agent_sessions(agent: dict = Depends(get_current_agent)) -> dict:
    """按用户（session_id）分组的对话列表：一个用户对应一个对话框，而不是每条问题单独一个。"""
    return {"items": database.list_sessions()}


@app.get("/api/agent/sessions/{session_id}")
def agent_session_messages(session_id: str, agent: dict = Depends(get_current_agent)) -> dict:
    """某个用户会话下的全部提问/回复，按时间顺序返回，用于渲染连续对话。"""
    return {"items": database.list_session_messages(session_id)}


@app.post("/api/agent/answer/{conversation_id}")
async def agent_answer(
    conversation_id: int, req: AnswerRequest, agent: dict = Depends(get_current_agent)
) -> dict:
    answer_text = req.answer.strip()
    if not answer_text:
        raise HTTPException(status_code=400, detail="回复内容不能为空")

    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    if conversation["status"] == "answered":
        raise HTTPException(status_code=409, detail="该问题已被回复，请勿重复发送")

    database.mark_answered(conversation_id, answer_text, agent["id"], agent["display_name"])
    await manager.broadcast({"type": "answered", "id": conversation_id})
    return {"success": True}


@app.get("/api/agent/agents")
def list_agent_accounts(agent: dict = Depends(require_admin)) -> dict:
    return {"agents": database.list_agents()}


@app.post("/api/agent/agents")
def create_agent_account(req: CreateAgentRequest, agent: dict = Depends(require_admin)) -> dict:
    existing = database.get_agent_by_username(req.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    role = req.role if req.role in {"agent", "admin"} else "agent"
    agent_id = database.create_agent(req.username, req.password, req.display_name, role)
    return {"id": agent_id, "username": req.username, "displayName": req.display_name, "role": role}


@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket, token: str = "") -> None:
    try:
        decode_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
