import asyncio
import base64
import hmac
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
import whatsapp
from auth import create_token, decode_token, get_current_agent, require_admin
from ai_client import get_ai_client
from doc_rag_chatbot import DocRagBot
from email_utils import DEFAULT_NOTIFY_EMAIL_TO, send_email, send_test_email
from excel_rag_chatbot import AnswerResult, ExcelFaqRagBot, retrieval_evidence
from ws_manager import manager

DEFAULT_EXCEL = "2026.01.26_肤润康-常见咨询问题_v2(1).xls"
UREA_DOC = "urea_hand_cream_info.md"
DEFAULT_MODEL = "qwen3.6-flash"
# 所有邮件通知的主题统一带上品牌名，方便客服在收件箱里一眼认出是哪个客服系统发的。
BRAND_NAME = "YUNI"
VALID_MODES = {"auto", "manual", "collab"}
MODE_LABELS = {"auto": "全AI模式", "manual": "全人工模式", "collab": "人机协同模式"}
DEFAULT_COLLAB_AUTO_SEND_SECONDS = 5
# AI 自动回复的置信度阈值（百分比）：题库命中后，只有置信度达到阈值，全AI模式才会直接回复
# 客户、协同模式才会安排自动发送倒计时；低于阈值一律转人工，AI 草稿保留给客服参考。
# 管理员可在工作台设置里调整，设为 0 表示不启用这道门槛（回到"命中即自动回复"的旧行为）。
DEFAULT_MIN_CONFIDENCE_PERCENT = 70
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


class MinConfidenceRequest(BaseModel):
    percent: float


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

# 客户上传的图片存到数据目录（Docker volume 持久化），文件名统一用随机 hex，
# 原始文件名不落盘，也就不存在路径穿越或文件名冲突问题。
UPLOAD_DIR = database.DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_UPLOAD_NAME_RE = re.compile(r"[a-f0-9]{32}\.(?:jpg|jpeg|png|gif|webp)")
_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# 客户发来的图片先由视觉模型转成一段文字描述，再跟客户的文字问题合并成一个问题，交给
# 原有的题库检索/回答流程处理——AI 始终只能照题库答，不会因为"看到了图"就自由发挥。
# 默认用 qwen3-vl-flash（阿里百炼上最便宜的视觉模型，做"描述图片"够用）；若该模型在你的
# 账号/地域不可用，把 VISION_MODEL 换成 qwen3.6-flash（本身也支持图像）即可。
DEFAULT_VISION_MODEL = "qwen3-vl-flash"
# 图片描述只用于喂给检索/回答，不直接展示给客户，所以要求客观、简短、不做任何诊断或建议。
VISION_SYSTEM_PROMPT = (
    "你是客服系统里的图片识别助手。客户在咨询护肤产品时发来一张图片，请只客观描述图片里"
    "实际能看到的内容，供后续的客服流程参考。\n"
    "要求：\n"
    "1. 只描述看得见的事实：皮肤问题的部位、颜色、范围、形态；产品包装上的名称、规格、"
    "生产日期、批号等文字；截图里的文字内容等。\n"
    "2. 绝对不要做医学判断、不要猜测病因、不要给任何用药或护理建议、不要安慰或寒暄。\n"
    "3. 如果图片与护肤品、皮肤状况、产品包装都无关，直接说明图片里是什么即可。\n"
    "4. 用一段中文陈述，100 字以内，不要分点，不要加任何前缀。"
)
# 图片描述与提问的合并窗口：客户"先发图、再打字"是两个独立请求，这段时间内的下一次提问
# 会自动带上图片描述。设成 10 分钟而不是 10 秒，是因为客户只发图不说话时，客服机器人会在
# 10 秒后主动追问"想咨询这张图的什么问题"，客户回答这句追问时图片必须还有效。
PENDING_IMAGE_MAX_AGE_SECONDS = 600

# 客户端聊天气泡的类型白名单（chat_messages.kind）。刷新页面重放历史记录时前端按 kind
# 决定把气泡渲染成静态文字、可交互表单、图片还是重新接回轮询（见 static/index.html 的
# restoreHistory）；不在白名单里的一律当普通文字，所以新增气泡类型时必须同步加到这里，
# 否则该类型落库后会被降级成 text，刷新后就还原不出原来的样子。
CHAT_MESSAGE_KINDS = (
    "text",
    "waiting",
    "transfer_form",
    "email_form",
    "busy_note",
    "image",
    "idle_prompt",
    "session_end",
)

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

    # WhatsApp 的重推去重表只在几分钟内有意义，旧记录清掉，避免长期无限增长。
    database.prune_whatsapp_processed_messages()

    asyncio.create_task(_reminder_loop())
    asyncio.create_task(_daily_report_loop())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent")
def agent_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "agent.html")


@app.get("/privacy")
def privacy_page() -> FileResponse:
    """Meta 要求 App 发布前必须填一个真实可访问的隐私政策地址，这个页面就是拿来填那一栏的。"""
    return FileResponse(STATIC_DIR / "privacy.html")


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
        client = get_ai_client(timeout=15.0, max_retries=1)
        resp = client.chat.completions.create(
            model=_ai_model(model),
            temperature=0,
            # 输出只是一个 {"irrelevant": bool} 的小 JSON，限制输出长度防模型偶尔啰嗦拖慢响应
            max_tokens=50,
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


def _vision_model() -> str:
    return os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)


def _describe_image(path: Path) -> str:
    """调视觉模型把客户发来的图片转成一段客观的文字描述。返回空字符串表示识别失败
    （没配 API Key、模型不支持图片、网络异常等）——调用方要能在没有描述的情况下照常工作，
    图片识别只是"锦上添花"，不能因为它挂了就让客户连图都发不出去。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        client = get_ai_client(api_key=api_key, timeout=40.0, max_retries=1)
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
        data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        resp = client.chat.completions.create(
            model=_vision_model(),
            temperature=0.1,
            # 描述要求 100 字以内，300 tokens 绰绰有余，防模型偶尔长篇大论拖慢上传响应
            max_tokens=300,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "请描述这张图片的内容。"},
                    ],
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()[:500]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 图片识别失败（将按无描述处理）: {exc}", file=sys.stderr)
        return ""


def _merge_image_into_question(question: str, descriptions: list) -> str:
    """把图片的文字描述并进客户的文字问题，合成一个交给题库检索/AI 回答的完整问题。
    客户常常是"发一张图 + 一句「这个正常吗」"，单看文字什么信息都没有、必然检索不中；
    带上图片描述之后，"手背发红脱皮"这类内容才能匹配到题库里对应的问题。"""
    parts = [desc for desc in descriptions if desc]
    if not parts:
        return question
    joined = "；".join(parts)
    return f"{question}\n【客户同时发来图片，图片内容】{joined}"


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
        await _push_answer_to_whatsapp(conversation, ai_answer)
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
    """全人工模式下不调用 AI 生成回复，但仍在后台记录题库检索结果，便于客服复核。
    这里没有 AI 语义判断，标注里的置信度只有"检索证据"这一部分（校准到与其他模式相同的
    0-1 尺度），供客服参考。"""
    product_bot = bots.get(product)
    if product_bot is None:
        database.set_retrieval_info(conversation_id, False, None, None, 0.0)
        return
    try:
        if hasattr(product_bot, "retrieve_detailed"):
            detailed = product_bot.retrieve_detailed(question)
            if detailed and detailed[0]["score"] >= product_bot.min_score:
                best = detailed[0]
                evidence = round(
                    retrieval_evidence(best["lexical"], best["semantic"], best["in_both"]), 4
                )
                database.set_retrieval_info(
                    conversation_id, True, best["item"].question, best["item"].answer, evidence,
                    retrieval_evidence=evidence,
                )
            else:
                miss_evidence = (
                    round(
                        retrieval_evidence(
                            detailed[0]["lexical"], detailed[0]["semantic"], detailed[0]["in_both"]
                        ),
                        4,
                    )
                    if detailed
                    else 0.0
                )
                database.set_retrieval_info(
                    conversation_id, False, None, None, miss_evidence, retrieval_evidence=miss_evidence
                )
            return
        # 文档题库（DocRagBot）等没有 retrieve_detailed 的检索器：沿用原始综合分。
        # 文档条目是 DocChunk（title/text），没有 question/answer 字段，做一层兼容。
        ranked = product_bot.retrieve(question)
        if ranked and ranked[0][0] >= product_bot.min_score:
            score, item = ranked[0]
            q_text = getattr(item, "question", None) or getattr(item, "title", "")
            a_text = getattr(item, "answer", None) or getattr(item, "text", "")
            database.set_retrieval_info(conversation_id, True, q_text, a_text, score)
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


def _min_confidence_threshold() -> float:
    """AI 自动回复所需的最低置信度（0-1）。后台设置的是百分比，这里换算并做健壮性兜底。"""
    try:
        percent = float(
            database.get_setting("min_confidence_percent", str(DEFAULT_MIN_CONFIDENCE_PERCENT))
        )
    except (TypeError, ValueError):
        percent = DEFAULT_MIN_CONFIDENCE_PERCENT
    return max(0.0, min(100.0, percent)) / 100.0


@app.get("/api/min-confidence")
def get_min_confidence() -> dict:
    return {"percent": round(_min_confidence_threshold() * 100, 1)}


@app.post("/api/min-confidence")
def set_min_confidence(req: MinConfidenceRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.percent < 0 or req.percent > 100:
        raise HTTPException(status_code=400, detail="置信度阈值需在 0-100 之间")
    database.set_setting("min_confidence_percent", str(req.percent))
    return {"percent": req.percent}


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
    return await _process_question(
        question,
        req.session_id or str(uuid.uuid4()),
        req.product,
        req.model,
        _client_ip(request),
    )


async def _process_question(
    question: str,
    session_id: str,
    product: Optional[str],
    model: Optional[str],
    client_ip: Optional[str],
) -> AskResponse:
    """一次提问的完整处理流程：落库、并入客户刚发的图片、无关闲聊/打招呼/转人工判断、
    按当前工作模式检索题库并生成回复。网页端（/api/ask）和 WhatsApp webhook 共用这一段，
    两个渠道的回答行为因此完全一致——改一处两边同时生效，不会出现"网页答得对、WhatsApp
    答得不一样"的分叉。"""
    product = _normalize_product(product)
    mode = database.get_setting("global_mode", "auto")
    is_greeting = _is_pure_greeting(question)
    # 客户直接说"转人工""人工客服"之类，是明确的转接要求，不是需要检索/AI 判断的正常问题——
    # 要在"无关闲聊"判断之前拦截，否则可能被误判成"与产品无关"而回一句引导语（真实发生过的 bug）。
    is_explicit_transfer = not is_greeting and _is_explicit_transfer_request(question)

    conversation_id = database.create_conversation(session_id, question, mode, client_ip, product)

    # 客户刚发来、还没被任何提问用掉的图片：把视觉模型给出的文字描述并进本次提问，作为
    # 一个完整问题走后面正常的检索/回答流程。conversations.question 仍然只存客户自己打的
    # 文字（工作台里客户气泡要显示客户的原话），图片描述单独存一列给客服复核。
    pending_images = database.take_pending_images(session_id, PENDING_IMAGE_MAX_AGE_SECONDS)
    image_descriptions = [item["description"] for item in pending_images if item["description"]]
    has_image = bool(pending_images)
    if pending_images:
        database.set_conversation_image(
            conversation_id, pending_images[-1]["url"], "；".join(image_descriptions)
        )
    # 后续检索/AI 判断统一用这个合并后的问题；打招呼、转人工这类"看客户原话"的判断仍用原文。
    query = _merge_image_into_question(question, image_descriptions)

    # 单纯打招呼（"你好""在吗"等，不带具体问题）：任何模式下都直接由 AI 回一句问候语并结束，
    # 不算"题库未命中"，不转人工、不发邮件提醒——避免客户每次只是打个招呼就惊动客服。
    # 附带图片时不走这条捷径：客户发了图再说一句"你好"，真正要问的是图片，不能只回一句问候语
    # 就把图片丢掉。
    if is_greeting and not has_image:
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

    # ---- 无关闲聊判断 与 题库检索/AI 匹配 并行执行 ----
    # 两者是相互独立的模型调用，串行等完要 2-4 秒。这里同时发起：主流程先等"无关闲聊"判断
    # （要靠它决定这条消息走不走正常问答），判定无关时直接返回、检索那边的结果作废（多花
    # 一次调用的钱，但无关问题占比很低）；判定相关时检索/匹配多半也已经跑完，总耗时约等于
    # 两者中较慢的那个，而不是两者之和。asyncio.to_thread 同时把这些阻塞的同步调用挪出
    # 事件循环，避免一个客户的提问把其他请求全卡住。
    #
    # 无关闲聊判断的行为与原来完全一致（详见 _classify_irrelevant 的说明）：命中时不进人工
    # 队列、不发邮件，只回一句引导语，但仍完整落库（matched=True + "无关闲聊/非常规提问"），
    # 管理员可用 skip_irrelevant_enabled 一键关闭。附带图片的提问跳过该判断（客户特意拍照
    # 说明有真实需求，"这个正常吗"这类短文本很容易被误判）；打招呼、明确转人工已在上面返回。
    irrelevant_task: Optional[asyncio.Task] = None
    if not has_image and database.get_setting("skip_irrelevant_enabled", "true") == "true":
        irrelevant_task = asyncio.create_task(
            asyncio.to_thread(_is_irrelevant_question, question, product, model)
        )

    reply_task: Optional[asyncio.Task] = None
    if mode in ("auto", "collab"):
        reply_task = asyncio.create_task(asyncio.to_thread(_generate_ai_reply, product, query, model))

    if irrelevant_task is not None:
        irrelevant = False
        try:
            irrelevant = await irrelevant_task
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 无关提问判断异常，按正常问题处理: {exc}", file=sys.stderr)
        if irrelevant:
            if reply_task is not None:
                # 线程里的调用无法真正中断，只是不再等它的结果
                reply_task.cancel()
            irrelevant_reply = _irrelevant_reply(product)
            database.set_retrieval_info(conversation_id, True, "无关闲聊/非常规提问", irrelevant_reply, 1.0)
            database.mark_answered(conversation_id, irrelevant_reply)
            return AskResponse(conversation_id=conversation_id, status="answered", answer=irrelevant_reply, mode=mode)

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
            result = await reply_task
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 全AI模式生成回复失败: {exc}", file=sys.stderr)

        if result is not None and result.matched:
            database.set_retrieval_info(
                conversation_id, True, result.matched_question, result.matched_answer, result.confidence,
                ai_confidence=result.ai_confidence, retrieval_evidence=result.evidence,
            )
            if result.confidence >= _min_confidence_threshold():
                database.mark_answered(conversation_id, result.text)
                return AskResponse(
                    conversation_id=conversation_id, status="answered", answer=result.text, mode=mode
                )
            # 命中题库但置信度低于阈值：不允许 AI 直接回复，转人工处理；AI 生成的回复
            # 存成建议草稿，客服在工作台一键采纳或改写后发送，不会自动发送。
            database.set_ai_suggestion(conversation_id, result.text, result.confidence)
        else:
            # 未命中也记录（最高候选的）置信度，工作台跟"未找到匹配内容"一起展示，供调阈值参考
            database.set_retrieval_info(
                conversation_id, False, None, None,
                result.confidence if result is not None else 0.0,
                retrieval_evidence=result.evidence if result is not None else None,
            )

    elif mode == "collab":
        try:
            result = await reply_task
            database.set_retrieval_info(
                conversation_id,
                result.matched,
                result.matched_question,
                result.matched_answer,
                result.confidence,
                ai_confidence=result.ai_confidence,
                retrieval_evidence=result.evidence,
            )
            if result.matched:
                database.set_ai_suggestion(conversation_id, result.text, result.confidence)
                if result.confidence >= _min_confidence_threshold():
                    timeout = float(
                        database.get_setting("collab_auto_send_seconds", str(DEFAULT_COLLAB_AUTO_SEND_SECONDS))
                    )
                    auto_send_at = (datetime.utcnow() + timedelta(seconds=timeout)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    database.set_auto_send_at(conversation_id, auto_send_at)
                    asyncio.create_task(_auto_send_after_timeout(conversation_id, result.text, timeout))
                    pending_matched = True
                # 命中但置信度低于阈值：AI 建议保留给客服参考，但不安排自动发送，
                # 必须由客服确认/改写后手动发送（客户端按转人工的等待流程处理）。
            # 未命中：不生成 AI 建议、不安排自动发送，完全交给客服人工处理
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 协同模式生成AI建议失败: {exc}", file=sys.stderr)
            database.set_retrieval_info(conversation_id, False, None, None, 0.0)
    elif mode == "manual":
        # 检索标注涉及 embedding 网络调用，放线程里跑，别阻塞事件循环
        await asyncio.to_thread(_log_retrieval_only, conversation_id, product, query)

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
                # 客户端只关心"AI 会不会自己回"（决定刷新后显示"AI 思考中"还是"转人工"等待）。
                # 数据库里的 matched 是题库是否命中，与之并不等价：命中但置信度低于阈值的对话
                # 不会自动发送、需要人工处理（status=pending 且没有 auto_send_at），刷新恢复时
                # 必须按"转人工"展示，否则客户会一直卡在"AI 正在思考"、留邮箱提示也不会出现。
                "matched": bool(item["matched"])
                and (item["status"] != "pending" or bool(item["auto_send_at"])),
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
    kind = req.kind if req.kind in CHAT_MESSAGE_KINDS else "text"
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
    kind = req.kind if req.kind in CHAT_MESSAGE_KINDS else None
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


@app.post("/api/upload-image")
async def upload_image(session_id: str = Form(""), file: UploadFile = File(...)) -> dict:
    """客户在聊天里发送图片：保存到数据目录，再调视觉模型把图片内容转成一段文字描述暂存起来，
    等客户接着提问时并进问题一起走正常的题库检索/回答流程（见 /api/ask）。
    客户端拿到 URL 后自己把图片气泡按 kind=image 落库（走既有的 chat-messages 流程，
    这样刷新重放、排序都跟普通气泡一套逻辑）。"""
    ext = Path(file.filename or "").suffix.lower()
    if ext == ".jpe":
        ext = ".jpg"
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="仅支持 jpg / png / gif / webp 格式的图片")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="图片不能超过 5MB，请压缩后重试")
    name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / name
    path.write_bytes(data)
    url = f"/uploads/{name}"

    # 识别失败（没配 Key、模型不支持、超时等）只会让这张图少一段描述，图片本身照常发送成功，
    # 客服仍能在工作台直接看图，所以这里不把异常抛给客户端。
    description = await asyncio.to_thread(_describe_image, path)
    if session_id:
        database.add_pending_image(session_id, url, description)
    return {"url": url, "recognized": bool(description)}


@app.get("/uploads/{filename}")
def get_uploaded_image(filename: str) -> FileResponse:
    """按文件名返回客户上传的图片。文件名必须完全匹配"32位hex+扩展名"的生成规则，
    天然排除路径穿越；不匹配或文件不存在一律 404。"""
    if not _UPLOAD_NAME_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="文件不存在")
    path = UPLOAD_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


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


# ---------------- WhatsApp 接入 ----------------
# 客户在 WhatsApp 里得到的答案，跟网页端走的是同一套问答逻辑（_process_question），
# 工作台也不用改：WhatsApp 客户的 session_id 固定是 "wa:手机号"，在工作台里就是一个
# 普通访客。区别只在表达方式——WhatsApp 没有网页那种表单、按钮气泡和轮询，所有过程性
# 提示都得是一条能独立看懂的消息，产品选择用 Meta 的交互式按钮实现。

WA_WELCOME_TEXT = f"您好，我是 {BRAND_NAME} AI 客服～请先选择您要咨询的产品："
WA_PRODUCT_PROMPT = "请选择您要咨询的产品："
# 「换产品」这个入口只能靠客户打关键词触发，WhatsApp 里没有网页那种常驻链接，
# 所以必须在选完产品的这条提示里说出来，否则客户根本不知道有这个功能。
WA_PRODUCT_CHOSEN_TEXT = "好的，请问想咨询什么？\n（想咨询其他产品，随时回复「换产品」）"
WA_TRANSFER_PENDING_TEXT = "已收到您的问题，正在为您转接人工客服，请稍候～"
WA_TRANSFER_DETAILS_TEXT = "好的，请把您想咨询的问题发给我，我马上转给人工客服～"
WA_IMAGE_ONLY_TEXT = "已收到您的图片～请问想咨询这张图片的什么问题呢？"
WA_IMAGE_FAILED_TEXT = "抱歉，这张图片没能接收成功，麻烦您重新发一次，或直接用文字描述一下～"
WA_UNSUPPORTED_TEXT = "抱歉，目前只能处理文字和图片消息，麻烦您用文字描述一下想咨询的问题～"
# 客户想换一款产品咨询时的口语说法（网页端是点"切换产品"链接，WhatsApp 只能靠关键词）
WA_SWITCH_PRODUCT_KEYWORDS = ("切换产品", "换产品", "换个产品", "重新选择", "选择产品", "选产品")
WA_PRODUCT_BUTTON_PREFIX = "product:"
# WhatsApp 客户十有八九先发一句"你好"再问正事。这类开场白不能当成待回答的问题存起来，
# 否则客户选完产品后会立刻收到一句"您的问题和本产品无关"，显得莫名其妙。
WA_GREETING_WORDS = (
    "你好", "您好", "哈罗", "哈啰", "喂", "在吗", "在么", "有人吗", "请问",
    "hi", "hello", "hey", "helo", "早上好", "下午好", "晚上好", "打扰了",
)
WA_MIME_EXTS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


@app.get("/api/whatsapp/webhook")
def whatsapp_verify(request: Request) -> PlainTextResponse:
    """Meta 在后台保存 webhook 地址时会先发一个 GET 握手请求：verify token 对得上，就把
    hub.challenge 原样以纯文本返回。返回 JSON 或多余的内容都会导致验证失败，这是接入
    WhatsApp 时最常见的坑。"""
    params = request.query_params
    expected = whatsapp.verify_token()
    provided = params.get("hub.verify_token") or ""
    if params.get("hub.mode") == "subscribe" and expected and hmac.compare_digest(provided, expected):
        return PlainTextResponse(params.get("hub.challenge") or "")
    raise HTTPException(status_code=403, detail="verify token 不匹配")


@app.post("/api/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background: BackgroundTasks) -> dict:
    """接收 Meta 推来的消息事件。

    Meta 要求 20 秒内返回 200，否则会判定失败并不断重推同一条消息；而识别图片、调用 AI
    生成回答远不止 20 秒，所以这里只做签名校验和解析，真正的处理丢进后台任务，立刻应答。"""
    raw = await request.body()
    # 接入排查用：只要 Meta 真的推到了服务器，这里一定有输出。日志里连这条都没有，
    # 说明消息根本没送达（多半是 webhook 没订阅 messages 字段，或号码没接进这个 App）。
    print(f"[info] 收到 WhatsApp webhook 推送 {len(raw)} 字节", file=sys.stderr)
    if not whatsapp.check_signature(raw, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=403, detail="签名校验失败")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
    background.add_task(_handle_whatsapp_payload, payload)
    return {"success": True}


@app.get("/api/whatsapp/status")
def whatsapp_status(agent: dict = Depends(require_admin)) -> dict:
    """给管理员自查用：只报告各项配置有没有填，不返回任何密钥内容。"""
    return {
        "configured": whatsapp.is_configured(),
        "phone_number_id_set": bool(whatsapp.phone_number_id()),
        "access_token_set": bool(whatsapp.access_token()),
        "verify_token_set": bool(whatsapp.verify_token()),
        "app_secret_set": bool(whatsapp.app_secret()),
        "api_version": whatsapp.api_version(),
    }


async def _handle_whatsapp_payload(payload: dict) -> None:
    """拆开 Meta 那层套了四五层的 webhook 结构，逐条消息处理。
    送达/已读回执（value.statuses）这里不需要，天然会被跳过（它们没有 messages 字段）。"""
    try:
        for entry in payload.get("entry") or []:
            for change in entry.get("changes") or []:
                value = change.get("value") or {}
                contacts = {c.get("wa_id"): c for c in (value.get("contacts") or [])}
                for message in value.get("messages") or []:
                    await _handle_whatsapp_message(message, contacts.get(message.get("from")) or {})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] WhatsApp webhook 处理失败: {exc}", file=sys.stderr)


async def _handle_whatsapp_message(message: dict, contact_info: dict) -> None:
    wa_id = (message.get("from") or "").strip()
    message_id = (message.get("id") or "").strip()
    if not wa_id or not message_id:
        return
    # Meta 没收到 200 就会重推，重推的 message id 与首次相同；这里挡掉重复处理，
    # 否则同一句话会被回答两遍、同一张图会被视觉模型识别两次。
    if not database.mark_whatsapp_message_processed(message_id):
        return

    profile_name = ((contact_info.get("profile") or {}).get("name") or "").strip() or None
    contact = database.upsert_whatsapp_contact(wa_id, profile_name)
    await whatsapp.mark_read(message_id)

    msg_type = message.get("type")
    if msg_type == "text":
        await _handle_whatsapp_text(wa_id, contact, ((message.get("text") or {}).get("body") or "").strip())
    elif msg_type == "image":
        await _handle_whatsapp_image(wa_id, contact, message.get("image") or {})
    elif msg_type == "interactive":
        await _handle_whatsapp_interactive(wa_id, message.get("interactive") or {})
    else:
        # 语音、视频、文件、位置、名片等：现有题库和视觉模型都处理不了，礼貌引导回文字。
        await whatsapp.send_text(wa_id, WA_UNSUPPORTED_TEXT)


async def _send_whatsapp_product_picker(wa_id: str, lead_text: str) -> None:
    buttons = [
        (f"{WA_PRODUCT_BUTTON_PREFIX}{product_id}", meta["label"])
        for product_id, meta in PRODUCTS.items()
    ]
    await whatsapp.send_buttons(wa_id, lead_text, buttons)


async def _handle_whatsapp_interactive(wa_id: str, interactive: dict) -> None:
    reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
    choice = reply.get("id") or ""
    if not choice.startswith(WA_PRODUCT_BUTTON_PREFIX):
        await whatsapp.send_text(wa_id, WA_UNSUPPORTED_TEXT)
        return

    product = _normalize_product(choice[len(WA_PRODUCT_BUTTON_PREFIX) :])
    database.set_whatsapp_product(wa_id, product)
    contact = database.get_whatsapp_contact(wa_id) or {}
    pending_question = (contact.get("pending_question") or "").strip()
    if pending_question:
        # 客户是先问了问题、被要求先选产品的：选完立刻按那个问题作答，不用他再打一遍。
        database.set_whatsapp_pending_question(wa_id, None)
        contact["product"] = product
        await _whatsapp_answer_question(wa_id, contact, pending_question)
        return
    await whatsapp.send_text(wa_id, WA_PRODUCT_CHOSEN_TEXT)


def _is_greeting_only(text: str) -> bool:
    """判断这句话除了打招呼之外还有没有实际内容。

    把所有开场白词和标点去掉后，剩下的字符太少就认为只是打招呼："你好" → 空，
    "你好在吗" → 空，都算；而"你好，孕妇能用吗"去掉"你好"后还剩一整个问题，不算。
    """
    stripped = text.lower()
    for word in WA_GREETING_WORDS:
        stripped = stripped.replace(word, "")
    stripped = re.sub(r"[\s\W_]+", "", stripped, flags=re.UNICODE)
    return len(stripped) <= 3


async def _handle_whatsapp_text(wa_id: str, contact: dict, text: str) -> None:
    if not text:
        return
    if any(keyword in text for keyword in WA_SWITCH_PRODUCT_KEYWORDS):
        await _send_whatsapp_product_picker(wa_id, WA_PRODUCT_PROMPT)
        return
    if not contact.get("product"):
        # 只有确实带着问题的开场消息才留着，选完产品后自动作答；纯打招呼不留，
        # 选完产品后回一句"请问想咨询什么？"就好。
        if not _is_greeting_only(text):
            database.set_whatsapp_pending_question(wa_id, text)
        await _send_whatsapp_product_picker(wa_id, WA_WELCOME_TEXT)
        return
    awaiting_id = contact.get("awaiting_transfer_conversation_id")
    if awaiting_id:
        await _whatsapp_submit_transfer_question(wa_id, int(awaiting_id), text)
        return
    await _whatsapp_answer_question(wa_id, contact, text)


async def _whatsapp_answer_question(wa_id: str, contact: dict, text: str) -> None:
    """把客户这句话交给跟网页端完全相同的问答流程，再按结果决定回什么。"""
    result = await _process_question(
        text,
        contact.get("session_id") or database.whatsapp_session_id(wa_id),
        contact.get("product"),
        None,
        f"WhatsApp {wa_id}",
    )
    if result.status == "answered" and result.answer:
        await whatsapp.send_text(wa_id, result.answer)
        return
    if result.need_transfer_details:
        # 客户说了"转人工"这类没有实际内容的话：记住这条对话，等他把真正想问的问题发过来
        # 补进同一条对话里，避免工作台队列里留下一条只写着"转人工"的空对话。
        database.set_whatsapp_awaiting_transfer(wa_id, result.conversation_id)
        await whatsapp.send_text(wa_id, WA_TRANSFER_DETAILS_TEXT)
        return
    if result.matched:
        # 人机协同模式且题库已命中：AI 建议已进入几秒后的自动发送倒计时，答案会由
        # _auto_send_after_timeout 推送过来，这里不用先回一句没信息量的"请稍候"。
        return
    await whatsapp.send_text(wa_id, WA_TRANSFER_PENDING_TEXT)


async def _whatsapp_submit_transfer_question(wa_id: str, conversation_id: int, text: str) -> None:
    """客户说完"转人工"之后补充的具体问题：更新到原来那条对话里，逻辑与网页端的
    /api/conversations/{id}/transfer-question 保持一致（同样要过一遍无关内容判断，
    否则客户可以靠一句"转人工"绕开这道拦截，把无关内容直接推给人工客服）。"""
    database.set_whatsapp_awaiting_transfer(wa_id, None)
    conversation = database.get_conversation(conversation_id)
    if conversation is None or conversation["status"] == "answered":
        # 客服已经处理完了，这句话按新问题走正常流程。
        await _whatsapp_answer_question(wa_id, database.get_whatsapp_contact(wa_id) or {}, text)
        return

    database.set_question(conversation_id, text)
    if database.get_setting("skip_irrelevant_enabled", "true") == "true":
        if _is_irrelevant_question(text, conversation["product"], None):
            irrelevant_reply = _irrelevant_reply(conversation["product"])
            database.set_retrieval_info(conversation_id, True, "无关闲聊/非常规提问", irrelevant_reply, 1.0)
            database.mark_answered(conversation_id, irrelevant_reply)
            await whatsapp.send_text(wa_id, irrelevant_reply)
            return

    conversation = database.get_conversation(conversation_id)
    await manager.broadcast({"type": "conversation_updated", "conversation": dict(conversation)})
    await whatsapp.send_text(wa_id, f"已将您的问题「{text}」更新给人工客服，请稍候～")


async def _handle_whatsapp_image(wa_id: str, contact: dict, image: dict) -> None:
    """客户发来的图片：下载下来存进和网页端同一个目录，走同一个视觉模型识别成文字描述，
    再存进 pending_images——之后客户的提问会自动把这段描述并进去（见 _process_question）。
    图片自带文字说明（caption）时，等于"图 + 问题"一起发来，直接按那句话作答。"""
    media_id = (image.get("id") or "").strip()
    caption = (image.get("caption") or "").strip()
    if not media_id:
        await whatsapp.send_text(wa_id, WA_IMAGE_FAILED_TEXT)
        return

    data, mime = await whatsapp.download_media(media_id)
    if not data:
        await whatsapp.send_text(wa_id, WA_IMAGE_FAILED_TEXT)
        return
    if len(data) > MAX_IMAGE_BYTES:
        await whatsapp.send_text(wa_id, "图片太大了（超过 5MB），麻烦压缩后再发一次～")
        return

    ext = WA_MIME_EXTS.get(mime, ".jpg")
    name = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / name
    path.write_bytes(data)
    session_id = contact.get("session_id") or database.whatsapp_session_id(wa_id)
    # 识别失败只是少一段描述，图片本身照常留在工作台供客服直接查看，所以不打断流程。
    description = await asyncio.to_thread(_describe_image, path)
    database.add_pending_image(session_id, f"/uploads/{name}", description)

    if not contact.get("product"):
        # 还没选产品：图片已经存下（10 分钟内有效），选完产品再提问就会自动带上它。
        await _send_whatsapp_product_picker(wa_id, WA_WELCOME_TEXT)
        return
    if caption:
        await _whatsapp_answer_question(wa_id, contact, caption)
        return
    await whatsapp.send_text(wa_id, WA_IMAGE_ONLY_TEXT)


async def _push_answer_to_whatsapp(conversation, answer: str) -> None:
    """客服在工作台回复、或协同模式自动发送之后，把最终答案主动推回客户的 WhatsApp。
    网页端客户是自己轮询把答案拉回去的，WhatsApp 没有这个机制，必须由服务端推。

    注意 Meta 的 24 小时客服窗口：客户最后一条消息之后超过 24 小时，这种普通文本会被
    拒收（只能改用审批过的模板消息），失败时 whatsapp.py 会把 Meta 的原始报错打进日志。"""
    if conversation is None:
        return
    session_id = conversation["session_id"] or ""
    if not session_id.startswith("wa:"):
        return
    await whatsapp.send_text(session_id[len("wa:") :], answer)


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
    """某个用户会话下的全部提问/回复，按时间顺序返回，用于渲染连续对话。
    客户发送的图片没有对应的 conversations 行（AI 不处理图片），单独从聊天明细里取出，
    按时间插入到问答流中，让客服在同一个对话里看到客户发的图。"""
    items = [dict(item, item_type="qa") for item in database.list_session_messages(session_id)]
    images = [
        {
            "item_type": "image",
            "id": f"img-{row['id']}",
            "image_url": row["content"],
            "image_description": row.get("description") or "",
            "created_at": row["created_at"],
        }
        for row in database.list_session_images(session_id)
    ]
    merged = sorted(items + images, key=lambda x: x.get("created_at") or "")
    return {"items": merged}


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
    # 网页端客户会自己轮询拿到这条回复；WhatsApp 客户必须由服务端主动推过去。
    await _push_answer_to_whatsapp(conversation, answer_text)
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
