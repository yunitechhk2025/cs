"""WhatsApp Cloud API 的底层封装：发消息、下载客户发来的媒体文件、校验 webhook 签名。

这里只做"跟 Meta 打交道"这一层，不含任何问答业务逻辑（业务流程都在 web_app.py 的 webhook
处理里），这样单独测试这层很容易，以后要换成别的通道（Twilio、360dialog 等 BSP）也只需要
替换这一个文件。

所有配置都从环境变量读，且是每次调用时现读：这样在服务器上改完 .env 重启即可生效，
不需要改代码；没配置时 is_configured() 返回 False，webhook 会直接拒绝，不影响网页端客服。
"""

import hashlib
import hmac
import os
import sys
from typing import List, Optional, Tuple

import httpx

GRAPH_HOST = "https://graph.facebook.com"
# Graph API 版本号：Meta 大约每季度发一个新版，旧版本约两年后停用。需要升级时改环境变量即可。
DEFAULT_API_VERSION = "v21.0"
TIMEOUT_SECONDS = 20
# WhatsApp 单条文本消息的上限是 4096 个字符，超出会被接口直接拒绝，所以长回复要拆开发。
MAX_TEXT_LENGTH = 4000
# 交互式按钮：最多 3 个，标题最长 20 个字符（超出同样会被接口拒绝）。
MAX_BUTTONS = 3
MAX_BUTTON_TITLE = 20


def access_token() -> str:
    return os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()


def phone_number_id() -> str:
    return os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()


def verify_token() -> str:
    return os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()


def app_secret() -> str:
    return os.getenv("WHATSAPP_APP_SECRET", "").strip()


def api_version() -> str:
    return os.getenv("WHATSAPP_API_VERSION", DEFAULT_API_VERSION).strip() or DEFAULT_API_VERSION


def is_configured() -> bool:
    """只有访问令牌和号码 ID 都配了才算接通——这两个是发消息的最低要求。"""
    return bool(access_token() and phone_number_id())


def check_signature(raw_body: bytes, header: Optional[str]) -> bool:
    """校验 Meta 在 X-Hub-Signature-256 里带的 HMAC-SHA256 签名，确认这个请求确实来自
    Meta 而不是别人伪造的（webhook 地址是公网可访问的，不校验等于谁都能往里灌消息）。

    没配 WHATSAPP_APP_SECRET 时跳过校验并打一条警告：方便刚接通时先跑流程，但正式对外
    使用前一定要把 app secret 配上。"""
    secret = app_secret()
    if not secret:
        print("[warn] 未配置 WHATSAPP_APP_SECRET，跳过 webhook 签名校验", file=sys.stderr)
        return True
    if not header or not header.startswith("sha256="):
        print("[warn] webhook 请求没带 X-Hub-Signature-256 签名头，已拒绝", file=sys.stderr)
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, header[len("sha256=") :].strip()):
        # 接入阶段最难查的一种失败：Meta 已经把消息推过来了，但因为 app secret 配错被
        # 静默拒绝，表现和"完全没收到消息"一模一样，所以这里必须留一条明确的日志。
        print(
            "[warn] webhook 签名校验失败，请检查 WHATSAPP_APP_SECRET 是否为该 App 的应用密钥",
            file=sys.stderr,
        )
        return False
    return True


def split_text(text: str) -> List[str]:
    """把过长的回复按上限切成多条：优先在换行处断开，实在没有换行才硬切。"""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    while len(text) > MAX_TEXT_LENGTH:
        cut = text.rfind("\n", 0, MAX_TEXT_LENGTH)
        if cut <= 0:
            cut = MAX_TEXT_LENGTH
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


async def _post_message(payload: dict) -> bool:
    if not is_configured():
        print("[warn] WhatsApp 未配置，消息未发送", file=sys.stderr)
        return False
    url = f"{GRAPH_HOST}/{api_version()}/{phone_number_id()}/messages"
    headers = {"Authorization": f"Bearer {access_token()}"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            res = await client.post(url, json=payload, headers=headers)
        if res.status_code >= 400:
            # 把 Meta 返回的错误原文打出来：绝大多数发送失败（令牌过期、超出 24 小时客服
            # 窗口、号码没注册 WhatsApp）都能在这段 JSON 里直接看出原因。
            print(f"[warn] WhatsApp 发送失败 {res.status_code}: {res.text}", file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] WhatsApp 发送异常: {exc}", file=sys.stderr)
        return False


async def send_text(to: str, text: str) -> bool:
    """给客户发一条纯文本。过长的内容会自动拆成多条按顺序发出。"""
    ok = True
    for chunk in split_text(text):
        sent = await _post_message(
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": False, "body": chunk},
            }
        )
        ok = ok and sent
    return ok


async def send_buttons(to: str, body: str, buttons: List[Tuple[str, str]]) -> bool:
    """发一条带回复按钮的交互式消息（用来做产品选择）。buttons 是 (id, 标题) 的列表，
    客户点击后 Meta 会在 webhook 里以 interactive.button_reply 的形式把 id 回传过来。"""
    items = []
    for btn_id, title in buttons[:MAX_BUTTONS]:
        items.append(
            {"type": "reply", "reply": {"id": btn_id, "title": title[:MAX_BUTTON_TITLE]}}
        )
    return await _post_message(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": body[:1024]},
                "action": {"buttons": items},
            },
        }
    )


async def mark_read(message_id: str) -> bool:
    """把客户的消息标记成已读（客户端会看到蓝色双勾）。失败不影响任何流程。"""
    return await _post_message(
        {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    )


async def download_media(media_id: str) -> Tuple[Optional[bytes], str]:
    """下载客户发来的媒体文件。Cloud API 分两步：先用 media id 换一个临时下载地址
    （只有 5 分钟有效期），再带着同一个访问令牌去下载实际内容。

    返回 (文件内容, mime 类型)；失败时返回 (None, "")。"""
    if not is_configured():
        return None, ""
    headers = {"Authorization": f"Bearer {access_token()}"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            meta_res = await client.get(
                f"{GRAPH_HOST}/{api_version()}/{media_id}", headers=headers
            )
            if meta_res.status_code >= 400:
                print(
                    f"[warn] WhatsApp 获取媒体地址失败 {meta_res.status_code}: {meta_res.text}",
                    file=sys.stderr,
                )
                return None, ""
            info = meta_res.json()
            url = info.get("url")
            mime = (info.get("mime_type") or "").split(";")[0].strip()
            if not url:
                return None, ""
            file_res = await client.get(url, headers=headers)
            if file_res.status_code >= 400:
                print(f"[warn] WhatsApp 下载媒体失败 {file_res.status_code}", file=sys.stderr)
                return None, ""
            return file_res.content, mime
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] WhatsApp 下载媒体异常: {exc}", file=sys.stderr)
        return None, ""
