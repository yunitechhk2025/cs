"""邮件提醒工具：客户问题需要转人工处理时，通过邮件提醒客服/管理员。

SMTP 配置的优先级：调用方显式传入的参数（通常来自后台「工作台设置」保存在数据库里的配置）
> 环境变量（.env 里的 SMTP_*）。两者都没有配置时，跳过发送并打印警告，不抛出异常，
避免邮件故障影响客户提问、客服回复等主业务流程。
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import List, Optional, Tuple

# 测试阶段默认转发到该邮箱；正式上线后可通过环境变量 NOTIFY_EMAIL_TO 或后台设置覆盖。
DEFAULT_NOTIFY_EMAIL_TO = "yunitechhk@gmail.com"


def _resolve_config(
    host: Optional[str],
    port: Optional[int],
    username: Optional[str],
    password: Optional[str],
    sender: Optional[str],
    use_tls: Optional[bool],
    use_ssl: Optional[bool],
) -> Tuple[Optional[str], int, Optional[str], Optional[str], str, bool, bool]:
    resolved_host = host or os.getenv("SMTP_HOST")
    resolved_port = int(port or os.getenv("SMTP_PORT", "587"))
    resolved_username = username if username is not None else os.getenv("SMTP_USERNAME")
    resolved_password = password if password is not None else os.getenv("SMTP_PASSWORD")
    resolved_sender = sender or os.getenv("SMTP_FROM") or resolved_username or "no-reply@fuyunhon.local"
    resolved_use_tls = (
        use_tls if use_tls is not None else os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false"
    )
    resolved_use_ssl = (
        use_ssl if use_ssl is not None else os.getenv("SMTP_USE_SSL", "false").strip().lower() == "true"
    )
    return (
        resolved_host,
        resolved_port,
        resolved_username,
        resolved_password,
        resolved_sender,
        resolved_use_tls,
        resolved_use_ssl,
    )


def _do_send(
    subject: str,
    body: str,
    recipients: List[str],
    host: str,
    port: int,
    username: Optional[str],
    password: Optional[str],
    sender: str,
    use_tls: bool,
    use_ssl: bool,
) -> None:
    """真正执行发送，失败时直接抛异常，由调用方决定如何处理（静默降级 or 返回详细错误）。"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)

    smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    with smtp_cls(host, port, timeout=15) as server:
        if use_tls and not use_ssl:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.sendmail(sender, recipients, msg.as_string())


def send_email(
    subject: str,
    body: str,
    to: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    sender: Optional[str] = None,
    use_tls: Optional[bool] = None,
    use_ssl: Optional[bool] = None,
) -> bool:
    """发送一封纯文本提醒邮件。成功返回 True，任何失败（未配置/连接失败/认证失败等）返回 False，
    并只打印警告日志——用于业务流程里的"顺手提醒"场景，邮件失败绝不能影响主流程。"""
    (
        resolved_host,
        resolved_port,
        resolved_username,
        resolved_password,
        resolved_sender,
        resolved_use_tls,
        resolved_use_ssl,
    ) = _resolve_config(host, port, username, password, sender, use_tls, use_ssl)

    if not resolved_host:
        print("[warn] 未配置 SMTP_HOST（环境变量或后台设置均未配置），跳过邮件发送", file=sys.stderr)
        return False

    recipients = [addr.strip() for addr in (to or os.getenv("NOTIFY_EMAIL_TO", DEFAULT_NOTIFY_EMAIL_TO)).split(",") if addr.strip()]
    if not recipients:
        print("[warn] 未配置收件邮箱，跳过邮件发送", file=sys.stderr)
        return False

    try:
        _do_send(
            subject,
            body,
            recipients,
            resolved_host,
            resolved_port,
            resolved_username,
            resolved_password,
            resolved_sender,
            resolved_use_tls,
            resolved_use_ssl,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 邮件发送失败: {exc}", file=sys.stderr)
        return False


def send_test_email(
    to: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    sender: Optional[str] = None,
    use_tls: Optional[bool] = None,
    use_ssl: Optional[bool] = None,
) -> Tuple[bool, str]:
    """供后台「测试发送」按钮使用：返回 (是否成功, 给管理员看的详细信息/报错原因)，不吞异常细节。"""
    (
        resolved_host,
        resolved_port,
        resolved_username,
        resolved_password,
        resolved_sender,
        resolved_use_tls,
        resolved_use_ssl,
    ) = _resolve_config(host, port, username, password, sender, use_tls, use_ssl)

    if not resolved_host:
        return False, "未配置 SMTP 服务器地址"

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    if not recipients:
        return False, "收件邮箱为空"

    try:
        _do_send(
            "【YUNI 客服系统测试邮件】SMTP 配置测试",
            "这是一封测试邮件，用来验证 YUNI 客服系统的邮件提醒功能是否配置成功。\n\n"
            "如果你收到了这封邮件，说明 SMTP 配置正确，功能可以正常使用。",
            recipients,
            resolved_host,
            resolved_port,
            resolved_username,
            resolved_password,
            resolved_sender,
            resolved_use_tls,
            resolved_use_ssl,
        )
        return True, f"已发送到 {', '.join(recipients)}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
