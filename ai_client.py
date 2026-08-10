"""进程内复用 OpenAI 客户端。

每次调用都 new 一个 OpenAI() 会重建 httpx 连接池：TCP + TLS 握手要几百毫秒，而一次客户
提问要经过"无关闲聊判断 → embedding 检索 → 语义匹配"最多三次模型调用，握手开销会白白
叠加三遍。OpenAI 客户端底层的 httpx 连接池是线程安全的，这里按 (api_key, base_url,
timeout, max_retries) 缓存复用：同参数的调用共享同一个客户端，跨请求保持长连接。
"""

import os
import threading
from typing import Optional

_lock = threading.Lock()
_cache: dict = {}


def get_ai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    max_retries: int = 2,
):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("未安装 openai 依赖，请执行: pip install openai") from exc

    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
    cache_key = (resolved_key, resolved_base, timeout, max_retries)
    with _lock:
        client = _cache.get(cache_key)
        if client is None:
            client = OpenAI(
                api_key=resolved_key,
                base_url=resolved_base,
                timeout=timeout,
                max_retries=max_retries,
            )
            _cache[cache_key] = client
        return client
