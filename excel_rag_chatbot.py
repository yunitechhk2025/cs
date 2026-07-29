import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kb_desensitize import desensitize


@dataclass
class FaqItem:
    sheet: str
    row_index: int
    serial: str
    question: str
    answer: str
    # 跨产品共用的问题（例如品牌类问题"是澳洲品牌？"）：无论客户当前咨询的是哪款产品，
    # 都应该能命中这一条，而不局限于某一款产品自己的题库。
    shared: bool = False

    def kb_text(self) -> str:
        prefix = f"序号: {self.serial} | " if self.serial else ""
        return f"{prefix}问题: {self.question} | 回答: {self.answer}"


@dataclass
class AnswerResult:
    text: str
    matched: bool
    score: float
    matched_question: Optional[str] = None
    matched_answer: Optional[str] = None


class ExcelFaqRagBot:
    def __init__(self, excel_path: str, top_k: int = 5, min_score: float = 0.05) -> None:
        self.excel_path = excel_path
        self.top_k = top_k
        self.min_score = min_score
        self.items: List[FaqItem] = []

        # 中文 FAQ 场景，字符 n-gram 检索更稳（无需分词）
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.doc_vectors = None

        # 语义向量检索：弥补字符检索对口语化/同义改写问句（如“涨豆豆” vs “长痘”）召回不足的问题
        self.embed_model = os.getenv("FAQ_EMBED_MODEL", "text-embedding-v3")
        self.item_embeddings: Optional[np.ndarray] = None

    def _find_header_row(
        self, df: pd.DataFrame, question_key: str = "咨询问题", answer_key: str = "解答"
    ) -> Optional[int]:
        for ridx, row in df.iterrows():
            values = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v) != "nan"]
            joined = " | ".join(values)
            if question_key in joined and answer_key in joined:
                return int(ridx)
        return None

    def _sheet_to_faq_items(self, sheet_name: str, df_raw: pd.DataFrame) -> List[FaqItem]:
        if df_raw.empty:
            return []

        header_row_idx = self._find_header_row(df_raw)
        if header_row_idx is None:
            return []

        header_values = df_raw.iloc[header_row_idx].tolist()
        columns = [
            str(v).strip() if str(v).strip() and str(v) != "nan" else f"col_{i}"
            for i, v in enumerate(header_values)
        ]

        df = df_raw.iloc[header_row_idx + 1 :].copy()
        df.columns = columns
        df = df.fillna("")

        question_col = next((c for c in df.columns if "咨询问题" in c or c == "问题"), None)
        answer_col = next((c for c in df.columns if "解答" in c or "答案" in c), None)
        serial_col = next((c for c in df.columns if c in {"序号", "编号"}), None)

        if question_col is None or answer_col is None:
            return []

        # 题库表格里偶尔会有"（同31的答案）"这类内部编辑备注（甚至偶尔打错成"囘"），
        # 这只是维护者留给自己看的交叉引用标记，客服/客户不需要看到，直接清理掉。
        note_pattern = re.compile(r"[（(]\s*[同囘]\S{0,10}答案\s*[）)]")

        items: List[FaqItem] = []
        for idx, row in df.iterrows():
            question = note_pattern.sub("", str(row.get(question_col, "")).strip()).strip()
            answer = str(row.get(answer_col, "")).strip()
            serial = str(row.get(serial_col, "")).strip() if serial_col else ""

            if not question or question.lower() == "nan":
                continue
            if not answer or answer.lower() == "nan":
                continue

            items.append(
                FaqItem(
                    sheet=sheet_name,
                    row_index=int(idx),
                    serial=serial,
                    # 脱敏版：题库原文里的真实品牌名/创始人姓名在这里就替换掉，答案是原文照搬
                    # 发给客户的，不在加载时替换，客户就会看到真实品牌（见 kb_desensitize.py）。
                    # 共用问题的判定用替换前的原文，避免脱敏后跟 _LEGACY_SHARED_QUESTIONS 对不上。
                    question=desensitize(question),
                    answer=desensitize(answer),
                    shared=self._is_shared_question(serial, question),
                )
            )
        return items

    # 品牌类问题（如"是澳洲品牌？"）不属于某一款具体产品，两款产品都应该能命中。
    # 约定：以后在 Excel 的"序号/编号"列填写"公共"/"通用"/"共用"，即可标记为跨产品共用问题，
    # 不需要改代码；下面这个集合只是兼容当前已有题库里、尚未按新约定标记的历史共用问题。
    _SHARED_SERIAL_MARKERS = {"公共", "通用", "共用"}
    _LEGACY_SHARED_QUESTIONS = {"是澳洲品牌？"}

    @classmethod
    def _is_shared_question(cls, serial: str, question: str) -> bool:
        return serial.strip() in cls._SHARED_SERIAL_MARKERS or question.strip() in cls._LEGACY_SHARED_QUESTIONS

    def build_index(self) -> None:
        sheets = pd.read_excel(self.excel_path, sheet_name=None, header=None)
        items: List[FaqItem] = []
        for sheet_name, df in sheets.items():
            items.extend(self._sheet_to_faq_items(sheet_name, df))

        if not items:
            raise ValueError("未识别到 FAQ 问答列（需要包含“咨询问题/解答”字段）。")

        self.items = items
        self._finalize_index()

    @classmethod
    def from_items(cls, items: List[FaqItem], top_k: int = 8, min_score: float = 0.1) -> "ExcelFaqRagBot":
        """不依赖 Excel 文件，直接用一批现成的 FaqItem 构建索引。
        用于还没有自己专属题库、但需要共用跨产品问题（如品牌类问题）的产品。"""
        bot = cls(excel_path="", top_k=top_k, min_score=min_score)
        bot.items = list(items)
        bot._finalize_index()
        return bot

    def load_extra_items(self, extra_items: List[FaqItem]) -> None:
        """合并额外的问答条目（例如其他产品题库里标记为共用的问题），按内容去重后重建索引。"""
        existing_keys = {(it.question, it.answer) for it in self.items}
        added = False
        for it in extra_items:
            key = (it.question, it.answer)
            if key in existing_keys:
                continue
            self.items.append(it)
            existing_keys.add(key)
            added = True
        if added:
            self._finalize_index()

    def _finalize_index(self) -> None:
        # 问题权重更高，回答作为补充，提升短问命中率
        corpus = [f"{it.question} {it.question} {it.answer}" for it in self.items]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self._build_semantic_index()

    def _get_ai_client(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai 依赖，请执行: pip install openai") from exc

        return OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=30.0,
            max_retries=2,
        )

    def _build_semantic_index(self, batch_size: int = 10, attempts: int = 3) -> None:
        """为所有题库问题计算一次语义向量，用于弥补字符检索对同义/口语化改写问句的召回不足。
        DashScope 的 embedding 接口单次请求最多支持 10 条文本，因此分批调用。
        这是启动时的一次性操作，遇到瞬时网络抖动时值得多试几次；
        全部失败时静默降级为纯关键词检索，不影响其余功能。"""
        self.item_embeddings = None
        texts = [it.question for it in self.items]

        for attempt in range(1, attempts + 1):
            try:
                client = self._get_ai_client()
                all_vectors: List[List[float]] = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = client.embeddings.create(model=self.embed_model, input=batch)
                    all_vectors.extend(d.embedding for d in resp.data)
                vectors = np.array(all_vectors, dtype="float32")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.item_embeddings = vectors / norms
                return
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] 语义向量索引构建失败（第 {attempt}/{attempts} 次尝试）: {exc}",
                    file=sys.stderr,
                )

        print("[warn] 语义向量索引最终构建失败，将仅使用关键词检索。", file=sys.stderr)
        self.item_embeddings = None

    def _semantic_scores(
        self, user_query: str, base_url: Optional[str] = None, api_key: Optional[str] = None, attempts: int = 2
    ) -> Optional[np.ndarray]:
        """实时调 embedding 接口算语义相似度：偶尔会因网络抖动/超时/限流失败，失败后会静默降级
        成纯关键词检索——对于跟题库问题字面重合很少、全靠语义才能命中的口语化问题（比如"小孩能用嘛"
        vs 题库里的"适用年龄？"），关键词分数往往很低，一旦这次语义调用失败就会从"命中"变成"未命中"，
        导致同一个问题连续问两次却出现不一样的识别结果。这里加一次快速重试，尽量把这种偶发失败盖住，
        而不是第一次失败就直接放弃退化。"""
        if self.item_embeddings is None:
            return None
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                client = self._get_ai_client(base_url=base_url, api_key=api_key)
                resp = client.embeddings.create(model=self.embed_model, input=[user_query])
                q_vec = np.array(resp.data[0].embedding, dtype="float32")
                norm = np.linalg.norm(q_vec)
                if norm == 0:
                    return None
                q_vec = q_vec / norm
                return self.item_embeddings @ q_vec
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < attempts:
                    time.sleep(0.3)
        print(f"[warn] 语义检索调用失败（已重试{attempts}次），本次仅使用关键词检索: {last_exc}", file=sys.stderr)
        return None

    def retrieve(
        self, user_query: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> List[Tuple[float, FaqItem]]:
        if self.doc_vectors is None:
            raise RuntimeError("索引尚未构建，请先执行 build_index()。")

        q_vec = self.vectorizer.transform([user_query])
        lexical_scores = cosine_similarity(q_vec, self.doc_vectors)[0]

        # 额外叠加关键词重合分，避免纯 TF-IDF 对短中文问句过严
        query_chars = set(user_query.replace(" ", ""))
        for i, item in enumerate(self.items):
            q_chars = set(item.question.replace(" ", ""))
            if not query_chars or not q_chars:
                continue
            overlap = len(query_chars & q_chars) / max(len(query_chars), 1)
            lexical_scores[i] = float(lexical_scores[i]) + 0.25 * overlap

        semantic_scores = self._semantic_scores(user_query, base_url=base_url, api_key=api_key)

        # 分别取关键词检索和语义检索各自的 top-k，取并集作为候选池：
        # 关键词检索命中字面/关键词重合明显的问题；语义检索能召回“涨豆豆”vs“长痘”这类
        # 字符完全不重合但意思相同的口语化改写问题。二者互补，交给后续 AI 语义判断做最终裁决。
        n = len(self.items)
        k = min(self.top_k, n)
        lexical_top_idx = sorted(range(n), key=lambda i: lexical_scores[i], reverse=True)[:k]
        candidate_idx = set(lexical_top_idx)
        if semantic_scores is not None:
            semantic_top_idx = sorted(range(n), key=lambda i: semantic_scores[i], reverse=True)[:k]
            candidate_idx.update(semantic_top_idx)

        def combined_score(i: int) -> float:
            score = float(lexical_scores[i])
            if semantic_scores is not None:
                score = max(score, float(semantic_scores[i]))
            return score

        ranked = sorted(
            [(combined_score(i), self.items[i]) for i in candidate_idx],
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked

    def _not_found_text(self) -> str:
        return (
            "亲，这个问题题库里暂时没有完全对应的说明。"
            "您可以再补充一下具体场景，比如：孕妇/备孕、敏感肌、使用顺序、搭配禁忌、见效周期，我再帮您查。"
        )

    def _call_ai_match(
        self,
        user_query: str,
        ranked: Sequence[Tuple[float, FaqItem]],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[int, str]:
        """让 AI 在关键词检索出的候选题库问题中做语义匹配（允许措辞、语序、同义表达不同，
        只要客户意图与某条候选实质相同即可），并生成引出官方说明的开场白。
        绝不让 AI 生成或复述专业内容本身，专业内容始终由调用方原文拼接，确保逐字一致。

        返回 (候选序号, 开场白)；序号为 0 表示 AI 判断所有候选都不匹配。
        """
        candidates = "\n".join(f"{i}. {item.question}" for i, (_, item) in enumerate(ranked, start=1))
        system_prompt = (
            "你是电商客服助手的题库匹配模块，只能依据候选题库问题做判断，禁止使用外部知识回答专业内容。\n"
            "任务：\n"
            "1. 判断客户问题在意图上最匹配下面哪一条候选题库问题——即使措辞、语序、同义表达不同，"
            "只要客户实际想问的事情和某条候选实质相同，就算匹配。\n"
            "2. 如果确实匹配到某一条，生成一句简短亲切、口语化的开场白/过渡语（不超过20个字），"
            "用于引出接下来将原文展示的官方说明；开场白中严禁提及、复述或猜测任何成分、功效、用法用量、"
            "禁忌等专业内容，那部分会在开场白之后原文附加，不需要你生成。\n"
            "3. 如果没有任何一条候选真正匹配客户的意图，index 填 0，greeting 填空字符串。\n"
            "4. 只输出如下格式的 JSON，不要输出任何多余文字、解释或代码块标记：\n"
            '{"index": 数字, "greeting": "字符串"}'
        )
        user_prompt = f"客户问题：{user_query}\n\n候选题库问题：\n{candidates}"

        client = self._get_ai_client(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0, ""
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return 0, ""

        try:
            idx = int(payload.get("index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        greeting = str(payload.get("greeting", "") or "").strip()
        return idx, greeting

    def answer(
        self,
        user_query: str,
        model: str = "qwen3.6-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AnswerResult:
        ranked = self.retrieve(user_query, base_url=base_url, api_key=api_key)
        if not ranked:
            return AnswerResult(text="抱歉，题库为空或尚未建立索引。", matched=False, score=0.0)

        selection: Optional[Tuple[int, str]] = None
        try:
            # 始终把候选交给 AI 做语义判断，不因关键词分数偏低而提前拒判，
            # 这样措辞不同但意思相同的相似问题也能命中题库答案。
            selection = self._call_ai_match(
                user_query=user_query,
                ranked=ranked,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] AI 语义匹配失败，已回退到关键词检索结果: {exc}", file=sys.stderr)

        if selection is not None:
            idx, greeting = selection
            if idx and 1 <= idx <= len(ranked):
                score, item = ranked[idx - 1]
                if not greeting:
                    greeting = "亲，为您查询到以下官方说明："
                # 专业内容始终是题库原文的直接拼接，不经过 AI 改写，确保逐字一致
                text = f"{greeting}\n{item.answer}"
                return AnswerResult(
                    text=text,
                    matched=True,
                    score=score,
                    matched_question=item.question,
                    matched_answer=item.answer,
                )
            return AnswerResult(text=self._not_found_text(), matched=False, score=ranked[0][0])

        # AI 调用异常时的兜底：仅依赖关键词检索分数
        best_score, best_item = ranked[0]
        if best_score < self.min_score:
            return AnswerResult(text=self._not_found_text(), matched=False, score=best_score)
        text = f"亲，为您查询到以下官方说明：\n{best_item.answer}"
        return AnswerResult(
            text=text,
            matched=True,
            score=best_score,
            matched_question=best_item.question,
            matched_answer=best_item.answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Excel FAQ RAG 客服机器人（仅 AI 模式）")
    parser.add_argument("--excel", required=True, help="Excel 文件路径（支持 .xls/.xlsx）")
    parser.add_argument("--top-k", type=int, default=5, help="每次检索候选条数，默认 5")
    parser.add_argument("--min-score", type=float, default=0.05, help="最低命中分数阈值，默认 0.05")
    parser.add_argument("--model", default="qwen3.6-flash", help="AI 模型名（DashScope 如 qwen3.6-flash）")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址（可选）")
    parser.add_argument("--api-key", default=None, help="API Key（可选，默认读取 OPENAI_API_KEY）")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="执行一次索引和样例问答后退出，便于快速验证",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    bot = ExcelFaqRagBot(excel_path=args.excel, top_k=args.top_k, min_score=args.min_score)

    try:
        bot.build_index()
    except FileNotFoundError:
        print(f"找不到 Excel 文件: {args.excel}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"加载 Excel 失败: {exc}")
        sys.exit(1)

    print(f"索引完成，共 {len(bot.items)} 条 FAQ。")

    if args.self_test:
        sample_q = "孕妇能用吗"
        print(f"\n[自测问题] {sample_q}")
        result = bot.answer(sample_q, model=args.model, base_url=args.base_url, api_key=args.api_key)
        print(result.text)
        print(f"\n[检索标注] matched={result.matched} score={result.score:.3f} matched_question={result.matched_question!r}")
        return

    print("客服机器人已启动。输入 exit 退出。")
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("机器人: 已退出。")
            break

        result = bot.answer(
            user_input,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        print(f"\n机器人:\n{result.text}")
        print(f"[检索标注] matched={result.matched} score={result.score:.3f} matched_question={result.matched_question!r}")


if __name__ == "__main__":
    main()
