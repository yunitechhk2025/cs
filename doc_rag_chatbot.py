"""基于产品说明文档（而非固定问答题库）的 RAG 客服机器人。

与 excel_rag_chatbot.ExcelFaqRagBot 的区别：
- FAQ 题库场景：问题必须命中某条现成的"问题-答案"，专业内容原文照搬，逐字不改写。
- 本模块场景：产品只有一份说明文档（成分/主治/用法等），没有现成的问答对，需要 AI
  根据检索到的相关段落，现场生成措辞自然的回答——但内容必须严格限定在文档范围内，
  文档没提到的信息（禁忌、孕妇能否使用等）一律不得回答，交由 AI 明确声明"未找到依据"，
  上层业务逻辑据此转人工，行为与 FAQ 题库未命中时完全一致。
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from excel_rag_chatbot import AnswerResult, calibrate_lexical, calibrate_semantic
from kb_desensitize import desensitize

NO_ANSWER_TOKEN = "NO_ANSWER"


@dataclass
class DocChunk:
    title: str
    text: str
    # 与 FaqItem 保持同样的字段命名习惯，方便复用 web_app.py 里通用的检索标注/共用问题逻辑：
    # question 对应"命中的段落标题"，answer 对应该段落正文。
    shared: bool = False

    @property
    def question(self) -> str:
        return self.title

    @property
    def answer(self) -> str:
        return self.text

    def kb_text(self) -> str:
        return f"{self.title}\n{self.text}"


class DocRagBot:
    def __init__(self, doc_path: str, top_k: int = 4, min_score: float = 0.12) -> None:
        self.doc_path = doc_path
        self.top_k = top_k
        self.min_score = min_score
        self.items: List[DocChunk] = []

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.doc_vectors = None

        self.embed_model = os.getenv("FAQ_EMBED_MODEL", "text-embedding-v3")
        self.item_embeddings: Optional[np.ndarray] = None

    # ---------------- 文档解析：按 Markdown 二级标题（## ）切成段落 ----------------

    def _parse_doc(self, raw_text: str) -> List[DocChunk]:
        sections = re.split(r"\n##\s+", raw_text)
        chunks: List[DocChunk] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split("\n", 1)
            title = lines[0].strip().lstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if not title or not body:
                continue
            # 顶部说明性引用块（> 开头）不是产品事实性内容，跳过，避免被当成可检索的"事实段落"
            body = "\n".join(ln for ln in body.split("\n") if not ln.strip().startswith(">")).strip()
            if not body:
                continue
            chunks.append(DocChunk(title=title, text=body))
        return chunks

    def build_index(self) -> None:
        with open(self.doc_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 脱敏版：文档原文里的真实品牌名/创始人姓名先替换掉再解析，AI 只能看到脱敏后的资料，
        # 也就不可能把真实品牌写进回答里（见 kb_desensitize.py）。
        raw_text = desensitize(raw_text)

        items = self._parse_doc(raw_text)
        if not items:
            raise ValueError(f"未能从文档中解析出任何段落: {self.doc_path}")

        self.items = items
        self._finalize_index()

    def load_extra_items(self, extra_items: List) -> None:
        """合并跨产品共用条目（如品牌类问题），与 ExcelFaqRagBot 保持一致的接口。"""
        existing_keys = {(it.question, it.answer) for it in self.items}
        added = False
        for it in extra_items:
            key = (it.question, it.answer)
            if key in existing_keys:
                continue
            self.items.append(DocChunk(title=it.question, text=it.answer, shared=True))
            existing_keys.add(key)
            added = True
        if added:
            self._finalize_index()

    def _finalize_index(self) -> None:
        corpus = [f"{it.title} {it.title} {it.text}" for it in self.items]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self._build_semantic_index()

    def _get_ai_client(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # 复用进程内缓存的客户端（见 ai_client.py），避免每次调用都重做 TLS 握手
        from ai_client import get_ai_client

        return get_ai_client(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=2)

    def _build_semantic_index(self, batch_size: int = 10, attempts: int = 3) -> None:
        """为每个段落计算一次语义向量，弥补字符检索对同义/口语化改写问句召回不足的问题。
        文档段落数量很少，即使全部失败也只是静默降级为关键词检索，不影响主流程。"""
        self.item_embeddings = None
        texts = [f"{it.title}：{it.text}" for it in self.items]

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
                    f"[warn] 文档语义向量索引构建失败（第 {attempt}/{attempts} 次尝试）: {exc}",
                    file=sys.stderr,
                )

        print("[warn] 文档语义向量索引最终构建失败，将仅使用关键词检索。", file=sys.stderr)
        self.item_embeddings = None

    def _semantic_scores(
        self, user_query: str, base_url: Optional[str] = None, api_key: Optional[str] = None, attempts: int = 2
    ) -> Optional[np.ndarray]:
        """实时调 embedding 接口算语义相似度：偶尔会因网络抖动/超时/限流失败，失败后会静默降级
        成纯关键词检索——同一个问题连续问两次，可能一次调用成功命中、另一次恰好失败退化成
        未命中，导致识别结果不一致。这里加一次快速重试，尽量把这种偶发失败盖住。"""
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
        print(f"[warn] 文档语义检索调用失败（已重试{attempts}次），本次仅使用关键词检索: {last_exc}", file=sys.stderr)
        return None

    def retrieve(
        self, user_query: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> List[Tuple[float, DocChunk]]:
        if self.doc_vectors is None:
            raise RuntimeError("索引尚未构建，请先执行 build_index()。")

        q_vec = self.vectorizer.transform([user_query])
        lexical_scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        semantic_scores = self._semantic_scores(user_query, base_url=base_url, api_key=api_key)

        n = len(self.items)
        k = min(self.top_k, n)

        def combined_score(i: int) -> float:
            score = float(lexical_scores[i])
            if semantic_scores is not None:
                score = max(score, float(semantic_scores[i]))
            return score

        ranked = sorted(
            [(combined_score(i), self.items[i]) for i in range(n)],
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[:k]

    def _not_found_text(self) -> str:
        return (
            "亲，这个问题目前的产品资料里暂时没有明确说明，已为您转接人工客服，请稍候~"
        )

    def _call_ai_answer(
        self,
        user_query: str,
        context_chunks: Sequence[Tuple[float, DocChunk]],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        context = "\n\n".join(f"【{item.title}】\n{item.text}" for _, item in context_chunks)
        system_prompt = (
            "你是 YUNI 10%碳酰二胺护手乳霜（10%尿素护手霜）的客服助手。\n"
            "你只能依据下面提供的《产品资料》回答用户问题，严禁使用产品资料之外的任何知识、常识或推测来补充、"
            "延伸答案，也不能给出医疗诊断、用药调整建议或任何资料未明确提及的健康/安全结论。\n\n"
            f"《产品资料》：\n{context}\n\n"
            "判断与回答规则：\n"
            "1. 如果《产品资料》中有明确内容可以直接回答用户问题，请用简洁、亲切、口语化的语气回答，"
            "内容必须完全基于资料，不能添加资料中没有的信息，不能夸大或弱化资料原意。\n"
            "2. 如果用户问题《产品资料》没有明确涉及（即使话题相关，但资料没有直接说明，比如孕妇/哺乳期能否使用、"
            "是否可与其他药物同用、具体人群禁忌等），你必须只输出："
            f"{NO_ANSWER_TOKEN}\n"
            "不要输出任何其他文字、不要道歉、不要解释原因，只输出这一个词。\n"
            "3. 不要编造、不要猜测、不要基于常识或医学常识补充说明。"
        )

        client = self._get_ai_client(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def answer(
        self,
        user_query: str,
        model: str = "qwen3.6-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AnswerResult:
        ranked = self.retrieve(user_query, base_url=base_url, api_key=api_key)
        if not ranked:
            return AnswerResult(text="抱歉，产品资料为空或尚未建立索引。", matched=False, score=0.0)

        best_score = ranked[0][0]
        # 未命中时也给出最高候选的证据分（校准后 0-1 尺度），供工作台展示参考
        miss_confidence = round(max(calibrate_lexical(best_score), calibrate_semantic(best_score)), 4)
        if best_score < self.min_score:
            return AnswerResult(
                text=self._not_found_text(), matched=False, score=best_score,
                confidence=miss_confidence, evidence=miss_confidence,
            )

        try:
            raw_answer = self._call_ai_answer(
                user_query=user_query, context_chunks=ranked, model=model, base_url=base_url, api_key=api_key
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 文档RAG生成回复失败: {exc}", file=sys.stderr)
            return AnswerResult(
                text=self._not_found_text(), matched=False, score=best_score,
                confidence=miss_confidence, evidence=miss_confidence,
            )

        if not raw_answer or NO_ANSWER_TOKEN in raw_answer:
            return AnswerResult(
                text=self._not_found_text(), matched=False, score=best_score,
                confidence=miss_confidence, evidence=miss_confidence,
            )

        matched_titles = "、".join(item.title for _, item in ranked if _ >= self.min_score * 0.6)
        matched_texts = "\n\n".join(item.text for _, item in ranked if _ >= self.min_score * 0.6)
        # 文档 RAG 的"能否作答"由生成时的严格判断把关（资料没明确涉及就输出 NO_ANSWER_TOKEN），
        # 走到这里说明 AI 已确认资料能回答，给 0.5 的基础置信度，再按检索证据往上加；
        # ranked 里只有两路取较大值后的综合分，无法区分来自哪一路，取两种校准的较大值近似。
        evidence = round(max(calibrate_lexical(best_score), calibrate_semantic(best_score)), 4)
        confidence = round(min(0.99, 0.5 + 0.5 * evidence), 4)
        return AnswerResult(
            text=raw_answer,
            matched=True,
            score=best_score,
            matched_question=matched_titles or ranked[0][1].title,
            matched_answer=matched_texts or ranked[0][1].text,
            confidence=confidence,
            evidence=evidence,
        )
