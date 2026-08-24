"""知识检索 Agent：调用检索核心，按三级阈值输出路由结果。

三级路由（检索层绝不"硬答"）：
- 最高分 >= KB_MATCH_THRESHOLD      -> kb_answer，直接回答
- 最高分 >= KB_SIMILAR_THRESHOLD    -> similar_questions，返回 top3 相似问题
- 否则                              -> no_match，交给工单 Agent 建单转人工
"""

from app.config import get_settings
from app.core import kb

ROUTE_KB_ANSWER = "kb_answer"
ROUTE_SIMILAR_QUESTIONS = "similar_questions"
ROUTE_NO_MATCH = "no_match"


def retrieve(question: str) -> tuple[str, list[tuple[dict, float]], float]:
    """返回 (路由结果, [(知识条目, 相似度分数)], 最高相似度分数)"""
    settings = get_settings()
    scored = kb.search(question, settings.kb_top_n)
    if not scored:
        return ROUTE_NO_MATCH, [], 0.0

    (_best_entry, best_score) = scored[0]
    if best_score >= settings.kb_match_threshold:
        return ROUTE_KB_ANSWER, [scored[0]], best_score
    if best_score >= settings.kb_similar_threshold:
        return ROUTE_SIMILAR_QUESTIONS, scored, best_score
    return ROUTE_NO_MATCH, scored, best_score
