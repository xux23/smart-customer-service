"""编排三个 Agent 的完整流程，管理会话候选暂存。

编排逻辑全部集中在 pipeline，Agent 之间零依赖，
这是它们能独立单测的关键。
任何需要建单的分支都统一走工单 Agent：建单 + 按意图部门路由 + 初始 PENDING。
"""

import logging

from app.agents import intent_agent, retrieval_agent, ticket_agent
from app.config import get_settings

logger = logging.getLogger(__name__)

# 会话候选暂存：session_id -> {"candidates": [...], "question"/"intent"/...}
# 演示级简化：进程内存，重启丢失、仅单进程有效；正式做法应放 Redis 并加过期时间。
session_state: dict[str, dict] = {}

COMFORT_REPLY = (
    "非常抱歉给您带来不好的体验，已为您创建工单 {ticket_id}，"
    "已转交{department}处理，会有专人尽快与您联系。"
)
HUMAN_REPLY = "不好意思，我没能在客服范围内理解您的问题，已为您转人工处理（工单 {ticket_id}）。"
TICKET_REPLY_TEMPLATES = {
    "unknown": HUMAN_REPLY,
    "complaint": COMFORT_REPLY,
    "other": HUMAN_REPLY,
}
SIMILAR_REPLY = "没有找到完全匹配的答案，您想问的是不是以下问题？回复序号选择，回复 0 表示都不是。"


def handle_message(session_id: str, message: str) -> dict:
    intent_result = intent_agent.classify(message)
    logger.info(
        "意图识别: intent=%s confidence=%.2f source=%s",
        intent_result.intent, intent_result.confidence, intent_result.source,
    )
    threshold = get_settings().intent_confidence_threshold

    # 分类不可信时宁可转人工，不可错分
    if intent_result.confidence < threshold:
        return _create_ticket_response(
            session_id, message, "unknown",
            intent_result.confidence, intent_result.source,
        )
    if intent_result.intent == "complaint":
        # 投诉是情绪和责任问题，不走知识库自动回答，直接转质控组人工介入
        return _create_ticket_response(
            session_id, message, "complaint",
            intent_result.confidence, intent_result.source,
        )
    if intent_result.intent == "other":
        return _create_ticket_response(
            session_id, message, "other",
            intent_result.confidence, intent_result.source,
        )

    route, results, best_score = retrieval_agent.retrieve(message)
    logger.info("检索路由: %s 最高分=%.4f", route, best_score)

    if route == retrieval_agent.ROUTE_KB_ANSWER:
        entry, score = results[0]
        return {
            "route": "kb_answer",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "intent_source": intent_result.source,
            "reply": entry["answer"],
            "matched_question": entry["question"],
            "score": round(score, 4),
        }

    if route == retrieval_agent.ROUTE_SIMILAR_QUESTIONS:
        candidates = [
            {
                "question_id": entry["id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "score": round(score, 4),
            }
            for entry, score in results
        ]
        session_state[session_id] = {
            "candidates": candidates,
            "question": message,
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "source": intent_result.source,
        }
        return {
            "route": "similar_questions",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "intent_source": intent_result.source,
            "reply": SIMILAR_REPLY,
            "similar_questions": [
                {"question_id": item["question_id"], "question": item["question"]}
                for item in candidates
            ],
        }

    return _create_ticket_response(
        session_id, message, intent_result.intent,
        intent_result.confidence, intent_result.source,
    )


def handle_choose(session_id: str, question_id: int | None) -> dict:
    state = session_state.get(session_id)
    if state is None:
        raise ValueError("该会话没有待选择的相似问题")

    if question_id is None:
        # 用户放弃选择：按原始意图建工单转人工
        session_state.pop(session_id, None)
        return _create_ticket_response(
            session_id, state["question"], state["intent"],
            state["confidence"], state["source"],
        )

    for candidate in state["candidates"]:
        if candidate["question_id"] == question_id:
            session_state.pop(session_id, None)
            return {
                "route": "kb_answer",
                "intent": state["intent"],
                "confidence": state["confidence"],
                "intent_source": state["source"],
                "reply": candidate["answer"],
                "matched_question": candidate["question"],
                "score": candidate["score"],
            }
    raise ValueError("无效的问题选择")


def _create_ticket_response(
    session_id: str,
    question: str,
    intent: str,
    confidence: float,
    source: str,
) -> dict:
    ticket = ticket_agent.create_ticket(session_id, question, intent, confidence)
    logger.info(
        "工单创建: %s intent=%s department=%s",
        ticket["ticket_id"], intent, ticket["department"],
    )
    reply_template = TICKET_REPLY_TEMPLATES.get(intent, HUMAN_REPLY)
    reply = reply_template.format(ticket_id=ticket["ticket_id"], department=ticket["department"])
    return {
        # 意图不明单独标记为 human 路由，其余建单场景都是 ticket_created
        "route": "human" if intent == "unknown" else "ticket_created",
        "intent": intent,
        "confidence": confidence,
        "intent_source": source,
        "reply": reply,
        "ticket_id": ticket["ticket_id"],
        "department": ticket["department"],
    }
