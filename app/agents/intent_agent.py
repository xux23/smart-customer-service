"""意图识别 Agent：构建 Prompt、调用 LLM、解析 JSON、四级兜底链。

兜底链：
1. LLM 返回合法 JSON -> 直接使用 (source=llm)
2. 解析/校验失败 -> 错误信息拼回 Prompt 重试 1 次 (source=llm_retry)
3. 重试仍失败 -> 关键词规则 (source=keyword_fallback)
4. LLM 调用异常（超时/网络/无 key）-> 不重试，直接关键词规则

兜底结果的置信度固定 0.4（低于转人工阈值）：
兜底触发本身说明 LLM 已不可用，宁可保守也不能把可能错误的分类继续往下传。
"""

import json
import logging
from dataclasses import dataclass

from app.core.keywords import INTENTS, KEYWORD_MAP, QUESTION_PREFIX
from app.core.llm import LLMError, create_llm_client

logger = logging.getLogger(__name__)

FALLBACK_CONFIDENCE = 0.4

INTENT_SYSTEM_PROMPT = """你是一个客服系统的意图分类器。请判断用户问题的类型，只输出一个 JSON 对象，不要输出任何其他内容。

意图类型只能是以下四种之一：
- after_sale：售后问题（退货、退款、换货、维修、物流异常、发票等）
- consultation：咨询问题（产品功能、使用方法、政策说明等）
- complaint：投诉（表达不满、明确要求投诉、服务质量差等）
- other：其他（与客服业务无关，或无法归入以上三类）

输出格式：
{"intent": "意图类型", "confidence": 0到1之间的小数, "reason": "一句话理由"}

示例：
用户：怎么申请退货？
输出：{"intent": "after_sale", "confidence": 0.95, "reason": "询问退货流程"}

用户：你们快递员态度太差了，我要投诉！
输出：{"intent": "complaint", "confidence": 0.97, "reason": "明确表达投诉意愿"}

用户：今天天气怎么样？
输出：{"intent": "other", "confidence": 0.90, "reason": "与客服业务无关"}
"""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str
    source: str


class IntentParseError(Exception):
    """LLM 输出不是合法的意图 JSON"""


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = create_llm_client()
    return _client


def classify(question: str, client=None) -> IntentResult:
    # client 参数供测试注入假 LLM；正常运行走模块级单例
    if client is None:
        client = _get_client()
    user_prompt = f"{QUESTION_PREFIX}{question}"
    try:
        text = client.chat(INTENT_SYSTEM_PROMPT, user_prompt)
        result = _parse_intent(text)
        result.source = "llm"
        return result
    except IntentParseError as exc:
        # 把错误信息拼回 Prompt 再给模型一次机会，仍失败才走关键词规则
        logger.warning("意图 JSON 首次解析失败，携带错误信息重试：%s", exc)
        retry_prompt = (
            f"{user_prompt}\n"
            f"注意：你上次的输出无法解析（{exc}）。请严格只输出一个 JSON 对象，"
            "不要输出任何其他内容。"
        )
        try:
            text = client.chat(INTENT_SYSTEM_PROMPT, retry_prompt)
            result = _parse_intent(text)
            result.source = "llm_retry"
            return result
        except IntentParseError as retry_exc:
            logger.warning("意图 JSON 重试仍失败：%s", retry_exc)
        except LLMError as retry_exc:
            logger.warning("意图识别重试调用异常：%s", retry_exc)
    except LLMError as exc:
        # 调用异常不重试（大概率还是失败），直接走关键词规则
        logger.warning("意图识别调用异常，直接关键词兜底：%s", exc)
    return _keyword_fallback(question)


def _parse_intent(text: str) -> IntentResult:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise IntentParseError(f"输出中没有 JSON 对象: {text[:80]}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"JSON 解析失败: {exc}") from exc

    intent = data.get("intent")
    confidence = data.get("confidence")
    if intent not in INTENTS or not isinstance(confidence, (int, float)):
        raise IntentParseError(f"intent 或 confidence 字段非法: {data}")
    if confidence < 0.0 or confidence > 1.0:
        raise IntentParseError(f"confidence 超出 0~1 范围: {confidence}")
    return IntentResult(
        intent=intent,
        confidence=float(confidence),
        reason=str(data.get("reason", "")),
        source="",
    )


def _keyword_fallback(question: str) -> IntentResult:
    for intent, words in KEYWORD_MAP:
        for word in words:
            if word in question:
                return IntentResult(
                    intent=intent,
                    confidence=FALLBACK_CONFIDENCE,
                    reason=f"关键词「{word}」命中规则",
                    source="keyword_fallback",
                )
    return IntentResult(
        intent="other",
        confidence=FALLBACK_CONFIDENCE,
        reason="未命中任何业务关键词",
        source="keyword_fallback",
    )
