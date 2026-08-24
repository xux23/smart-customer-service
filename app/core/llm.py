"""LLM 客户端：OpenAI 兼容实现 + Mock 实现 + 工厂函数。

LLM 层只负责"调一次、返回文本或抛异常"，不做重试；
重试属于意图 Agent 兜底链的职责。
"""

import json
import logging

from openai import OpenAI

from app.config import get_settings
from app.core.keywords import (
    FRUSTRATION_WORDS,
    KEYWORD_MAP,
    QUESTION_PREFIX,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用失败（网络、超时、接口错误），由调用方决定如何兜底"""


class OpenAICompatibleClient:
    """OpenAI 兼容接口的薄封装，只暴露 chat 一个方法"""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.llm_model
        # 超时给足真实模型生成时间，同时保证单次对话总耗时可预期
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=10.0,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"LLM 调用失败: {exc}") from exc


class MockLLMClient:
    """关键词规则模拟意图识别，返回固定格式 JSON，测试和离线演示用。

    与兜底链共用同一份关键词词典；区别在于 mock 模拟的是"正常工作的 LLM"，
    命中明确业务词时给出高置信度，而不是兜底链的固定低值 0.4。
    """

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        question = user_prompt.replace(QUESTION_PREFIX, "").strip()
        result = _mock_classify(question)
        return json.dumps(result, ensure_ascii=False)


def _mock_classify(question: str) -> dict:
    # 模拟真实 LLM 的判断优先级：明确投诉意愿 > 业务词 > 情绪句 > 纯疑问词。
    # 「我要投诉」不能被快递词抢走分类，「破系统」这种情绪句
    # 也不该因为带「什么」就被当成咨询。
    hit = _hit_word(question, "complaint")
    if hit is not None:
        return {"intent": "complaint", "confidence": 0.95, "reason": f"命中关键词「{hit}」"}

    hit = _hit_word(question, "after_sale")
    if hit is not None:
        return {"intent": "after_sale", "confidence": 0.9, "reason": f"命中关键词「{hit}」"}

    for word in FRUSTRATION_WORDS:
        # 有情绪但说不清业务诉求：模拟 LLM 给出低置信度，让流程自然转人工
        if word in question:
            return {"intent": "other", "confidence": 0.3, "reason": "表达不满但无法归类"}

    hit = _hit_word(question, "consultation")
    if hit is not None:
        return {"intent": "consultation", "confidence": 0.8, "reason": f"命中关键词「{hit}」"}

    return {"intent": "other", "confidence": 0.75, "reason": "未命中任何业务关键词"}


def _hit_word(question: str, intent: str) -> str | None:
    words_by_intent = dict(KEYWORD_MAP)
    for word in words_by_intent.get(intent, []):
        if word in question:
            return word
    return None


def create_llm_client():
    settings = get_settings()
    if settings.llm_provider == "mock":
        return MockLLMClient()
    if not settings.llm_api_key:
        # 未配 key 时自动降级为 mock，服务不中断
        logger.warning("LLM_API_KEY 为空，自动降级为 mock 模式")
        return MockLLMClient()
    return OpenAICompatibleClient()
