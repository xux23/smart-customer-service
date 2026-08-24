"""意图识别 Agent 测试：JSON 解析、四级兜底链、关键词规则"""

import json

import pytest

from app.agents import intent_agent
from app.agents.intent_agent import IntentParseError, _keyword_fallback, _parse_intent
from app.core.llm import LLMError


class FakeLLMClient:
    """按脚本依次返回内容；raise_exc=True 时模拟 LLM 调用异常"""

    def __init__(self, replies, raise_exc=False):
        self.replies = list(replies)
        self.raise_exc = raise_exc
        self.call_count = 0

    def chat(self, system_prompt, user_prompt):
        self.call_count += 1
        if self.raise_exc:
            raise LLMError("网络异常")
        return self.replies.pop(0)


def _intent_json(intent="complaint", confidence=0.97, reason="测试"):
    return json.dumps(
        {"intent": intent, "confidence": confidence, "reason": reason},
        ensure_ascii=False,
    )


def test_valid_json_used_directly():
    client = FakeLLMClient([_intent_json()])
    result = intent_agent.classify("快递员态度太差", client)
    assert result.intent == "complaint"
    assert result.confidence == 0.97
    assert result.source == "llm"
    assert client.call_count == 1


def test_json_with_surrounding_text_parses():
    text = '好的，结果是 {"intent": "other", "confidence": 0.9, "reason": "无关"} 请查收'
    result = _parse_intent(text)
    assert result.intent == "other"


def test_non_json_retry_once_succeeds():
    client = FakeLLMClient(["这是废话不是json", _intent_json(intent="after_sale")])
    result = intent_agent.classify("怎么退货", client)
    assert result.intent == "after_sale"
    assert result.source == "llm_retry"
    assert client.call_count == 2


def test_retry_still_fails_uses_keyword():
    client = FakeLLMClient(["还是不是json", "依然不是json"])
    result = intent_agent.classify("怎么退货", client)
    assert result.source == "keyword_fallback"
    assert result.confidence == pytest.approx(0.4)
    assert client.call_count == 2


def test_intent_outside_enum_is_parse_failure():
    client = FakeLLMClient([
        _intent_json(intent="finance"),
        _intent_json(intent="consultation"),
    ])
    result = intent_agent.classify("支持哪些支付方式", client)
    assert result.intent == "consultation"
    assert result.source == "llm_retry"


def test_confidence_out_of_range_is_parse_failure():
    with pytest.raises(IntentParseError):
        _parse_intent(_intent_json(confidence=1.5))


def test_llm_exception_skips_retry_direct_keyword():
    client = FakeLLMClient([], raise_exc=True)
    result = intent_agent.classify("怎么退货", client)
    assert result.source == "keyword_fallback"
    # 调用异常不重试，只调了一次
    assert client.call_count == 1


def test_keyword_hits_by_priority():
    # 售后词与疑问词同时出现：售后优先（KEYWORD_MAP 顺序）
    assert _keyword_fallback("怎么申请退货").intent == "after_sale"
    assert _keyword_fallback("我要投诉").intent == "complaint"
    assert _keyword_fallback("请问怎么开发票").intent == "after_sale"


def test_keyword_miss_all_returns_other():
    result = _keyword_fallback("你好呀")
    assert result.intent == "other"
    assert result.confidence == pytest.approx(0.4)
    assert result.source == "keyword_fallback"
