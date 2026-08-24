"""工单 Agent 测试：建单、部门路由、状态机、审计事件"""

import re

import pytest

from app.agents import ticket_agent
from app.agents.ticket_agent import InvalidTransition, TicketNotFoundError
from app.core import store


@pytest.mark.parametrize(
    "intent, department",
    [
        ("after_sale", "售后服务部"),
        ("complaint", "客服质控组"),
        ("consultation", "在线客服组"),
        ("other", "综合受理组"),
        ("unknown", "综合受理组"),
    ],
)
def test_create_ticket_routes_department(fresh_db, intent, department):
    ticket = ticket_agent.create_ticket("s1", "问题内容", intent, 0.9)
    assert ticket["department"] == department
    assert ticket["status"] == "PENDING"
    assert ticket["handler"] == ""
    assert re.fullmatch(r"T\d{8}-\d{4}", ticket["ticket_id"])
    events = store.get_events(ticket["ticket_id"])
    assert len(events) == 1
    assert events[0]["action"] == "created"


def test_ticket_id_increments_within_same_day(fresh_db):
    first = ticket_agent.create_ticket("s1", "问题一", "complaint", 0.9)
    second = ticket_agent.create_ticket("s1", "问题二", "complaint", 0.9)
    first_seq = int(first["ticket_id"].split("-")[1])
    second_seq = int(second["ticket_id"].split("-")[1])
    assert second_seq == first_seq + 1


def test_full_lifecycle_assign_then_resolve(fresh_db):
    ticket = ticket_agent.create_ticket("s1", "包裹坏了", "after_sale", 0.9)

    assigned = ticket_agent.transfer(ticket["ticket_id"], "assign", "张三")
    assert assigned["status"] == "PROCESSING"
    assert assigned["handler"] == "张三"

    resolved = ticket_agent.transfer(
        ticket["ticket_id"], "resolve", "张三", "已补发商品"
    )
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolution"] == "已补发商品"

    actions = [event["action"] for event in store.get_events(ticket["ticket_id"])]
    assert actions == ["created", "assigned", "resolved"]


def test_resolve_pending_ticket_rejected_and_audited(fresh_db):
    ticket = ticket_agent.create_ticket("s1", "包裹坏了", "after_sale", 0.9)
    with pytest.raises(InvalidTransition):
        ticket_agent.transfer(ticket["ticket_id"], "resolve", "李四", "直接解决")
    # 状态不变，且记录了非法操作审计事件
    assert store.get_ticket(ticket["ticket_id"])["status"] == "PENDING"
    events = store.get_events(ticket["ticket_id"])
    assert events[-1]["action"] == "invalid_attempt"
    assert "resolve" in events[-1]["detail"]
    assert "李四" in events[-1]["detail"]


def test_assign_resolved_ticket_rejected_and_audited(fresh_db):
    ticket = ticket_agent.create_ticket("s1", "问题", "other", 0.8)
    ticket_agent.transfer(ticket["ticket_id"], "assign", "张三")
    ticket_agent.transfer(ticket["ticket_id"], "resolve", "张三", "完成")
    with pytest.raises(InvalidTransition):
        ticket_agent.transfer(ticket["ticket_id"], "assign", "王五")
    events = store.get_events(ticket["ticket_id"])
    assert events[-1]["action"] == "invalid_attempt"


def test_missing_ticket_raises_not_found(fresh_db):
    with pytest.raises(TicketNotFoundError):
        ticket_agent.transfer("T20990101-9999", "assign", "张三")
