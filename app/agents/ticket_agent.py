"""工单处理 Agent：创建、部门路由、状态机流转、查询。

状态机设计要点：
- ALLOWED_TRANSITIONS 是全系统唯一的合法流转依据，判断逻辑只有一处；
- 不允许跳过"处理中"直接解决，保证每张工单都有明确承接人；
- 非法流转拒绝并写 invalid_attempt 审计事件，可追溯。
要扩展重开、转派，只需在流转表中加一行，机制不变。
"""

from app.core import store


class TicketNotFoundError(Exception):
    """工单不存在"""


class InvalidTransition(Exception):
    """非法的工单状态流转"""


ALLOWED_TRANSITIONS = {
    ("PENDING", "assign"): "PROCESSING",
    ("PROCESSING", "resolve"): "RESOLVED",
}

# 动作名到事件名的映射：事件流水用过去式（assigned/resolved），便于阅读
ACTION_EVENT_NAMES = {
    "assign": "assigned",
    "resolve": "resolved",
}

DEPARTMENT_MAP = {
    "after_sale": "售后服务部",
    "complaint": "客服质控组",
    "consultation": "在线客服组",
    "other": "综合受理组",
    "unknown": "综合受理组",
}


def create_ticket(session_id: str, question: str, intent: str, confidence: float) -> dict:
    ticket_id = store.next_ticket_id()
    now = store.now_iso()
    department = DEPARTMENT_MAP[intent]
    ticket = {
        "ticket_id": ticket_id,
        "session_id": session_id,
        "question": question,
        "intent": intent,
        "confidence": confidence,
        "department": department,
        "status": "PENDING",
        "handler": "",
        "resolution": "",
        "created_at": now,
        "updated_at": now,
    }
    store.insert_ticket(ticket)
    store.add_event(
        ticket_id,
        "created",
        f"intent={intent} confidence={confidence:.2f} -> {department}",
    )
    return ticket


def transfer(
    ticket_id: str,
    action: str,
    handler: str,
    resolution: str = "",
) -> dict:
    """按流转表执行接单/解决；非法流转记审计事件后抛异常"""
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        raise TicketNotFoundError(f"工单 {ticket_id} 不存在")

    target_status = ALLOWED_TRANSITIONS.get((ticket["status"], action))
    if target_status is None:
        detail = (
            f"处理人 {handler or '未知'} 尝试执行 {action}，"
            f"当前状态 {ticket['status']} 不允许"
        )
        store.add_event(ticket_id, "invalid_attempt", detail)
        raise InvalidTransition(
            f"工单当前状态为 {ticket['status']}，不允许执行 {action}"
        )

    fields = {"status": target_status, "handler": handler, "updated_at": store.now_iso()}
    event_detail = f"{handler} 接单"
    if action == "resolve":
        fields["resolution"] = resolution
        event_detail = resolution

    store.update_ticket(ticket_id, fields)
    store.add_event(ticket_id, ACTION_EVENT_NAMES[action], event_detail)
    return store.get_ticket(ticket_id)
