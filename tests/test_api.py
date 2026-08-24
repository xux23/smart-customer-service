"""接口测试：四种 chat 路由、相似问题选择、工单管理全流程。

全程 LLM_PROVIDER=mock（conftest 强制设置），不访问网络。
"""

import uuid


def new_session() -> str:
    return "test-" + uuid.uuid4().hex[:8]


def chat(client, message: str) -> dict:
    response = client.post("/api/chat", json={
        "session_id": new_session(), "message": message,
    })
    assert response.status_code == 200
    return response.json()


def test_chat_kb_answer(client):
    data = chat(client, "怎么申请退货？")
    assert data["route"] == "kb_answer"
    assert data["intent"] == "after_sale"
    assert data["matched_question"] == "如何申请退货"
    assert data["score"] >= 0.55


def test_chat_similar_questions_then_choose(client):
    session_id = new_session()
    response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "买的东西用了两天就坏了，能修吗？要钱吗？",
    })
    data = response.json()
    assert data["route"] == "similar_questions"
    assert len(data["similar_questions"]) == 3

    chosen_id = data["similar_questions"][0]["question_id"]
    follow_up = client.post("/api/chat/choose", json={
        "session_id": session_id, "question_id": chosen_id,
    })
    result = follow_up.json()
    assert result["route"] == "kb_answer"
    # 选择后清除暂存，再次选择视为无效
    again = client.post("/api/chat/choose", json={
        "session_id": session_id, "question_id": chosen_id,
    })
    assert again.status_code == 400


def test_choose_give_up_creates_ticket(client):
    session_id = new_session()
    chat_response = client.post("/api/chat", json={
        "session_id": session_id,
        "message": "买的东西用了两天就坏了，能修吗？要钱吗？",
    })
    assert chat_response.json()["route"] == "similar_questions"

    give_up = client.post("/api/chat/choose", json={
        "session_id": session_id, "question_id": None,
    })
    data = give_up.json()
    # 放弃选择按原始意图建单：after_sale -> 售后服务部
    assert data["route"] == "ticket_created"
    assert data["intent"] == "after_sale"
    assert data["department"] == "售后服务部"


def test_chat_complaint_direct_to_ticket(client):
    data = chat(client, "快递员把我的包裹扔坏了，我要投诉！")
    assert data["route"] == "ticket_created"
    assert data["intent"] == "complaint"
    assert data["department"] == "客服质控组"
    assert data["ticket_id"].startswith("T")


def test_chat_low_confidence_transfers_to_human(client):
    data = chat(client, "你们这什么破系统啊")
    assert data["route"] == "human"
    assert data["intent"] == "unknown"
    assert data["department"] == "综合受理组"


def test_choose_without_stored_candidates_rejected(client):
    response = client.post("/api/chat/choose", json={
        "session_id": "never-asked", "question_id": 1,
    })
    assert response.status_code == 400


def test_choose_invalid_question_id_rejected(client):
    session_id = new_session()
    client.post("/api/chat", json={
        "session_id": session_id,
        "message": "买的东西用了两天就坏了，能修吗？要钱吗？",
    })
    response = client.post("/api/chat/choose", json={
        "session_id": session_id, "question_id": 99999,
    })
    assert response.status_code == 400


def test_ticket_lifecycle_and_filters(client):
    created = chat(client, "快递员把我的包裹扔坏了，我要投诉！")
    ticket_id = created["ticket_id"]

    pending = client.get("/api/tickets", params={"status": "PENDING"}).json()["tickets"]
    assert any(t["ticket_id"] == ticket_id for t in pending)

    by_department = client.get(
        "/api/tickets", params={"department": "客服质控组"}
    ).json()["tickets"]
    assert any(t["ticket_id"] == ticket_id for t in by_department)

    assigned = client.post(f"/api/tickets/{ticket_id}/assign", json={"handler": "张三"})
    assert assigned.status_code == 200
    assert assigned.json()["status"] == "PROCESSING"

    resolved = client.post(
        f"/api/tickets/{ticket_id}/resolve",
        json={"handler": "张三", "resolution": "已电话致歉并补发商品"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "RESOLVED"

    done = client.get("/api/tickets", params={"status": "RESOLVED"}).json()["tickets"]
    assert any(t["ticket_id"] == ticket_id for t in done)

    detail = client.get(f"/api/tickets/{ticket_id}").json()
    actions = [event["action"] for event in detail["events"]]
    assert actions == ["created", "assigned", "resolved"]
    assert detail["ticket"]["resolution"] == "已电话致歉并补发商品"


def test_resolve_pending_returns_400_with_audit_event(client):
    created = chat(client, "快递员把我的包裹扔坏了，我要投诉！")
    ticket_id = created["ticket_id"]

    response = client.post(
        f"/api/tickets/{ticket_id}/resolve",
        json={"handler": "李四", "resolution": "想跳过接单"},
    )
    assert response.status_code == 400
    assert "PENDING" in response.json()["detail"]

    events = client.get(f"/api/tickets/{ticket_id}").json()["events"]
    assert events[-1]["action"] == "invalid_attempt"


def test_missing_ticket_returns_404(client):
    assert client.get("/api/tickets/T20990101-9999").status_code == 404
    missing_assign = client.post(
        "/api/tickets/T20990101-9999/assign", json={"handler": "张三"}
    )
    assert missing_assign.status_code == 404
