"""命令行演示客户端：终端问答循环。

输入问题 -> 展示意图、置信度、路由结果；
返回相似问题列表时按编号选择，0 表示都不是；q 退出。
"""

import os
import sys

import httpx

API_BASE = os.getenv("CHAT_API_BASE", "http://127.0.0.1:8000")


def main() -> None:
    print(f"智能客服演示客户端（服务地址 {API_BASE}，输入 q 退出）")
    session_id = "cli-" + os.urandom(4).hex()
    with httpx.Client(base_url=API_BASE, timeout=15) as client:
        while True:
            try:
                message = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.lower() == "q":
                break
            try:
                response = client.post("/api/chat", json={
                    "session_id": session_id,
                    "message": message,
                })
            except httpx.ConnectError:
                print("连接不上服务，请先启动：uv run uvicorn app.main:app --port 8000")
                break
            _print_chat_response(client, session_id, response)


def _print_chat_response(client: httpx.Client, session_id: str, response: httpx.Response) -> None:
    if response.status_code != 200:
        print(f"[请求失败 {response.status_code}] {response.json().get('detail', '')}")
        return
    data = response.json()
    print(
        f"意图: {data['intent']}（置信度 {data['confidence']:.2f}，来源 {data['intent_source']}）"
        f" -> 路由: {data['route']}"
    )
    if data["route"] == "similar_questions":
        for index, item in enumerate(data["similar_questions"], start=1):
            print(f"  {index}. {item['question']}")
        choice = _ask_choice(len(data["similar_questions"]))
        # 0 表示都不是，对应 question_id=null
        chosen_id = None if choice == 0 else data["similar_questions"][choice - 1]["question_id"]
        follow_up = client.post("/api/chat/choose", json={
            "session_id": session_id,
            "question_id": chosen_id,
        })
        _print_choose_response(follow_up)
        return

    print(f"客服: {data['reply']}")
    if data["route"] in ("ticket_created", "human"):
        print(f"     工单号 {data['ticket_id']}，部门 {data['department']}")


def _print_choose_response(response: httpx.Response) -> None:
    if response.status_code != 200:
        print(f"[请求失败 {response.status_code}] {response.json().get('detail', '')}")
        return
    data = response.json()
    print(f"客服: {data['reply']}")
    if data["route"] in ("ticket_created", "human"):
        print(f"     工单号 {data['ticket_id']}，部门 {data['department']}")


def _ask_choice(total: int) -> int:
    while True:
        raw = input(f"请输入序号选择（1-{total}），输入 0 表示都不是: ").strip()
        if raw.isdigit() and 0 <= int(raw) <= total:
            return int(raw)
        print("输入无效，请重新输入。")


if __name__ == "__main__":
    sys.exit(main())
