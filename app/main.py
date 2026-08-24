"""FastAPI 路由：对话两个接口 + 工单四个接口。

错误约定：非法状态流转/无效选择 -> 400，工单不存在 -> 404，
错误信息统一 {"detail": "..."}（FastAPI 默认风格）。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents import ticket_agent
from app.agents.ticket_agent import InvalidTransition, TicketNotFoundError
from app.config import get_settings
from app.core import kb, pipeline, store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    store.init_db()
    entry_count = kb.load_knowledge(settings.kb_path)
    logger.info("服务启动: 数据库=%s 知识库=%s(%d条)", settings.db_path, settings.kb_path, entry_count)
    yield


app = FastAPI(title="智能客服工单系统", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChooseRequest(BaseModel):
    session_id: str
    question_id: int | None = None


class AssignRequest(BaseModel):
    handler: str


class ResolveRequest(BaseModel):
    handler: str
    resolution: str


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    return pipeline.handle_message(request.session_id, request.message)


@app.post("/api/chat/choose")
def choose(request: ChooseRequest) -> dict:
    try:
        return pipeline.handle_choose(request.session_id, request.question_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tickets")
def list_tickets(status: str | None = None, department: str | None = None) -> dict:
    return {"tickets": store.list_tickets(status=status, department=department)}


@app.get("/api/tickets/{ticket_id}")
def ticket_detail(ticket_id: str) -> dict:
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")
    return {"ticket": ticket, "events": store.get_events(ticket_id)}


@app.post("/api/tickets/{ticket_id}/assign")
def assign_ticket(ticket_id: str, request: AssignRequest) -> dict:
    return _do_transfer(ticket_id, "assign", request.handler)


@app.post("/api/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str, request: ResolveRequest) -> dict:
    return _do_transfer(ticket_id, "resolve", request.handler, request.resolution)


def _do_transfer(ticket_id: str, action: str, handler: str, resolution: str = "") -> dict:
    try:
        return ticket_agent.transfer(ticket_id, action, handler, resolution)
    except TicketNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidTransition as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
