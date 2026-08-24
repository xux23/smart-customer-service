"""SQLite 访问：建表、工单 CRUD、工单号生成、事件流水。

标准库 sqlite3 直连、裸 SQL，不用 ORM：
表只有两张、查询简单，直观可读。
"""

import os
import sqlite3
import threading
from datetime import datetime

from app.config import get_settings

_conn: sqlite3.Connection | None = None
_id_lock = threading.Lock()

DDL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    question    TEXT NOT NULL,
    intent      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    department  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING',
    handler     TEXT NOT NULL DEFAULT '',
    resolution  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ticket_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def init_db() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
    db_path = get_settings().db_path
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False)
    # Row 工厂让查询结果能按列名取值，转 dict 后直接给 API 用
    _conn.row_factory = sqlite3.Row
    _conn.executescript(DDL)
    _conn.commit()


def _connection() -> sqlite3.Connection:
    if _conn is None:
        init_db()
    return _conn


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def insert_ticket(ticket: dict) -> None:
    _connection().execute(
        "INSERT INTO tickets (ticket_id, session_id, question, intent, confidence,"
        " department, status, handler, resolution, created_at, updated_at)"
        " VALUES (:ticket_id, :session_id, :question, :intent, :confidence,"
        " :department, :status, :handler, :resolution, :created_at, :updated_at)",
        ticket,
    )
    _connection().commit()


def get_ticket(ticket_id: str) -> dict | None:
    row = _connection().execute(
        "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    return dict(row) if row else None


def list_tickets(status: str | None = None, department: str | None = None) -> list[dict]:
    sql = "SELECT * FROM tickets WHERE 1=1"
    params: list[str] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if department:
        sql += " AND department = ?"
        params.append(department)
    sql += " ORDER BY created_at DESC, ticket_id DESC"
    rows = _connection().execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def update_ticket(ticket_id: str, fields: dict) -> None:
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values())
    values.append(ticket_id)
    _connection().execute(
        f"UPDATE tickets SET {assignments} WHERE ticket_id = ?", values
    )
    _connection().commit()


def next_ticket_id() -> str:
    """工单号规则：T + 当天日期 + 当日 4 位序号。

    单进程内用锁保证并发不重号；分布式场景应换数据库序列或号段模式。
    """
    with _id_lock:
        date_part = datetime.now().strftime("%Y%m%d")
        prefix = f"T{date_part}-"
        row = _connection().execute(
            "SELECT ticket_id FROM tickets WHERE ticket_id LIKE ?"
            " ORDER BY ticket_id DESC LIMIT 1",
            (prefix + "%",),
        ).fetchone()
        last_seq = int(row["ticket_id"].split("-")[1]) if row else 0
        return f"{prefix}{last_seq + 1:04d}"


def add_event(ticket_id: str, action: str, detail: str) -> None:
    _connection().execute(
        "INSERT INTO ticket_events (ticket_id, action, detail, created_at)"
        " VALUES (?, ?, ?, ?)",
        (ticket_id, action, detail, now_iso()),
    )
    _connection().commit()


def get_events(ticket_id: str) -> list[dict]:
    rows = _connection().execute(
        "SELECT action, detail, created_at FROM ticket_events"
        " WHERE ticket_id = ? ORDER BY id ASC",
        (ticket_id,),
    ).fetchall()
    return [dict(row) for row in rows]
