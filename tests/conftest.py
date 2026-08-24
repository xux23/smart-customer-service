"""公共 fixture：强制 mock 模式、临时数据库。

测试全程不碰网络；环境变量必须在导入任何 app 模块之前设置好，
所以写在模块顶层而不是 fixture 里。
"""

import os
import tempfile

_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "scs_test_tickets.db")

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DB_PATH"] = _TEST_DB_PATH

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import pipeline, store  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def fresh_db():
    """每个工单相关测试都从空库开始，互不串数据"""
    if os.path.exists(_TEST_DB_PATH):
        os.remove(_TEST_DB_PATH)
    store.init_db()
    yield


@pytest.fixture(autouse=True)
def clean_session_candidates():
    pipeline.session_state.clear()
    yield
    pipeline.session_state.clear()


@pytest.fixture()
def client(fresh_db):
    # 上下文管理器触发 lifespan：建表 + 加载知识库
    with TestClient(app) as test_client:
        yield test_client
