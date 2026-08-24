"""集中管理所有配置项，其他模块只从这里读配置，不直接碰环境变量。"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    intent_confidence_threshold: float
    kb_match_threshold: float
    kb_similar_threshold: float
    kb_top_n: int
    kb_path: str
    db_path: str


def _load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "glm-4-flash"),
        intent_confidence_threshold=float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6")),
        kb_match_threshold=float(os.getenv("KB_MATCH_THRESHOLD", "0.55")),
        kb_similar_threshold=float(os.getenv("KB_SIMILAR_THRESHOLD", "0.30")),
        kb_top_n=int(os.getenv("KB_TOP_N", "3")),
        kb_path=os.getenv("KB_PATH", "app/data/knowledge.json"),
        db_path=os.getenv("DB_PATH", "tickets.db"),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings
