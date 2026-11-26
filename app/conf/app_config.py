"""
应用主配置

定义 conf/app_config.yaml 在程序中的结构化配置对象
项目启动后会在这里一次性完成配置文件加载和类型化转换，其他模块只需要导入 app_config
就可以按属性方式读取日志 MySQL Qdrant Embedding Elasticsearch 和 LLM 配置
"""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf


@dataclass
class File:
    """文件日志配置"""

    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    """控制台日志配置"""

    enable: bool
    level: str


@dataclass
class LoggingConfig:
    """日志总配置"""

    file: File
    console: Console


@dataclass
class DBConfig:
    """MySQL 连接配置"""

    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class QdrantConfig:
    """Qdrant 连接与向量维度配置"""

    host: str
    port: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    """Embedding 服务配置"""

    host: str
    port: int
    model: str


@dataclass
class ESConfig:
    """Elasticsearch 配置"""

    host: str
    port: int
    index_name: str


@dataclass
class LLMConfig:
    """大模型调用配置"""

    model_name: str
    api_key: str
    base_url: str


@dataclass
class RecallBudget:
    """召回阶段的超时预算，单位为秒。"""

    total_seconds: float = 8.0
    column_seconds: float = 6.0
    metric_seconds: float = 6.0
    value_seconds: float = 6.0
    expansion_seconds: float = 2.0

    def __post_init__(self):
        for value in (
            self.total_seconds,
            self.column_seconds,
            self.metric_seconds,
            self.value_seconds,
            self.expansion_seconds,
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError("Recall budgets must be positive and finite")


@dataclass
class SQLExecutionConfig:
    """Generated query limits owned by the deployment."""

    max_rows: int = 1000
    # Keep authored YAML types; the SQL policy rejects non-text values rather than coercing them.
    row_filters: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    row_scope_column_types: dict[str, dict[str, str]] = field(default_factory=dict)
    max_correction_attempts: int = 2


@dataclass
class AppConfig:
    """项目级总配置入口"""

    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig
    recall: RecallBudget = field(default_factory=RecallBudget)
    sql_execution: SQLExecutionConfig = field(default_factory=SQLExecutionConfig)


# 从当前文件位置回到项目根目录，再定位到 conf/app_config.yaml
project_root = Path(__file__).parents[2]
config_file = project_root / "conf" / "app_config.yaml"

# 先读取本地 .env，让 YAML 中的 ${oc.env:...} 可以解析到敏感配置
load_dotenv(project_root / ".env")

# 读取 YAML 配置内容
context = OmegaConf.load(config_file)

# 根据 AppConfig 生成结构化配置 schema
schema = OmegaConf.structured(AppConfig)

# 把配置结构和配置值合并，再转换成可以直接按属性访问的对象
app_config: AppConfig = OmegaConf.to_object(OmegaConf.merge(schema, context))

if __name__ == "__main__":
    # 简单测试：验证配置是否能正常读取
    print(app_config.es.host)
