"""Read-only live adapters. Importing this module does not import the app."""

import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

from langchain_core.callbacks import BaseCallbackHandler


def redact(value, secrets=()):
    """Remove configured secret values, credential URLs and common token fields."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if re.fullmatch(
                r"(?i)(api[_-]?key|password|authorization|access_token|token|secret)",
                str(key),
            )
            else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if not isinstance(value, str):
        return value
    for secret in sorted(
        {s for s in secrets if isinstance(s, str) and s}, key=len, reverse=True
    ):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+(?::[^\s/@]*)?@", r"\1[REDACTED]@", value
    )
    # Consume the scheme AND credential before the generic one-token field rule.
    value = re.sub(
        r"(?i)([\"']?authorization[\"']?\s*[:=]\s*[\"']?)(?:Bearer|Basic)[ \t]+[^\s&\"'`,}]+",
        r"\1[REDACTED]",
        value,
    )
    return re.sub(
        r"(?i)([\"']?(?:api[_-]?key|password|authorization|access_token|token|secret)[\"']?\s*[:=]\s*[\"']?)([^\s&\"'`]+)",
        r"\1[REDACTED]",
        value,
    )


def config_secrets(config):
    """Collect configured secret values without logging the configuration."""
    return tuple(
        section[key]
        for section in config.values()
        if isinstance(section, dict)
        for key in ("api_key", "password", "access_token", "token", "secret")
        if isinstance(section.get(key), str) and section[key]
    )


def safe_metadata(value, config):
    """One serialization boundary for initial and enriched metadata."""
    return redact(value, config_secrets(config))


def model_metadata(config, args):
    model = config.get("llm", {})
    parsed = urlsplit(model.get("base_url", ""))
    return safe_metadata(
        {
            "requested": model.get("model_name"),
            "base_url": urlunsplit(
                (parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", "")
            ),
            "temperature": 0,
            "timeout": getattr(args, "model_timeout", None),
            "max_retries": getattr(args, "model_retries", None),
            "max_tokens": getattr(args, "max_tokens", None),
        },
        config,
    )


def safe_configuration(config, args):
    """Only explicitly approved connection/model fields enter the artifact."""
    fields = {
        "db_dw": ("host", "port", "database"),
        "db_meta": ("host", "port", "database"),
        "qdrant": ("host", "port"),
        "es": ("host", "port", "index_name"),
        "embedding": ("host", "port", "model"),
        "llm": ("model_name",),
    }
    result = {
        name: {field: config[name][field] for field in allowed if field in config[name]}
        for name, allowed in fields.items()
        if isinstance(config.get(name), dict)
    }
    result.setdefault("llm", {})["base_url"] = model_metadata(config, args)["base_url"]
    return safe_metadata(result, config)


class ResponseRecorder(BaseCallbackHandler):
    """Capture provider text before parsers; never infer a response model."""

    def __init__(self, records, scrub, sink=None):
        self.records, self.scrub, self.sink = records, scrub, sink
        self.prompts = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        from eval.live_evaluation import fingerprint

        self.prompts[str(run_id)] = fingerprint(
            [[message.model_dump() for message in batch] for batch in messages]
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                info = getattr(message, "response_metadata", {}) or {}
                response_info = response.llm_output or {}
                record = self.scrub(
                    {
                        "type": "model_response",
                        "run_id": str(run_id),
                        "observed_at": datetime.now(UTC).isoformat(),
                        "prompt_hash": self.prompts.pop(str(run_id), None),
                        "text": message.content
                        if message is not None
                        else generation.text,
                        "response_model": info.get("model_name")
                        or info.get("model")
                        or response_info.get("model_name")
                        or response_info.get("model"),
                        "finish_reason": info.get("finish_reason"),
                        "usage": getattr(message, "usage_metadata", None),
                    }
                )
                self.records.append(record)
                if self.sink is not None:
                    self.sink(record)

    def on_llm_error(self, error, *, run_id, **kwargs):
        record = {
            "type": "model_error",
            "run_id": str(run_id),
            "prompt_hash": self.prompts.pop(str(run_id), None),
            "error_type": type(error).__name__,
            "response_model": None,
        }
        self.records.append(record)
        if self.sink is not None:
            self.sink(record)


class LiveDW:
    """Use the deployed EXPLAIN adapter, preserving cursor labels before mapping."""

    def __init__(self, session):
        from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository

        self.repository = DWMySQLRepository(session)
        self.projection = None

    async def get_db_info(self):
        return await self.repository.get_db_info()

    async def validate(self, sql):
        return await self.repository.validate(sql)

    async def run(self, sql):
        from sqlalchemy import text

        result = await self.repository.session.execute(text(sql))
        self.projection = tuple(result.keys())
        if len({key.casefold() for key in self.projection}) != len(self.projection):
            raise ValueError("Non-unique result projection")
        return [dict(row) for row in result.mappings().fetchall()]


async def read_physical_schema(session, database):
    """Read actual names/types in server column order; ignore label casing."""
    from sqlalchemy import text

    result = await session.execute(
        text(
            "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, CHARACTER_SET_NAME "
            "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :database "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        ),
        {"database": database},
    )
    tables = {}
    for row in result.mappings().all():
        row = {key.lower(): value for key, value in row.items()}
        name = row["table_name"]
        tables.setdefault(name, {"name": name, "columns": []})["columns"].append(
            {
                "id": f"{name}.{row['column_name']}",
                "name": row["column_name"],
                "type": row["column_type"],
                "charset": row["character_set_name"],
            }
        )
    if not tables:
        raise ValueError("Empty physical schema")
    return list(tables.values())


async def read_business_context(session, physical):
    """Add business annotations to actual server fields; never trust stale types."""
    from sqlalchemy import select

    from app.models.column_info import ColumnInfoMySQL
    from app.models.metric_info import MetricInfoMySQL
    from app.models.table_info import TableInfoMySQL

    tables = (
        await session.scalars(select(TableInfoMySQL).order_by(TableInfoMySQL.id))
    ).all()
    columns = (
        await session.scalars(select(ColumnInfoMySQL).order_by(ColumnInfoMySQL.id))
    ).all()
    metrics = (
        await session.scalars(select(MetricInfoMySQL).order_by(MetricInfoMySQL.id))
    ).all()
    names = {table.id: table.name for table in tables}
    table_map = {table.name: table for table in tables}
    column_map = {
        (names.get(column.table_id), column.name): column for column in columns
    }
    result = []
    for physical_table in physical:
        name = physical_table["name"]
        table = table_map.get(name)
        enriched = {
            **physical_table,
            "role": getattr(table, "role", None),
            "description": getattr(table, "description", None),
            "columns": [],
        }
        for column in physical_table["columns"]:
            entity = column_map.get((name, column["name"]))
            enriched["columns"].append(
                {
                    **column,
                    **{
                        field: getattr(
                            entity,
                            field,
                            [] if field in {"alias", "examples"} else None,
                        )
                        for field in ("role", "description", "alias", "examples")
                    },
                }
            )
        result.append(enriched)
    return result, [
        {
            key: getattr(metric, key)
            for key in (
                "id",
                "name",
                "description",
                "relevant_columns",
                "alias",
            )
        }
        for metric in metrics
    ]


class LiveRuntime(SimpleNamespace):
    @asynccontextmanager
    async def gold(self):
        async with self.resources.dw_sessions() as session:
            yield LiveDW(session)

    @asynccontextmanager
    async def context(self, response_records):
        from langchain_openai import ChatOpenAI

        from app.repositories.es.value_es_repository import ValueESRepository
        from app.repositories.mysql.meta.meta_mysql_repository import (
            MetaMySQLRepository,
        )
        from app.repositories.qdrant.column_qdrant_repository import (
            ColumnQdrantRepository,
        )
        from app.repositories.qdrant.metric_qdrant_repository import (
            MetricQdrantRepository,
        )
        from eval.live_evaluation import persist

        model = self.config["llm"]
        recorder = ResponseRecorder(
            response_records,
            self.redact,
            sink=lambda record: persist(
                self.args.output_dir / "model-responses.jsonl",
                record,
                append=True,
            ),
        )
        llm = ChatOpenAI(
            model=model["model_name"],
            base_url=model["base_url"],
            api_key=model["api_key"],
            temperature=0,
            timeout=self.args.model_timeout,
            max_retries=self.args.model_retries,
            max_completion_tokens=self.args.max_tokens,
            callbacks=[recorder],
        )
        values = ValueESRepository(self.resources.es)
        values.index_name = self.config["es"]["index_name"]
        async with (
            self.resources.dw_sessions() as dw,
            self.resources.meta_sessions() as meta,
        ):
            yield {
                "llm": llm,
                "embedding_client": self.resources.embedding,
                "column_qdrant_repository": ColumnQdrantRepository(
                    self.resources.qdrant
                ),
                "metric_qdrant_repository": MetricQdrantRepository(
                    self.resources.qdrant
                ),
                "value_es_repository": values,
                "meta_mysql_repository": MetaMySQLRepository(meta),
                "dw_mysql_repository": LiveDW(dw),
                "sql_execution_policy": self.policy,
                "recall_budget": self.recall_budget,
                "max_sql_correction_attempts": self.correction_limit,
            }


@asynccontextmanager
async def readonly_session(factory):
    from sqlalchemy import text

    async with factory() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        yield session


@asynccontextmanager
async def connect_services(config, args):
    """Construct request-owned clients; no index writes, syncs or init probes."""
    from elasticsearch import AsyncElasticsearch
    from huggingface_hub import AsyncInferenceClient, InferenceClient
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.engine import URL
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async with AsyncExitStack() as stack:
        engines = {}
        for name in ("db_dw", "db_meta"):
            db = config[name]
            engine = create_async_engine(
                URL.create(
                    "mysql+asyncmy",
                    username=db["user"],
                    password=db["password"],
                    host=db["host"],
                    port=db["port"],
                    database=db["database"],
                    query={"charset": "utf8mb4"},
                ),
                connect_args={"connect_timeout": max(1, int(args.model_timeout))},
                pool_pre_ping=True,
            )
            stack.push_async_callback(engine.dispose)
            engines[name] = engine
        qdrant = AsyncQdrantClient(
            host=config["qdrant"]["host"],
            port=config["qdrant"]["port"],
            timeout=max(1, int(args.model_timeout)),
            check_compatibility=False,
        )
        stack.push_async_callback(qdrant.close)
        es = AsyncElasticsearch(
            hosts=[
                {
                    "host": config["es"]["host"],
                    "port": config["es"]["port"],
                    "scheme": "http",
                }
            ],
            request_timeout=args.model_timeout,
            max_retries=0,
            retry_on_timeout=False,
        )
        stack.push_async_callback(es.close)
        embedding = HuggingFaceEndpointEmbeddings(model=config["embedding"]["model"])
        endpoint = (
            f"http://{config['embedding']['host']}:{config['embedding']['port']}/embed"
        )
        embedding.client = InferenceClient(model=endpoint, timeout=args.model_timeout)
        stack.callback(embedding.client.close)
        embedding.async_client = AsyncInferenceClient(
            model=endpoint, timeout=args.model_timeout
        )
        stack.push_async_callback(embedding.async_client.close)
        dw_factory = async_sessionmaker(engines["db_dw"], expire_on_commit=False)
        meta_factory = async_sessionmaker(engines["db_meta"], expire_on_commit=False)
        yield SimpleNamespace(
            dw_engine=engines["db_dw"],
            meta_engine=engines["db_meta"],
            dw_sessions=lambda: readonly_session(dw_factory),
            meta_sessions=lambda: readonly_session(meta_factory),
            qdrant=qdrant,
            es=es,
            embedding=embedding,
        )


@asynccontextmanager
async def live_runtime(config, args, *, resources_factory=None):
    """Load deployment policy unchanged; connection overrides cannot widen it."""
    import os

    from loguru import logger

    from eval.live_evaluation import fingerprint

    # Required only by the application's default YAML loader. All actual model
    # requests use the explicitly injected user model below, never these sentinels.
    os.environ.setdefault("LLM_MODEL", "evaluation-injected-model")
    os.environ.setdefault("LLM_API_KEY", "evaluation-injected-model")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    from app.conf.app_config import app_config

    app_config.logging.file.enable = False
    app_config.logging.console.enable = False
    logger.remove()  # App nodes may log raw exception URLs; artifacts use type-only errors.
    from app.agent.sql_policy import SQLExecutionPolicy

    if resources_factory is None:
        resources_factory = connect_services
    deployment = app_config.sql_execution
    policy = SQLExecutionPolicy(
        config["db_dw"]["database"],
        deployment.max_rows,
        row_filters=deployment.row_filters,
        row_scope_column_types=deployment.row_scope_column_types,
    )
    async with resources_factory(config, args) as resources:
        async with resources.dw_sessions() as dw, resources.meta_sessions() as meta:
            physical = await read_physical_schema(dw, policy.database)
            tables, metrics = await read_business_context(meta, physical)
        safe_config = safe_configuration(config, args)
        secrets = config_secrets(config)
        root = Path(__file__).parents[1]
        runtime = LiveRuntime(
            physical_tables=physical,
            business_tables=tables,
            business_metrics=metrics,
            policy=policy,
            correction_limit=deployment.max_correction_attempts,
            recall_budget=app_config.recall,
            resources=resources,
            config=config,
            args=args,
            redact=lambda value: redact(value, secrets),
            metadata={
                "physical_schema_hash": fingerprint(physical),
                "business_metadata_hash": fingerprint([tables, metrics]),
                "safe_config_hash": fingerprint(safe_config),
                "safe_config": safe_config,
                "policy_hash": fingerprint(asdict(policy)),
                "prompt_hashes": {
                    path.name: fingerprint(path.read_text(encoding="utf8"))
                    for path in sorted((root / "prompts").glob("*.prompt"))
                },
                "recall_budget": asdict(app_config.recall),
                "correction_limits": {
                    "A": 0,
                    "B": 0,
                    "C": 0,
                    "D": deployment.max_correction_attempts,
                },
                "model": model_metadata(config, args),
            },
        )
        runtime.metadata = safe_metadata(runtime.metadata, config)
        yield runtime
