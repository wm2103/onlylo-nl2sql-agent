# 评测：离线可信链路与真实服务对照

## Safe live smoke CLI

以下命令均从仓库根目录执行。先按主 README 完成服务启动、配置与元数据导入。
`eval/data/retail-smoke-v1.json` 只包含问题、标签和基于初始化 DDL 审核的参考 SQL，
不包含预填结果或已验证的准确率；实际结果必须在目标数据库上重新执行。
真实评测会请求配置的模型并可能产生费用。该小样本只检查链路，不能代表问数准确率。

```sh
uv run --frozen --env-file .env python -m eval.live_evaluation \
  --dataset eval/data/retail-smoke-v1.json \
  --config conf/app_config.yaml \
  --output-dir reports/live-smoke-new-run \
  --groups A B C D --case-timeout 90 --model-timeout 30 \
  --model-retries 0 --max-tokens 2048 --init-timeout 30 --cleanup-timeout 2
```

**This command makes real service calls; it is not part of the offline tests.** Use a new output directory per run. The evaluator refuses to overwrite previous evidence. No live API/model run is implied by successful unit tests.

- `--config` loads user model/DB/retrieval connection values in memory, without printing credentials. It does **not** replace integration `sql_execution` or recall budgets. All groups keep the full request-evidence/pruning gates, the deployed SQL policy and authorization, and EXPLAIN. A/B are gated context ablations, not ungated model baselines. A supplies physical INFORMATION_SCHEMA names/types; B adds the Meta database's business annotations and metrics; C uses recalled context without repair; D allows the deployment's bounded repair. A/B/C have zero correction attempts. Only the generator receives replacement local inputs; graph state retained for policy is unchanged.
- Generated SQL is never replayed by the scorer. `run_calls` counts adapter dispatches, at most one per case/group; `run_completed=null` means completion was not established, including cancellation after dispatch. A timeout is **not** evidence that MySQL stopped an already dispatched query. Use read-only, least-privilege accounts and operational server limits; Python cancellation is cooperative, not a server-side kill.
- Each graph case/group and each independent gold observation has its own `--case-timeout` deadline covering read-only session acquisition (including `SET TRANSACTION READ ONLY`), observation and teardown. Teardown waits for at most the remaining case budget or `--cleanup-timeout`, whichever is shorter. A timed-out task is asked to cancel, with up to `--cleanup-timeout` additional drain time per cancellation. Runtime initialization has its own `--init-timeout`; runtime resource teardown uses `--cleanup-timeout`. These bound asynchronous waits, not synchronous work, interpreter shutdown waiting on cancellation-resistant tasks, or server-side execution.
- Gold uses a separate read-only session, actual server table metadata, shared SQL policy and EXPLAIN, independently counted in `type=gold` records. Gold SQL/audit never enters the model context. Reference results at the row cap are marked unavailable rather than accepting possibly truncated truth. Gold failures yield null result scores, not fabricated expected rows.
- Dataset contract: `{version,cases:[{id,query,label,audit,gold_sql,gold:{columns:[canonical names in projection order],ordered,abs_tolerance}}]}`. `gold_sql` and `gold` are required only for `answerable`; labels also include `needs_info` and `refuse`. Audit text is required, but the loader cannot establish its semantic correctness: review question/SQL, joins, money fields and empty-set semantics independently before running. This CLI does not reuse toy gold.
- Projection matching follows the declared positions using actual cursor labels, not model alias names or expected values. Width mismatch and case-insensitive duplicate labels are rejected, including empty results. Numeric absolute tolerance, one-to-one unordered matching, declared ordering, NULL, zero and empty-row distinctions use the trusted comparator after explicit canonicalization.
- `metadata.json` is written before config or service initialization and enriched with dataset/schema/business metadata/prompt/safe-config/policy hashes, requested model and credential-free URL, temperature, model timeout/retry/token and unchanged recall budgets. `records.jsonl` is flushed/fsynced per gold and case; `model-responses.jsonl` incrementally preserves provider response text before parsing (redacted), response model if actually supplied, usage and per-request prompt hash. Per-case records include full recall decision/evidence/requirements/missing terms, branch outcomes, SQL outcome and actual counts. Raw exception messages are not persisted; configured secrets, credential URLs and common secret fields are redacted. Existing application logging is suppressed in this dedicated CLI to avoid raw dependency error leakage.
- Aggregate only `type=case` for graph results and `type=gold` for reference results. Each has one observation per `record_id` within the run, flushed before teardown. Separate `type=lifecycle` records share that ID (`subject`, case `id`, and `group`, null for gold). They describe `started` and, when the context was acquired, `teardown`; do not count them as executions. A start with no observation is interrupted/unknown, not a zero-dispatch result. A teardown failure leaves a completed observation and its score intact but makes the run incomplete. Acquire failures/timeouts produce a case/gold observation with zero dispatches; initialization failure/timeout is separate.
- Completed records survive later initialization/run failure or interruption. Exit 2 signals initialization, reference, graph error or timeout; ordinary safety blocks/result mismatches remain recorded failures and need not make the process fail. Exit 0 is **not** an accuracy claim. This diagnostic smoke is not held-out evaluation; keep the original question labels even when conservative evidence gates block every group. Evaluate behavior over all cases and SQL equivalence over all answerable cases; do not drop blocked cases or reclassify gold failures as successes.

Offline verification (no live services):

```sh
PYTHONUTF8=1 LLM_MODEL=offline LLM_API_KEY=offline LLM_BASE_URL=http://127.0.0.1:1/v1 uv run --frozen python -m unittest discover -s tests -t . -q
uv run --frozen ruff check .
git diff --check
```

The offline public seams exercise real graph topology/policy/validation/run, independent gold, cursor projection handling, model-response recording/redaction, stalled acquisition/teardown and durable startup/partial-run artifacts. They do not prove live provider support, database execution cancellation, deployed column attestations, retrieval quality or model accuracy.

## Offline trusted harness


Run from the repository root using the offline environment:

```sh
uv run --frozen python -m eval.trusted_evaluation --mode offline
PYTHONUTF8=1 LLM_MODEL=offline LLM_API_KEY=offline LLM_BASE_URL=http://127.0.0.1:1/v1 uv run --frozen python -m unittest discover -s tests -t . -v
```

The CLI drives the actual LangGraph terminal/event path with deterministic in-process substitute adapters. It consumes returned terminal status, result rows, and recorded adapter calls; it does not re-execute SQL, invoke a live database, or call the model. The checked-in `toy-retail-v1` fixture is the source of business truth for audited ordinary labels and gold rows. Gold values are derived from its raw cents and region membership, never from the scripted adapter's chosen SQL or observed output. The `empty-west` question explicitly asks to group by region; an absent West order group therefore yields zero rows, distinct from an ungrouped scalar `SUM` over no matching orders (one NULL row) and from a fabricated zero-valued group. A narrow stdlib SQLite regression independently checks only these generic aggregate empty-set semantics against the fixture; this is not evidence of MySQL behavior or product SQL support. `ordinary-v1` covers answerable, needs-info, and refuse outcomes. Behavior uses every ordinary case as denominator; SQL-result equivalence uses answerable cases only. Gold comparison explicitly maps result columns, uses absolute numeric tolerance, honors ordered/unordered rows with one-to-one matching, and distinguishes empty rows from zero/NULL rows.

`fault-safety-v1` separately records recall-value failures, read-only and row-scope policy denials, correction exhaustion, syntax correction dropping a verified request filter, and an unsupported IN filter shape. Its report includes interception and false-block denominators, terminal classes, zero-execution violations, per-case terminal reasons, and validation/correction/run counts. The correction-drop case asserts one adapter validation total: the corrected query is rejected by policy before a second validation or execution.

`--mode live-preflight` deliberately returns blocked records with null metrics (exit status 2). Even configured credentials do not make this offline CLI perform live calls; live services and production model/DB behavior are not tested here. Never interpret offline substitute pass rates as production or model accuracy. A limitation surfaced when the ordinary query was made explicit: the current recall policy blocks `empty-west` as `required_filter_unverified` before SQL generation, so it does not answer the grouped sample. Offline behavior and SQL-result scores include that failure (3/5 and 2/3); the scripted adapter's preselected grouped SQL is not treated as graph behavior. This harness exposes rather than bypasses that limitation, does not widen product SQL policy, and does not claim a successful grouped-graph result. The ordinary delete example is also stopped as a missing/unverified required filter rather than the ideal policy-refusal class, and is marked accordingly as a behavior failure; harness correctness is tested separately.
