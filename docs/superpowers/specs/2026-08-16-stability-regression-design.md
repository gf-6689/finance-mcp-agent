# Stability Regression Fix Design

## Objective

Make the end-to-end financial analysis run complete without genuine `ERROR`
records while preserving the existing report workflow and public MCP tool
interfaces.

The change addresses four verified defects from execution
`20260816_165014_ef2b91d9`:

1. Monthly K-line requests inherit daily-only default fields and receive
   Baostock error `10004012`.
2. One News Agent run invokes `crawl_news` three times in parallel, causing
   three independent MCP subprocesses to load duplicate QLoRA models.
3. The execution logger leaves the `main` record in `started` state, so the
   summary reports it as failed even when the workflow succeeds.
4. MCP tool calls are not connected to `ExecutionLogger.log_tool_usage`, so
   `tools_used_count` remains zero.

## Safety and Recovery

- The pre-change commit is `e8b7e94e0469ac997ccbc4185e52570d37713249`.
- The remote recovery branch is
  `backup/pre-stability-fixes-20260816` and points to that exact commit.
- Implementation occurs on `fix/stability-regression`; `main` remains unchanged
  until targeted tests and a full regression pass.
- Model weights, datasets, reports, logs, and `.env` remain untracked.

## Architecture

### 1. Frequency-aware K-line fields

Add a small pure function in the Baostock data-source module that resolves
the effective field list from `frequency` and the caller-supplied `fields`.

- Daily data (`d`, `5`, `15`, `30`, `60`) retains the existing full default
  field list.
- Weekly and monthly data (`w`, `m`) use Baostock's supported aggregate fields:
  `date`, `code`, `open`, `high`, `low`, `close`, `volume`, `amount`, and
  `adjustflag`.
- Explicit weekly/monthly fields are validated before the network call. An
  unsupported field produces a local `ValueError` naming the invalid fields,
  rather than a remote Baostock `10004012` error.
- The public `get_historical_k_data` signature does not change.

### 2. Exactly one news crawl per News Agent run

The News Agent will split acquisition from synthesis:

1. Obtain the cached MCP tool list.
2. Locate `crawl_news` by name.
3. Invoke it exactly once with one combined query containing the company name,
   latest news, stock performance, earnings, and industry context.
4. Put the returned, already risk/sentiment-scored news text into the LLM input.
5. Remove `crawl_news` from the ReAct tool list used for subsequent synthesis.
6. Keep other MCP tools available so the LLM can request supporting market
   data when needed.

If the single crawl fails or returns an empty/error result, News Agent records
the failure and stops instead of silently asking the ReAct loop to retry and
load more model processes.

This guarantees only one MCP subprocess loads the risk and sentiment adapters
for a News Agent execution. It deliberately avoids a new inference service or
cross-process cache.

### 3. Tool-usage instrumentation

Add a LangChain callback handler dedicated to execution logging.

- It records tool start time and input on `on_tool_start`.
- It records output, duration, success, and error on `on_tool_end` or
  `on_tool_error`.
- Each Agent constructs the callback with its own `agent_name` and supplies it
  in the ReAct invocation config.
- The News Agent also records its one direct `crawl_news` invocation through
  the same logging helper.
- `tools_used_count` continues to be derived from JSONL records; no synthetic
  count is introduced.

Concurrent calls must use per-run identifiers rather than a single shared
start timestamp so durations cannot overwrite each other.

### 4. Main execution lifecycle

The main workflow will close its execution record explicitly:

- On success, call `log_agent_complete("main", ...)` before finalizing the
  overall execution.
- On an exception, call the same method with `success=False` before finalizing
  the failure.
- Record total main duration, report path on success, and exception text on
  failure.
- Ensure finalization happens once per run.

The readable summary should then show `main: success` with a non-zero duration.

## Data Flow

```text
user query
  -> News Agent
  -> one direct crawl_news invocation
  -> one MCP subprocess
  -> one risk model + one sentiment model load
  -> scored news text
  -> ReAct synthesis without crawl_news
  -> news analysis
  -> Summary Agent
  -> final report
```

The other three analysis Agents retain their current ReAct workflows. Their
tool calls receive logging callbacks but no behavioral changes.

## Error Handling

- Unsupported weekly/monthly fields fail locally with a precise validation
  message.
- Missing `crawl_news` is a News Agent failure with an explicit error.
- A failed direct crawl is recorded as a failed tool use and a failed News
  Agent execution; it is not retried implicitly.
- Callback logging failures must not break financial analysis. They are caught
  and emitted as logger warnings.
- Expected Baostock empty results for unpublished quarters remain warnings, not
  fatal errors.

## Tests

### Targeted unit tests

1. Daily default fields still include daily valuation and price fields.
2. Monthly and weekly defaults exclude `preclose`, `peTTM`, `pbMRQ`, `psTTM`,
   `pcfNcfTTM`, `turn`, `tradestatus`, `pctChg`, and `isST`.
3. Valid explicit monthly fields are accepted.
4. Invalid explicit monthly fields fail before calling Baostock.
5. News Agent invokes `crawl_news` once, removes it from ReAct tools, and embeds
   its result in the synthesis prompt.
6. Direct crawl failure is logged and does not invoke the ReAct agent.
7. Concurrent callback tool events produce one JSONL entry per completed call.
8. Main success and failure paths both complete the `main` execution record.
9. Generated execution summary reports a non-zero tool count and correct main
   status.

Each production change must be preceded by a failing test that demonstrates
the verified defect.

### Integration checks

- Compile all modified Python modules.
- Run all targeted tests.
- Run a minimal Baostock daily/monthly field test without repeated login.
- Run one complete `python -m src.main` regression.
- Require exit code zero, all Agents successful, complete report above 10 KB,
  `tools_used_count > 0`, `main` successful with non-zero duration, exactly one
  risk-model success and one sentiment-model success, and no genuine `ERROR`
  records.

## Acceptance Criteria

The repair is complete only when a fresh full regression proves all of the
following:

- `MAIN_EXIT=0`.
- Overall execution has `success: true` and `error: null`.
- Main and all five business Agents report success.
- The final report is complete and larger than 10 KB.
- Monthly K-line calls do not emit Baostock field error `10004012`.
- One News Agent execution invokes `crawl_news` exactly once.
- The run logs exactly one successful risk-model load and one successful
  sentiment-model load, with no model-load error.
- `tools_used_count` is greater than zero and matches persisted tool records.
- The fresh run log contains no genuine `ERROR` or traceback.
- The worktree is clean after committing, and the repair commit is pushed only
  after verification.
