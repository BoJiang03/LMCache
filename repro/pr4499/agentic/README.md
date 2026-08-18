# Agentic session replay

A third workload shape for the lazy-offload policy A/B, next to the
synthetic hot/cold document set and the QASPER paper-QA sweep: **real
SWE-agent sessions**, where a prompt grows one action and one tool
observation at a time and the session idles while the tool runs.

## Why this shape is different

| | hot/cold | QASPER multi-round | agentic replay |
| --- | --- | --- | --- |
| prompt across turns | fixed documents, re-sent | fixed paper, re-sent once | **grows every step** |
| reuse distance | interleaved requests | one 16 s revisit | every step, seconds apart |
| what is written | whole document | whole paper | one step's increment |
| session length | 1 turn | 2 rounds | 4--158 steps, as recorded |

The agentic case is the one where a *store decision is made about KV that
will be read again almost immediately* -- so it is the sharpest test of a
policy whose whole claim is "do not copy what the GPU can still serve".

## Data

`nebius/SWE-agent-trajectories`: published SWE-agent runs against real
GitHub issues. Each row is the recorded conversation -- system prompt,
issue text, then alternating actions and command output. No synthetic
filler, no generated history.

Two preparation stages, because reading parquet and tokenizing need
different environments:

```bash
# stage A: parquet -> flat JSONL (needs pyarrow)
python agentic/extract_trajectories.py <shard.parquet> raw.jsonl 1500

# stage B: pick the cohort (needs the serving tokenizer)
AGENTIC_RAW=raw.jsonl AGENTIC_COHORT_OUT=cohort.json \
SMOKE_MODEL=Qwen/Qwen3-8B python agentic/prepare_cohort.py
```

Stage B rejects any trajectory whose role pattern is irregular, whose final
prompt falls outside the token window, or whose steps are not
token-prefix-stable, and it prints the cohort's SHA-256 so a rerun can be
shown to have replayed the same sessions.

By default it truncates nothing: `AGENTIC_STEPS=0` replays every recorded
step, `AGENTIC_MIN_TOKENS=0` and `AGENTIC_MAX_TOKENS=40928` accept every
prompt the serving context can hold, and `AGENTIC_SESSIONS=0` takes every
usable trajectory in the file. Setting those to `12`, `8000`, `22000` and
`48` reproduces the capped cohort of AGENTIC_WORKLOAD.md sections 1--3 --
which is worth knowing about, because that cap is what put its median
request below the length at which a cache can pay for itself.

## Running

```bash
source env.sh                       # SMOKE_GPU / SMOKE_TP / model / cache paths
AGENTIC_SWEEP_SIZES=8,16,24,32,48 AGENTIC_SWEEP_REP=0 \
python agentic/run_agentic_sweep.py

AGENTIC_SWEEP_REVERSE=1 AGENTIC_SWEEP_REP=1 \
python agentic/run_agentic_sweep.py   # second repetition, policy order reversed

python agentic/agentic_table.py results/

# whole trajectories, pressure swept with the L1 budget instead
AGENTIC_COHORT=cohort_full.json AGENTIC_SLOT_STEPS=158 \
AGENTIC_MAX_MODEL_LEN=40960 AGENTIC_FULL_SLOTS=14 \
AGENTIC_FULL_BUDGETS=20,40,10 python agentic/run_full_replay.py
```

Each point boots its own MP server and engine, replays the cohort, and
writes one JSON with every per-request record, the vLLM counter deltas, the
L1 fill series, the lazy counter ledger, and the log's warnings and
tracebacks.

## Load model

The run is `sessions` concurrent **slots**. Each slot replays a queue of
whole trajectories back to back until it has issued `AGENTIC_SLOT_STEPS`
steps, and slot `s` releases its `j`-th step at
`t0 + (s + j * sessions) / RATE`, so:

- the aggregate step rate is `RATE` (default 2/s) at **every** size, which
  keeps the offered load fixed while pressure is swept;
- a slot's own gap between steps is `sessions / RATE` seconds -- the agent's
  tool-execution time;
- trajectories vary in length by a factor of forty, so giving each slot a
  queue rather than a single session is what keeps concurrency constant
  without shortening any session. When one ends the next starts on the
  following tick: a new session on the same slot, first step cold, the
  finished session's KV now dead weight in the cache. Only a slot's last
  trajectory can be left unfinished, identically under both policies;
- queues are packed longest trajectory first, so every slot can issue its
  full budget from a small pool;
- a step that could not be released on time (its predecessor was still
  running) is recorded in `lag`, so a saturated run is visible rather than
  silently reshaped.

The engine's answer is thrown away and the recorded action is appended
instead: both policies then replay byte-identical request streams.
