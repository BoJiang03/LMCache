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
| session length | 1 turn | 2 rounds | 12 steps |

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

# stage B: pick the fixed cohort (needs the serving tokenizer)
AGENTIC_RAW=raw.jsonl AGENTIC_COHORT_OUT=cohort.json \
AGENTIC_SESSIONS=48 AGENTIC_STEPS=12 SMOKE_MODEL=Qwen/Qwen3-8B \
python agentic/prepare_cohort.py
```

Stage B rejects any trajectory whose role pattern is irregular, whose
step-12 prompt falls outside the token window, or whose steps are not
token-prefix-stable, and it prints the cohort's SHA-256 so a rerun can be
shown to have replayed the same sessions.

## Running

```bash
source env.sh                       # SMOKE_GPU / SMOKE_TP / model / cache paths
AGENTIC_SWEEP_SIZES=8,16,24,32,48 AGENTIC_SWEEP_REP=0 \
python agentic/run_agentic_sweep.py

AGENTIC_SWEEP_REVERSE=1 AGENTIC_SWEEP_REP=1 \
python agentic/run_agentic_sweep.py   # second repetition, policy order reversed

python agentic/agentic_table.py results/
```

Each point boots its own MP server and engine, replays the cohort, and
writes one JSON with every per-request record, the vLLM counter deltas, the
L1 fill series, the lazy counter ledger, and the log's warnings and
tracebacks.

## Load model

Session `s` releases its step `k` at `t0 + (s + k * sessions) / RATE`, so:

- the aggregate step rate is `RATE` (default 2/s) at **every** cohort size,
  which keeps the offered load fixed while the working set is swept;
- a session's own gap between steps is `sessions / RATE` seconds -- the
  agent's tool-execution time, 4 s at 8 sessions and 24 s at 48;
- a step that could not be released on time (its predecessor was still
  running) is recorded in `lag`, so a saturated run is visible rather than
  silently reshaped.

The engine's answer is thrown away and the recorded action is appended
instead: both policies then replay byte-identical request streams.
