#!/usr/bin/env python
"""Stage A of cohort preparation: SWE-agent parquet -> flat JSONL.

The source is the published `nebius/SWE-agent-trajectories` dataset: real
SWE-agent runs against real GitHub issues, recorded as an alternating
system / user / assistant message list where every `user` turn after the
first is a tool observation (command output, file view, traceback).

This stage only reshapes the rows; every selection decision is in
`prepare_cohort.py`, which needs the serving tokenizer and therefore runs in
the vLLM environment. Reading parquet needs `pyarrow`, which the vLLM
environment does not carry, so the two stages are separate programs.

Usage:
    python extract_trajectories.py <parquet> <out.jsonl> [limit]
"""

import json
import sys

import pyarrow.parquet as pq

#: Roles as the dataset spells them, mapped to OpenAI chat roles.
_ROLES = {"system": "system", "user": "user", "ai": "assistant"}


def main() -> int:
    """Reshape `limit` rows of a trajectory parquet shard into JSONL.

    Returns:
        Process exit code.
    """
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, out = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    written = 0
    with open(out, "w") as fh:
        parquet = pq.ParquetFile(src)
        columns = ["instance_id", "model_name", "exit_status", "trajectory"]
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                messages = []
                for message in row["trajectory"]:
                    text = message["text"] or ""
                    if message["role"] == "system":
                        text = message.get("system_prompt") or text
                    messages.append(
                        {"role": _ROLES[message["role"]], "content": text}
                    )
                fh.write(
                    json.dumps(
                        {
                            "instance_id": row["instance_id"],
                            "model": row["model_name"],
                            "exit_status": row["exit_status"],
                            "messages": messages,
                        }
                    )
                    + "\n"
                )
                written += 1
                if written >= limit:
                    break
            if written >= limit:
                break
    print(f"[extract] wrote {written} trajectories to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
