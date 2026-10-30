# SPDX-License-Identifier: Apache-2.0
"""Print what one L0 state snapshot holds.

Usage:
    python -m lmcache.tools.state_snapshot SNAPSHOT_DIR SNAPSHOT_ID [--run-id ID]
"""

# Standard
import argparse

# First Party
from lmcache.tools.state_snapshot.summary import format_summary, summarize


def main() -> None:
    """Parse the command line and print the snapshot's summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", help="lmcache.mp.snapshot_dir of the run")
    parser.add_argument("snapshot_id", type=int, help="snapshot id, from 0")
    parser.add_argument(
        "--run-id",
        default=None,
        help="scheduler run id; needed when the directory holds several runs",
    )
    args = parser.parse_args()
    print(format_summary(summarize(args.snapshot_dir, args.snapshot_id, args.run_id)))


if __name__ == "__main__":
    main()
