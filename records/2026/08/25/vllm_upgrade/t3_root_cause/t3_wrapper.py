"""Run one isolated scenario with the exact conftest environment applied."""
import sys
sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal/tests/e2e_mm")
from harness import configure_environment
configure_environment()
import isolated_cases
sys.exit(isolated_cases.main(sys.argv[1:]))
