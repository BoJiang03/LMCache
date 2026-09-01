# Parked: the PR's last test file

`test_lazy_offload_policy.py` was the only test in `lazy_offloading_policy_pr`
(174 lines) until 2026-09-01, when it moved here so the PR carries no tests.

It does not belong in this branch's `tests/v1/`: it is written against the PR
interfaces (`create_offload_policy`, `lazy_offload_policy.base`,
`add(meta, block_hashes, epoch)`, `drain(DrainSignals)`), none of which exist
on this branch, whose policy package still has `types.py`, `admit(op)`, and
`observe_step`. The five live lazy-offload suites in `tests/v1/` test this
branch's code and keep running.

To put it back in the PR: `git checkout` this file into
`tests/v1/test_lazy_offload_policy.py` on `lazy_offloading_policy_pr`. It
passed 12/12 there at commit dcfc59cf.

Coverage it holds: policy selection (eviction-aware default, FIFO selectable,
unknown name rejected) and the FIFO policy's threshold, select count, epoch
aggregation, mixed-epoch rejection, eligibility, blocked requests, ordering,
hash snapshot, discards, and failed store.
