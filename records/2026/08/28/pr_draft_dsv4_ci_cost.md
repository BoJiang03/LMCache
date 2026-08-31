# PR draft: dsv4_flash_tp CI cost

Branch: `BoJiang03/LMCache` `dsv4_ci_cost_pr` @ `c3454869` (one commit, on top of `dev`)
Files: `.buildkite/k3_tests/multiprocess/{BK_WEB_SETUP.md,pipeline.yml,scripts/run-dsv4-flash-tp.sh}`

## Title

```
[CI][MP] Take dsv4_flash_tp off the default multiprocess run
```

## Body

```markdown
`dsv4_flash_tp` is the only 4-GPU step in the multiprocess suite, every other
step there takes one or two, and it runs 40+ minutes. So every `mp`/`full`
labelled PR build and every `dev` push held a whole 4-GPU node for that long
while the 1- and 2-GPU steps of other builds queued behind it.

### Gate the 4-GPU group

It now runs on three sources instead of by default:

- a **scheduled** build -- the nightly coverage
- a PR labelled **`dsv4`**, alongside the `mp`/`full` label that already gates
  this pipeline
- **`RUN_DSV4_TEST=true`** on a manual build

The schedule carries nightly coverage without depending on an env var staying
configured. The label lets a PR touching the hybrid-KV-group / slot-compression
path opt in with nothing but GitHub write access. The vLLM pin canary is a
scheduled build too, so the existing `VERIFY_AND_PIN_VLLM` clause keeps the
group out of it.

`BK_WEB_SETUP.md` documents all three, plus the Scheduled Build that has to
exist for the nightly path to fire.

### Drop the SM120 DeepGEMM build

vllm#52035 pinned vLLM back to deepseek-ai/DeepGEMM's `nv_dev` tip
(`8b1392b9`), which carries the SM120 kernels, and the vLLM revision CI pins
today already includes it. The guard never short-circuited either: it probed a
top-level `deep_gemm` while the wheel ships `vllm.third_party.deep_gemm`, so
SM120 nodes rebuilt it on every run.

### Persist the JIT caches across builds

The step compiles through four JIT frameworks on the way up, and the pod's
filesystem starts empty every run. Locally a fully cold start takes 788s and a
fully warm one 103s, split roughly as DeepGEMM 400s, FlashInfer 198s, Triton
and TileLang together about 87s.

One hostPath, `/data/jit_cache`, mounted at `/root/.cache/jit`, with each
framework pointed at a subdirectory of it:

| cache | env var |
|---|---|
| DeepGEMM | `DG_JIT_CACHE_DIR=/root/.cache/jit/deep_gemm` |
| Triton | `TRITON_CACHE_DIR=/root/.cache/jit/triton` |
| TileLang | `TILELANG_CACHE_DIR=/root/.cache/jit/tilelang` |
| FlashInfer | `FLASHINFER_WORKSPACE_BASE=/root/.cache/jit/flashinfer` (it appends `.cache/flashinfer/<version>/<arch>/cached_ops` itself) |

All four resolve from an explicit env var rather than from whatever `HOME` is
inside the image, and one mount instead of four means one directory to
provision on the agent nodes. DeepGEMM's dir is named rather than left to
default to `VLLM_CACHE_ROOT/deep_gemm`, which would mean persisting all of
`/root/.cache/vllm` for the sake of one subdirectory.

Sharing them across builds is safe because each keys on something that changes
when the toolchain does: DeepGEMM hashes the kernel source, the nvcc version
and the compile flags (which carry the `-gencode` arch); FlashInfer puts its
version and arch in the path; Triton's cache key hashes its own compiler and
backend sources along with the kernel, the backend options and the env;
TileLang namespaces its cache by its own version and the torch version. A new
pin, a new CUDA or a different arch lands on new keys rather than reusing an
old cubin.

One difference worth flagging in review: unlike DeepGEMM, which compiles into a
temp dir and renames into place, FlashInfer builds in place and serialises
concurrent builders on a per-module filelock. Concurrent builds on one node
wait rather than each recompiling, and a build killed mid-link is left for
ninja to redo on the next run.

The four caches total 35MB once filled.

### Per-phase timing, and a JIT cache report

The script prints a phase timing summary from an EXIT trap, so a failed or
timed-out run reports it too, and the step's wall clock can be attributed
instead of guessed.

It also reports, before and after the run, what each of the four JIT caches
actually resolved to:

```
=== JIT caches (before) ===
  HOME=/root
  DeepGEMM    entries=1350    exists=True  writable=True  /root/.cache/jit/deep_gemm
  Triton      entries=6403    exists=True  writable=True  /root/.cache/jit/triton
  TileLang    entries=180     exists=True  writable=True  /root/.cache/jit/tilelang
  FlashInfer  entries=607     exists=True  writable=True  /root/.cache/jit/flashinfer/.cache/flashinfer
```

The `after` pass adds `new_this_run=N` per cache. A mount that fails to take
effect is otherwise completely silent -- the step just recompiles and pays the
time again -- so this makes it visible: `exists=False` is a missing mount,
`writable=False` a read-only one, and `entries=0` on a *second* build means the
hostPath is not persisting. FlashInfer's path is read back from the library
rather than echoed from the env var, so a base that does not line up with the
mount shows as a mismatched path.

## Test

Ran the script end-to-end on 4x H200 outside k8s (its own `lmcache server` +
`vllm serve`, no `launch-processes.sh`), against pre-cached weights and with
both JIT caches warm:

```
=== Phase timing (total 173s) ===
  lmcache_server             10s
  prompt_build                0s
  vllm_startup              103s
  cold_run                   20s
  store_drain                20s
  apc_reset                   0s
  retrieve_run               20s
  verify                      0s

PASS: vLLM-run and LMCache-retrieve outputs are identical.
  outputs identical; LMCache served 4 retrieves.
```

Not vacuous: the cold run stored 8960 tokens (35 chunks of 256, so several
slot-compressed blocks per group), the retrieve run logged four
`Retrieved 8960 tokens` lines at 0.036-0.051s, one per TP rank, and
`retrieves_before=0 -> after=4`, with both 628-char outputs `cmp` clean.

Nothing was installed during the run, so the wheel's bundled DeepGEMM served
all of it -- the direct check on the SM120 deletion.

Two things a local run can't cover, both for a manual `RUN_DSV4_TEST=true`
build before merge: this box is SM90, and local vLLM is two minor versions
behind the CI pin.
```

## Note for reviewers / follow-up

The nightly path needs a Buildkite Scheduled Build that does not exist yet:
pipeline `multiprocess`, branch `dev`, `0 4 * * *`, no env vars, and it has to
be a *new* schedule rather than the pin canary. Needs someone with Buildkite
access, plus somewhere nightly failures get read.

Both hostPaths (`/data/deep_gemm_jit`, `/data/flashinfer_jit`) need to be
writable on the 4-GPU agent nodes; neither has been confirmed with whoever runs
the fleet.
