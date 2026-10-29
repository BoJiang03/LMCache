# l1_snapshot server module

`lmcache/v1/multiprocess/l1_snapshot.py` answers `l1_snapshot.take`: it lists
every L1 object and the eviction order, replies, and writes one L1 file per
request on its own writer thread. It is the server half of the L0/L1 state
snapshot; the trigger, file schema, costs and limits are in
[`docs/design/integration/vllm/state_snapshot.md`](../../integration/vllm/state_snapshot.md).

Load it with:

```bash
lmcache server ... \
  --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'
```
