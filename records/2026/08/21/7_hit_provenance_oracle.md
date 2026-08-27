# 命中计数的度量有效性:"缓存里有" ≠ "引擎用了"

日期:2026-08-21(05:00–07:30)
分支:`multi_modal` @ `2620a17f` → `bba080f5` → `8f878ef7`(均未推送)
前置记录:`6_hybrid_suite_adaptation.md`(混合套件改造设计)

## 起因

混合(Qwen3.5-2B)全套件跑完:1 failed / 25 passed。那个 fail 是
`isolated_cases.py` 里 `server.wait()` 的残留调用(launcher 重构后
`MPServerHandle` 没有 `.wait`),纯手误。但排查它时在日志里看到一串:

```
Failed to reset prefix cache because some blocks (10) are not freed yet
```

——我为混合模型加的 `_reset_local_prefix_cache()` **一次都没成功过**。
顺着这条线查下去,发现的问题比那个手误严重得多。

## 三个实测事实

1. **vLLM 自己的前缀缓存确实会服务混合模型的重放。**
   无 connector 的纯引擎,同一 prompt 连跑三次:
   `num_cached_tokens` = 0 / 544 / 544(block=544)。所以"align 模式
   强制开 prefix caching → 重放被 GPU 缓存服务"不是理论担忧。
2. **LMCache 的命中计数器报的是"缓存里有什么",不是"引擎装载了什么"。**
   带 MP connector 的实验(`oracle_design_probe.json`):

   | 运行 | 清缓存 | vLLM local | vLLM external | LMCache 报命中 |
   |---|---|---|---|---|
   | 冷启 | – | 0 | 0 | 0 |
   | 重放 | 否 | **544** | **0** | **544** |
   | 重放 | 强制清 | 0 | **544** | 544 |

   第二行和第三行,**LMCache 侧计数器完全一样**。也就是说套件里所有
   命中数断言在混合路径上都可以在"retrieve 路径一次没跑"的情况下通过。
   connector 源码印证:`need_to_load = max(0, ret - num_computed_tokens)`
   ——`ret` 是服务端有多少,减掉 vLLM 本地已算的才是真要装载的。
3. **MP 路径上 `reset_prefix_cache()` 永远不可能成功。**
   vLLM 要求"除 null block 外零引用"才允许清;MP connector 会把最近
   一个请求的 block(实测 12405 里的 4 个)一直引用着,直到后续某个
   scheduler step 释放——而空闲时不再有 step。所以重试毫无意义
   (实测 120 次重试 / 60 秒,数字纹丝不动)。

## 修复

- `reset_vllm_prefix_cache()`:先走公开 API,失败则**直接清索引**
  (`cached_block_hash_to_block`)。清索引就够了——查命中只看索引。
- `VllmPrefillCounters`:patch `PrefillStats.set`,拿 vLLM **自己**的
  per-prefill 归属拆分(`num_local_cached_tokens` /
  `num_external_cached_tokens`)。这是地面真值。
- 每个被测步骤新增 `_check_hit_provenance`:单请求重放里 vLLM 本地
  缓存必须一个 token 都没服务;LMCache 报的命中必须真的被装载。
- MME parity:非平凡性 gate 从"报命中比例"改成**装载覆盖率**,并且
  分母从 `pass1_stored_tokens` 换成**可达命中量**
  `Σ chunk*((t-1)//chunk)`——store 计数会因去重失真(MME 一图两问,
  共享前缀存一次命中两次,覆盖率读出 2.0)。换分母后读 1.0
  (13056/13056 装载,本地 0,flip 0,parse 1.0)。
- `certify.py` 的 scope / known_not_covered 改为按 spec 生成,混合模型
  只 claim MP 路径 + 544 粒度,并拒绝在别的路径上录的 parity 报告。

## 校准出来的两个合法例外

第一版检查太严,全套件回归时抓出两个**合法**情况:

1. **末 token**:vLLM 永远重算 prompt 最后一个 token,所以整段命中时
   装载量比 connector 报的少 1(它自己的日志就写着
   `LMCache hit tokens: 304, need to load: 303`)。按步骤内请求数给
   1 token/请求的余量。
2. **PREEMPTED 请求 LMCache 故意不重新装载**(connector 里显式
   early return)。所以 preemption 场景会"报命中然后重算"——而重算
   正确性正是该场景要验的。改为显式 `harness.unloaded_hits_allowed()`
   opt out,同时保留"vLLM 本地缓存没插手"那一半检查。

## 影响面

- **已认证的 5 个模型不受影响**:它们跑在 prefix caching 关闭的进程内
  路径,local 恒为 0,报命中 == 装载量。用 qwen2.5-vl-3b 全套件回归
  验证:**29 passed**(带新检查)。证书无需重出。
- 混合模型的命中断言此前是"半真空"的,现在真的在跑 retrieve 路径。
  混合冒烟子集 6/6 通过(含跨图、部分共享、负对照)。

## 顺带修掉/发现的

- `isolated_cases.py`:`server.wait()` → `server.process.wait()`(崩在
  teardown,报告写不出来,pytest 只能报 "crashed before reporting")。
- `benchmark_parity.py`:测量段没有 `finally`,一次失败就把 MP server
  连同它 IPC 映射的 **80 GB GPU 显存**泄漏在机器上,要手工 kill。
- **强制清缓存不能碰被引用的 block**:第一版对所有 block
  `reset_hash()`,结果打崩了 MP connector 的会话——客户端进入
  `unhealthy → degraded → recovery → 重新注册 KV` 的循环(15 分钟里 4
  轮),请求零进展,套件挂死。原因是它按 hash 跟踪那几个"存储在飞"的
  block。改成只对 `ref_cnt == 0` 的 block 重置 hash 后,unhealthy 事件
  归零、不再挂死。证据留在
  `tmp/hang_client.log` / `tmp/hang_server.log`(见下)。

## 未完成

`test_t0_collision_pressure`(64 张不同图)在真装载生效后**失败**,
失败详情还没拿到(重测在跑:`hybrid_diag2.log`)。这是本次改造暴露的
第一个真信号——之前它是靠 vLLM 本地缓存"通过"的。混合模型证书在这个
问题查清前不出。下一步:
1. 拿到 collision_pressure 的断言细节(可能只是压力测试的公差要按
   pre-pad 共享前缀重算,也可能是真 bug);
2. `concurrent_batch`(批内含重复请求)在 MP 路径上是新覆盖,要确认
   "一个副本在存、它的重复副本在查"这条竞态没问题;
3. 全套件绿 → MME 全量 parity(冒烟已通)→ Qwen3.5-2B 证书。

## 经验

- **绿色不等于路径跑过。**每个"命中"断言都要能回答"谁服务的这些
  token"。这次的教训值得推广到别的计数器断言:计数器报的是它自己
  知道的事,不一定是系统实际做的事。
- 度量有效性问题只能靠**交叉源**发现:LMCache 计数器和 vLLM
  `PrefillStats` 互相校验,任一单独看都自洽。
- 临时脚本/实验用 `os._exit()` 跳过 teardown 时,server 之类的子进程
  会连着几十 GB 显存活下来;实验脚本也要写 `finally`。
