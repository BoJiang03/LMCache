# 命中路径成段坏输出:定位到上游连接器的抢占块表,决定绕开不修

**日期**: 2026-08-27 12:13
**代码状态**: `multi_modal@c37a3d38`(工作树干净;本会话未改任何源码)
**推送**: `fork/multi_modal` 991a88c3 → **c37a3d38**(43 个提交,fast-forward,未建 PR)

## 一、结论先写

`5_` 记的第二类坏输出(成段重复循环垃圾,pass1 与 baseline 逐字节相同、只有
pass2 坏)**不是并发竞态,是 vLLM 抢占触发的上游缺陷**,而且不在本分支的改动里。

vLLM 的契约(`vllm/v1/core/sched/output.py:118`,`CachedRequestData`):

> For request ids not in `resumed_req_ids`, `new_block_ids` will be appended to
> the request's block IDs. For those in the set, `new_block_ids` will be **used
> as** the request's block IDs instead of appending.

LMCache 的连接器只实现了前一半:

```python
# lmcache/integration/vllm/lmcache_mp_connector.py:1140
if request_id not in cached_reqs.resumed_req_ids:
    request_tracker.append_block_ids(new_block_ids)
```

`resumed` 分支既不追加也**不替换**。于是被抢占恢复过的请求,tracker 里留着的是
已释放、很可能已易主的旧块 id;之后由该 tracker 构出的 STORE 就从别人的块里
gather,把别人的 KV 以自己的 key 提交进 L1。pass2 再读这个 key 就是别人的 KV,
输出成段坏。

## 二、证据(全部离线,零 GPU 成本)

数据源:`$SPOLD/parity_internvl3.5-2b*.answers.json` + 同名客户端日志 +
`*.mp_server.log`。脚本 `$SP/race/assoc{,2,3,4,5}.py`。

1. **坏输出与被抢占请求几乎同一集合**(基率 138/2374 = 5.8%):

| 跑法 | 抢占请求 | 坏输出 | 坏 ∩ 抢占 | 占坏的比例 |
|---|---|---|---|---|
| run1 (256) | 138 | 79 | 69 | **87.3%** |
| run2 (256) | 288 | 129 | 121 | **93.8%** |
| spread (256) | 858 | 16 | 14 | 87.5% |
| seqs=1 | **0** | 3(噪声底) | — | — |

   verdict 翻转同样有 77-89% 落在被抢占的请求上。最近距离中位数 = 0,即
   **坏的那个请求自己就是被抢占的那个**。

2. **全部 19 份 parity 日志的方向门与抢占数完全一致**:抢占数 0 的跑,方向门
   都判两向噪声(p 不显著);单向显著退化的跑,抢占数都 > 0(internvl
   62/134/253、phi4 20、gemma-3-4b 8)。唯一例外 molmo2-4b 有 28 次抢占但干净
   —— 抢占是必要不充分,还要那一刻确有块被别的请求接手。

3. **抢占全部发生在 pass1 的 store 窗口,pass2 零抢占**(run1:抢占
   07:31:24-07:32:14,stores 07:31:01-07:36:49,retrieves 才 07:35:14 起)。
   这解释了为什么 pass1 输出与 baseline 逐字节相同:vLLM 自己重算是对的,
   错的只是**存进去的那份**,损伤只能经 pass2 的读回显现。

## 三、排除掉的解释(别再查)

- **server 端多线程共用 staging 槽**:23 份 server 日志全是
  `AffinityThreadPool ... with 1 worker slots` + 单一 `affinity_key`,
  STORE/RETRIEVE 严格串行(affinity key = ZMQ identity),`transfer_kv_per_object_group`
  不存在跨线程交错。
- **L1 读写锁 TTL 过期**(write 600s / read 300s,`TTLLock.is_locked()` 会因过期
  返回 False):这条路径是**响的** —— `storage_manager` 会打
  "N prefetched keys lost their read lock" 并 fail-closed;这些跑里 0 行。
- **`stage_block_ids` 的 pageable `non_blocking=True`**:同流有序,非致因(是
  性能问题:pageable H2D 会先做一次 stream sync)。
- **partial prefetch / 容量驱逐 / 图片大小 / hit coverage 记账漂移**:`5_` 已排除。

## 四、归属:上游,不是本分支

- 该行在 `dev` 上一字不差(同为第 1140 行),三个版本变体
  `lmcache_mp_connector{,_0180,_0201}.py` 全一样。
- 本分支对该文件的全部改动只有一行且无关(`token_ids=tracker.get_token_ids()`);
  `handle_preemptions`、`flush_inflight_stores`、`lmcache_driven_transfer.py`、
  `worker_transfer.py` 均未改。
- 次生问题(也在上游):`handle_preemptions` 的 fence 对 MP 路径是空的 ——
  `LMCacheDrivenTransferContext.flush_inflight_stores` 是 `pass`,
  `torch_dev.synchronize()` 只同步 worker 自己进程,够不到 server 进程那条流。
- **为什么偏偏在这条分支上炸**:抢占要内存压力,多模态提示很长才触发得到
  (所有干净跑抢占数 0);且 MME 一张图两道题,被污染的 key 在 pass2 会按同一
  下标原样再读一次。
- **套件自己的抢占场景 T0.11 为什么没抓到**:它只有 6 个请求、专用小块池,
  释放的块大概率回到同一请求,旧块表因此“碰巧还是对的”;要 1024 深的批次才会
  被不相干的请求接手。这是套件的一处覆盖盲区,不是模型问题。

## 五、用户决定(2026-08-27)

> 不是我的代码的问题就不归我管,我们的策略是**避开**,而不是正面解决。
> 这个就算改了,也和 PR 的主线(多模态支持)无关。

以及随后:**实验全部停了**。

据此:
- **不改** `lmcache/` 里的连接器。
- 避开的实现方向(未落地,留给下轮):① parity 的
  `engine_kwargs`(`benchmark_parity.py:1129` 硬编码 `gpu_memory_utilization=0.6`)
  改为走 spec 已有的 `gpu_memory_utilization` 字段并抬高,使块池不会耗尽;
  ② 加一条「本次运行抢占数必须为 0」的前置校验,复用套件已有的
  `vllm_preemption_total()`(`isolated_cases.py` 的 T0.11 已在用),抢占 > 0 时
  报告判为**无效**(PROVISIONAL / 环境因素),而不是把模型判红;③ certify 加一条
  exclusion,按 `DEEPSTACK_NOT_COVERED` 的写法点明上游成因。全部落在
  `tests/e2e_mm/`,不扩 PR 范围。
- PR 里提一嘴即可,和 `defer_block_free` 同等处理。

## 六、被停掉的实验(无结果)

判别实验原本要打破「抢占 vs 并发」的混淆:并发不变,只调块池。

| GPU | 臂 | gpu_mem_util | 状态 |
|---|---|---|---|
| 5 | internvl 抑制抢占 | 0.88 | 12:12 按指示终止,未出报告 |
| 2 | gemma-3-4b 抑制抢占 | 0.88 | 12:12 终止(含孤儿 baseline 子进程) |
| 2 | internvl 放大抢占 | 0.45 | 12:05 主动换掉,未出报告 |

工具留在 `$SP/race/`:`probe_nopreempt.py`(只 patch 本进程的
`engine_kwargs`,不动仓库)、`launch_nopreempt.sh`、`iso_loop.sh`、
`assoc*.py`(上面所有关联分析)。

另外测过:gemma-3-4b 的 `test_isolated_scenario[mp_connector-gemma-3-4b]`
**连跑 3 轮全过**(253/244/271 秒),它几乎不触发抢占,所以**不是**这个缺陷的
廉价复现入口 —— 撤回我上午给的那个建议。

## 七、本会话其它进度

- **证书**:glm-4.6v-flash 由 schema 2 刷到 **SUPPORTED / schema 8 / stable**;
  phi4-mm 与 internvl3.5-2b 出诚实证书(**NOT_SUPPORTED** / schema 8 / stable,
  顺带修掉 internvl 那张 `stable=False`)。qwen3.8-27b 的 parity 在另一个会话的
  队列里跑着(GPU 3),qwen3.5-2b 仍是 schema 3(活锁,类已被 qwen3.6-27b 覆盖)。
- **Llama 4 放弃**(用户决定):官方 `meta-llama/Llama-4-Scout-17B-16E-Instruct`
  是 manual gate(当前 token 拿 `GatedRepoError`)且 bf16 217 GB 单卡装不下;
  开放的 `RedHatAI/...-FP8-dynamic`(115 GB)下到 79 G 时按指示终止并删除,
  `/raid` 回到 5.8T 空闲。
- **Step-3 出范围**:bf16 642 GB,最小的 `stepfun-ai/step3-fp8` 328 GB,至少
  TP≥3,而 TP>1 是每张证书的第一条 exclusion。
- **in-process 正式放弃**(用户决定):代码 8/26 已删净,只剩两处口径要改 ——
  `certify.IN_PROCESS_NOT_COVERED` 现在写成「套件只驱动 MP」像是暂时没测,
  该改为 out-of-scope;DeepStack 的 TD.1-TD.4 oracle 依赖读回 in-process
  `LocalCPUBackend`,永久放弃 in-process 等于**这一类永远没有活检测器**,
  除非给 MP server 加一个只读回存储对象的 debug API(建议放阶段 3 重构)。
- 本会话未提交任何代码;HEAD 从 9b2dbcd8 前进到 c37a3d38 是**另一个会话**的产出
  (`test(e2e_mm): make phi4-mm's suite runnable`),推送把它一并保存了。

## 八、GPU 归属提醒

12:1x 观察到另一会话(用旧 scratchpad `39f54daf`)在 GPU 2/3/5 上跑
`chain4.sh` 队列(`cert:phi4-mm`、`certonly:internvl3.5-2b`、chain9 的
qwen3.8-27b)。我只终止了自己的进程(`probe_nopreempt.py`、`iso_loop.sh` 的
pytest、Llama 4 下载),没碰它们。**开跑前查 venv + pgid,别再按 GPU 号假设空闲。**
