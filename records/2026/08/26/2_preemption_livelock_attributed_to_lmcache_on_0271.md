# 0.27.1 抢占活锁判归 LMCache:裸 vLLM 批量对照坐实,服务端静默定位到引擎侧调度门

**日期**: 2026-08-26(当天第 2 篇,接 `1_`;`1_` 的 §结论 4 已按本篇结果改写)
**代码状态**: `multi_modal@48314ad6`,工作树干净(本篇不含仓库代码改动;
判别脚本与诊断 wrapper 归档在 `vllm_0271_mp/`,共 14 项)

> **本篇结论已被 `3_` 推翻(2026-08-26 同日)。**
> 活锁的病灶是 **vLLM 0.27.1 的 `defer_block_free`**(抢占释放的块延迟归还,
> 与抢占互锁),LMCache 只是把 `is_kv_consumer` 置真的那个连接器。
> 本篇用的"裸 vLLM 对照"不成立:裸 vLLM 没有 `kv_transfer_config`,
> 那个标志恒为假 —— 对照组把待判开关一起去掉了。判据、trace 与两组
> 单开关控制见 `3_`。下文保留原样以存错误推理的过程。

## 结论先写

1. ~~**0.27.1 上 `preemption` 场景的活锁属 LMCache,不是 vLLM 回归。**~~
   (已推翻,见篇首)

   | 实验(128 块小池,六请求**一批**提交) | 抢占数 | 结果 |
   |---|---|---|
   | 0.23.0 + LMCache MP | 2 | 通过 |
   | **0.27.1 + LMCache MP** | 每步全抢占,×13,544 | **活锁** |
   | 0.27.1 + LMCache MP,池加倍(256 块) | **0** | 批次干净跑完(场景按设计判 vacuous 失败) |
   | **0.27.1 裸 vLLM** | **2** | **6/6 输出,2.8 秒** |
   | **0.23.0 裸 vLLM** | **2** | **6/6 输出,25.8 秒** |

   裸 vLLM 在两个版本上都能抢占并恢复 → 版本本身没坏;只有挂上
   `LMCacheMPConnector` 的 0.27.1 卡死。

2. **不是"阈值挪了一点"**:池加大到不触发抢占,一切正常;**一旦真的发生
   抢占就再也回不来**。所以是抢占→恢复这条路上的硬故障,不是容量压力。

3. **定位:卡在引擎侧的调度门,不是服务端。**
   MP 服务端日志在 **07:27:33** 收完最后一笔 store(6 个请求的 336/352 token
   预填都进去了)之后**整段静默**,而引擎又空转了三分多钟。全线程栈转储显示
   主线程在 vLLM 的 `_run_engine` 生成循环里(`llm.chat` → `run_batch`),
   没有 LMCache 侧的 Python 死锁。
   **最可疑的环节**:连接器的异步 lookup 门控 —— 被抢占的请求重排时,
   若 `check_lookup_result` 永不返回,vLLM 会一直推迟这些请求,于是每步把
   六个全抢占一遍,永不前进。**未验证**,需要连接器调度侧插桩。
   已排除:`lazy_offload`(默认 False,我们的 `mp_kv_transfer_config` 没开,
   所以"块被 pending store 占住"这个先前猜测不成立)。

## 一、方法学:一条被推翻的推断,和补上的真对照

先前(`1_` 初稿)用的归属论据是"同一 engine 配置的裸 vLLM **baseline** 先跑完
了,所以 vLLM 自己没问题"。**这条不成立**:`baseline_runner.py` 明确
一个一个请求顺序跑(注释写着"Requests are executed one at a time to keep
batching identical to the test-side sequential execution"),六个单请求根本
不产生批次压力,也就从不触发抢占。它跑通什么也证明不了。

补的真对照 `preempt_plain_vllm.py`(已归档):同一 spec、同一
`num_gpu_blocks_override=128` / `max_model_len=2048` / `max_num_seqs=6`、
同样的六个请求、**一次 `llm.chat` 批量提交**、不挂任何连接器,读 vLLM 自己的
`vllm:num_preemptions` 计数。这才是"vLLM 单独能不能从抢占里恢复"的判据。

## 二、证据(`vllm_0271_mp/`,14 项)

| 文件 | 内容 |
|---|---|
| `ctl_0271_plain_preempt.log` / `ctl_0230_plain_preempt.log` | 裸 vLLM 批量对照(2 次抢占、6/6 输出) |
| `preempt_plain_vllm.py` | 上述对照脚本(可复跑) |
| `scen_0271_preempt.log` | 活锁全程(13,544 条同集合抢占日志 + SIGUSR1 全线程栈) |
| `scen_0271_preempt_256blk.log` | 池加倍后的干净跑 |
| `scen_0230_preempt.log` | 0.23.0 同配置通过 |
| `mp_server_livelock.log` | 活锁那趟的服务端日志:末笔 store 07:27:33 后静默 |
| `scenario_wrapper.py` | 独立跑单场景 + SIGUSR1 栈转储(补齐 pytest 注入的环境) |

## 三、教训

1. **"控制组跑通了"必须先问控制组跑的是不是同一件事。** baseline 顺序跑 vs
   引擎批量跑,压力模型完全不同;我拿它当 vLLM 无罪的证据,是把"没触发"
   当成了"触发了也没事"。判别实验必须复现**触发条件本身**(这里是抢占计数
   大于 0),否则是空转对照 —— 和 08-25 记录 3 里上游 CI"命中门空转"同一类
   错误,这次犯在自己头上。
2. **服务端日志的"静默"是有信息量的**:它把"卡在哪一侧"一刀切开,比任何
   引擎侧推理都快。排障时先看两侧时间线的最后一笔。
3. `faulthandler.register(SIGUSR1, chain=True)` 转储后链到默认处理会**杀掉
   进程**;诊断活锁必须 `chain=False`(第一次转储把进程带走了)。
4. 场景常量的"档位"要按它真正依赖的维度 key(见 `1_` §三):这次的
   `preemption` 池大小是靠环境变量临时覆盖才做出的判别实验,verify 树里加的
   `LMCACHE_MM_E2E_PREEMPT_BLOCKS` 是一次性调试钩子,**没有进 PR**。

## 四、诚实边界

1. 只在 **qwen2-vl-2b** 上测过;别的模型是否同样活锁未知。
2. **确切机制未验证**(异步 lookup 门控 vs 其他调度门),目前只有"服务端静默 +
   主线程在生成循环"这两条定位证据。
3. 0.23.0 与 0.27.1 的裸 vLLM 耗时差 25.8s vs 2.8s,原因没查(与本题无关,
   但别把它当性能结论)。
4. 所有 0.27.1 的跑都在 `vllm-mm`(无 cuda_ops → torch 回退 + 08-25 的 memcpy
   修复);native 路径未验证。

## 五、下一步

1. 连接器调度侧插桩(lookup 提交/回收、每步 scheduled tokens),把活锁的
   确切环节钉死。
2. 钉死后报上游:**vLLM 0.27.1 + LMCacheMPConnector,一旦发生抢占即不可恢复**
   (带最小复现:`preempt_plain_vllm.py` 作阴性对照 + 场景脚本作阳性)。
3. MP-only 改造(含 `test_deepstack` 的四个缓存外科手术辅助搬到服务端
   `/cache/objects`、`/cache/checksums`)—— 等指令,拦路项见 `1_` §四。
