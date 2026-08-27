# eviction 打通三个 Qwen;preemption:排除 recurrent-state,并撤回 Gemma 4 的一个错误主张

日期:2026-08-22

接 [`3_`](3_read_lock_ttl_root_cause_gemma4_certified.md) 第八节 / [`4_`](4_session_closeout_state_and_escalations.md) 第四节的 todo 1:「给 recurrent-state hybrid 测出两个数」。

一个数根本不需要测,另一个不是「数」的问题。**而在查第二个的过程中,发现 `b1836ce1` 里我自己写下的一条「已验证」是错的 —— 那次验证跑的不是套件真正会跑的 prompt 形状。**

---

## 一、结论

| 模型 | capacity_eviction | preemption |
|---|---|---|
| qwen3.5-2b | ✅ 现有 64 MB cap,无需新数 | ❌ 排除(第三节:窗口是空的) |
| qwen3.6-27b | ✅ `eviction_capacity_gb=0.5` | ❌ 同上 |
| qwen3.8-27b | ✅ `eviction_capacity_gb=0.5` | ❌ 同上 |
| gemma-4-e4b | ✅(未变,复验 0.965 of cap) | ❌ **撤回** `preemption_gpu_blocks=512`(第四节) |
| gemma-3-4b | ✅(未变) | ✅ 1024,在正确 prompt 形状下复验 = 1 preemption |

三个 Qwen 的 isolated 场景 1 → 2 个;Gemma 4 从 3 个回落到 2 个 —— 后者是纠错,不是退化:之前那一个本来就没真的通过。

---

## 二、eviction:我上一轮的排除理由太粗

`4_` 和 memory 写的是「recurrent-state 一个 object 就是整张状态页(~205 MB),比整个 cap 还大」。**这句只对 27B 成立。**

- **qwen3.5-2b**:object 只有 **12 MB**,现有一个分配单元(64 MB)直接过 —— 5 个 resident key,60162048 bytes = cap 的 **0.897**,对 1.6 GB 意图流量溢出 **23.9×**。**不需要任何新数字。**
- **qwen3.6-27b / qwen3.8-27b**:784-token block 跨 64 层共 ~205 MB,拆成两个 object(状态页那个 ~154 MB)。8 个单元(512 MB)是第一个能装下整块、又给 0.80 驱逐水位留余量的尺寸。实测 6 个 object,513802240 bytes = cap 的 **0.957**,对 11.6–12.2 GB 溢出 **21.7×**。

真正的门槛是**单个 object 的大小,不是 hybrid family**。按 family 关掉这个场景,跟「按测过的模型清单 gate」是同一个错误、反方向:性质选得太粗,把本来能过的模型也关在门外。新字段 `ModelSpec.eviction_capacity_gb` 把门槛放回它该在的位置。

---

## 三、preemption 对 recurrent-state:窗口是空的

qwen3.5-2b,6 请求 × 3518 token,block 544,`max_model_len` 4352:

| GPU blocks | step budget | 连接器 | 结果 |
|---|---|---|---|
| 128 | 544 | MP | 0 preemption(concurrency 9.14×) |
| 48 | 550 | MP | 0 preemption |
| 32 | 550 | MP | 0 preemption |
| **28** | 550 | MP | **AssertionError** |
| 24 | 550 | MP | **AssertionError**(scheduler.py:462,RUNNING 分支) |
| 20 | 550 | MP | **AssertionError** |
| 16 | 544 | MP | **AssertionError**(scheduler.py:761,WAITING 分支) |
| 32 | 544 | 无 | 0 preemption |
| 32 | 550 | 无 | **1 preemption**,干净 |
| 24 / 20 / 16 | 550/544 | 无 | 0 preemption,干净 |

qwen3.6-27b 对照(prompt 4956,block 784):32 blocks / budget 790 / 无连接器 → **1 preemption**,干净。

**上半边空洞的机制**(从 vLLM 源码读出再实测确认):

```python
# vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split
num_new_tokens = num_new_tokens // block_size * block_size
```

调度器先排 RUNNING 再排 WAITING(scheduler.py:376 / 562)。只要有一个请求在 decode,它先拿走 1 个 token,剩 543 给某个 waiting 请求的 prefill —— `543 // 544 * 544 = 0`,该请求当步被 `continue` 跳过。`hybrid_engine_kwargs` 里 `max_num_batched_tokens = block_tokens` 是 align 模式能工作的**最小**值,同时正好是让并发不可能的值。**单变量证据:budget 544 → 0,budget 550(+PREEMPTION_N)→ 1,同一个 32-block 池子、同一个 plain vLLM。**

**下半边:见第五节的 crash。** 四个崩溃点(28/24/20/16)、三个空洞点(128/48/32),中间没有重叠。**所以这不是「差一个测量」。**

推论(比场景本身更重要):**三个已认证的 Qwen hybrid 从来没有真的让两个请求同时占用过 GPU。** 证书里的 "concurrent batches" 描述的是提交并发。已改成 `concurrent batch submission (vLLM executes it serially -- see known_not_covered)`,排除列表里写清机制与测量。

---

## 四、撤回:Gemma 4 的 `preemption_gpu_blocks=512`

`b1836ce1` 的 commit message 里我写「Verified on both registered hybrids … preemption 1」。用 pytest(套件真正的入口)重跑:

```
FAILED test_isolated_paths.py::test_isolated_scenario[preemption-gemma-4-e4b]
  no preemption occurred (counter delta 0) -- the scenario is vacuous
```

**原因:上一轮那次验证没有套上 conftest 的 hybrid prompt padding。** 对照实验一次说清 —— 关掉 padding 重跑,拿到的数字和 `b1836ce1` 的 artifact **逐字节相同**:

| | lookup_tokens | stored_delta | preemptions |
|---|---|---|---|
| 无 padding(= 上轮验证) | 1794 | 3232 | **1** |
| 有 padding(= 套件真实形状) | 60984 | 4352 | **0** |

引擎自己报的两个数在两次里完全一样:pool 买到 **2,314 tokens**,concurrency **1.13×**。场景要的压力是「六个 prompt 全被接纳,然后 decode 增长撑爆」:
- 无 padding:6×299 = 1794 ≤ 2314 < 6×411 = 2466 → 撑爆 → preempt。
- 有 padding:6×504 = 3024 > 2314 → 只接纳三个左右,其余在队列里每步被重新 lookup(这就是 60984 的来源)→ vLLM 从来不需要抢占。

于是把池子往上扫,让它装得下六个 prompt:

| blocks | pool tokens | preemptions |
|---|---|---|
| 512(已发布的值) | 2,314 | 0 |
| 768 | 3,472 | 0 |
| 992 | 4,484 | 0 |
| 1024 | 4,629 | 0 |
| 1152 | 5,208 | 0 |

**2.25× 的扫描里一个都没有。** 机制:这个模型的单请求占用会**饱和** —— sliding window 是 512 token,而 padded prompt 的跨度远超它,滑窗组会把窗口后面的 block 释放掉,于是再多 112 个 decode token 在那些组里不花钱。无 padding 时 prompt 只有 299 token(小于窗口),什么都不释放,批次才可能撑爆池子。

Gemma 3 的窗口是 1024,**宽于**它 ~700 token 的 padded prompt,所以不饱和,在正确形状下仍然 1 preemption。**这就是两个 sliding-window hybrid 分道的地方**,不是调参差异。

处理:Gemma 4 去掉 `preemption_gpu_blocks`,场景不再对它跑,spec 注释里留下上面整张表和机制;Gemma 3 保留 1024,并补上「在 padded 形状下复验」的记录。加大 `PREEMPTION_MAX_TOKENS` 能把压力找回来,但那个常量被所有已认证模型共用,属于另一件带自己回归成本的事 —— 记下来,不在这里做。

---

## 五、要上报的 bug:tight block pool + MP 连接器 → vLLM 断言崩溃

```
vllm/v1/core/block_pool.py:273 in cache_full_blocks
    assert blk.block_hash is None
AssertionError
```

路径 `allocate_slots -> coordinator.cache_blocks -> MambaManager.cache_blocks -> block_pool.cache_full_blocks`;28/24/20 走 RUNNING 分支(scheduler.py:462),16 走 WAITING 分支(scheduler.py:761)。

**归因干净:同样的 28 / 24 / 20 / 16 blocks,plain vLLM 全部跑完且输出正确,挂上 `LMCacheMPConnector` 全部崩。** 32 blocks 两边都不崩。

机制自洽:断言所在分支正是给「从 `num_external_computed_tokens` 拿到 external hit 的请求」缓存 block 的那段,断言含义是「一个已经带 hash 的 block 又被 cache 一次」。

复现物留在本目录:`plain_preempt_probe.py`(对照组)、`run_isolated.py`(场景运行器,`PROBE_BLOCKS=24 PROBE_BUDGET=550`)。**没有去修**:不在模型支持这条线上,触发配置也不是真实部署会用的;但它是硬崩溃,且正好长在这个 family 唯一可能测 preemption 的区间里,所以必须上报而不是绕过。

---

## 六、顺带挖出的四个既有缺陷

**1. eviction 的 false-hit 参照值是个竞态。** 原来 `steady = pass1[1].lookup_hits`,默认「请求 0 的 store 在请求 1 lookup 之前落地」。object 一大就不成立:qwen3.6-27b 量到 1568,**架构完全相同的** qwen3.8-27b 量到 **0**(154 MB 状态页还在飞),后面 28 个合法命中全被报成 false hit;重跑同一配置又变回 1568。**参照值由竞态决定,那它就不能是一次测量。** 改成绝对上界 `lookup_tokens - image_span_margin`,并把 `pass1_hits` 记进 metrics。

**2. 证书和实际跑的东西已经漂移过一次。** `b1836ce1` 给 sliding-window hybrid 打开两个场景,却没动静态的 `certify._HYBRID_SCHEDULING` —— Gemma 4 的证书**少报两个已通过的场景**、**多报一个从没跑过的 chunked prefill**。修法不是补那个 dict,而是让两边从同一个谓词派生:新模块 `tests/e2e_mm/isolated_routing.py`,parametrization 与 certificate 都读它,`isolated_cases` 在导入时校验自己的 `SCENARIOS` 键。

**3. preemption 的 under-storage 断言在它自己的目标场景里失效。** 它用 `missed = batch.lookup_tokens - batch.lookup_hits`。而排队的请求**每个调度步都会被重新 lookup 一次**,所以这两个计数器会把同一批 token 数很多遍:Gemma 3 上「missed 26730」对应的其实是六个 ~700 token 的 prompt,**6.4× 虚高**。一旦池子装不下整批(正是这个场景要造的压力),这条断言就不可满足;它过去能过,只因为 prompt 短到能一步全部接纳。改成用 replay pass 数**不重复的** prompt token,slack 给「共享前缀 + 一个不满的 chunk」;顺带 decode-relookup slack 也不需要了 —— 它当初就只是为了吸收同一份重复计数。

保留而不是删掉这条断言,是因为本场景 opt-out 了 unloaded-hit 规则(`harness.unloaded_hits_allowed`),store 计数器是**唯一**还能独立证明「LMCache 真被要求保存过」而不是 vLLM 自己的 prefix cache 把 replay 喂饱了的信号。

**4. `lookup_tokens` 是累积计数差,不是 prompt 长度。** 上面第 3 条的根:`StepResult.lookup_tokens = tokens_after - tokens_before`。单请求被重新调度时它也会翻倍 —— 768-block 那次跑里 t11-0 报了 1512 = 3 × 504,于是连 `again.lookup_hits >= again.lookup_tokens - 2*chunk` 这条 replay 断言都误红了。所有写成 `hits >= lookup_tokens - k*chunk` 的断言都隐含「每个请求只被 lookup 一次」。已发布的配置里 replay pass 是顺序单发的,不会触发;**但这是全套件的一个系统性脆弱点**,记在这里。

**5. 我自己差点留下一段死代码。** 为让 recurrent-state 的 padded prompt 装得下,我先加了 `PREEMPTION_MAX_MODEL_LEN_BLOCKS` 和一个 `max()`;等这个 family 整体排除后,这个分支没有任何注册模型会走。在一个「所有主张都必须被测量」的套件里留没被执行过的代码是负债 —— 删掉,知识留在常量注释里。

---

## 七、方法论

**这一轮最贵的一课:验证必须跑套件真正会构造的环境,而不是一个手写的近似。**

`2_` 的教训是「把一段代码当嫌疑犯前,先证明它被执行过」。`4_` 的补充是「排除理由要按性质 gate,不能按测过的清单」。这一轮的补充比前两条都更基础:**先证明你的验证跑的是真东西。** 上一轮我用手写 runner 验证了 Gemma 4 的 preemption,runner 少设了 conftest 的三个 pad 环境变量,于是我把一个「套件从来不会构造的引擎」上的绿灯,写成了 commit message 里的 "Verified"。整整一天里没人会发现 —— 直到有人用 pytest 跑它。

对策不是「以后更小心」,而是可操作的:**手写 runner 要么复制 conftest 的环境设置,要么最终验证必须走 pytest。** 本轮所有最终结论都用 `python -m pytest test_isolated_paths.py` 复核过。

**做对的:每一步都先做对照组。** 串行化、崩溃归因、padding 归因,三个结论都不是从场景失败里读出来的 —— 场景失败只说「0 preemption」「AssertionError」。是 plain vLLM 和 no-pad 的对照把它们变成单变量陈述:budget 544 vs 550 差 6 个 token,连接器有无差一个 kwarg,padding 有无差三个环境变量。

**两次被实测推翻的推理,都留着记账:**
1. 「112 个 decode token 跨不过 544 的 block 边界,所以永远不会 preempt」—— 错。plain vLLM 在 32 blocks 会 preempt,压力来自多请求 **prefill 阶段**的 block 争用,不是 decode 增长。
2. 「Gemma 4 的 padded 窗口只有 4% 宽」—— 错,我用错了 prompt 长度(736 而不是 504)。实际窗口是 [669, 817) blocks,22% 宽 —— 而 768 就在里面,却仍然 0 preemption。**正是这个「预测落在窗口内却不发生」的结果,才把结论从「调参」推到「饱和」。** 如果我按 4% 的估计只试 992/1024,会得到同样的 0,却讲一个错的机制。

---

## 八、代码改动

新增 `tests/e2e_mm/isolated_routing.py`;改 `specs.py`(`eviction_capacity_gb` 字段 + 两个 27B 的值;Gemma 4 撤回 pool 并留表;Gemma 3 补复验)、`isolated_cases.py`(per-model cap、race-proof false-hit 上界、`pass1_hits`、distinct-token under-storage 断言、SCENARIOS 一致性校验)、`test_isolated_paths.py`(路由外移)、`certify.py`(scheduling 派生、per-family 的 chunked-prefill 与 preemption 排除理由)、`tests/e2e_mm/README.md`(那段「两个场景尚未覆盖 hybrid」已经过时)。

`ruff check` / `ruff format` 干净。

---

## 九、下一步

1. **Qwen3-Omni** —— 第一个音频模型,所有证书都把 audio 列为未覆盖;需要 AIR-Bench/MMAU 而非 MME,探针要重设计。
2. 五个 hybrid 的证书都需要重跑才会带上新的主张与排除文字(Gemma 4 的旧证书 JSON 现在有两处不实)。
3. 上报第五节的 crash;以及 `4_` 第二节仍待上报的设计问题(retrieve 是否该在 transfer 时刻续读锁)。
4. 备选改进,已记未做:把 `PREEMPTION_MAX_TOKENS` 加大,让 preemption 场景对「单请求占用会饱和」的模型也有压力;代价是所有已认证模型都要重验这个场景。
