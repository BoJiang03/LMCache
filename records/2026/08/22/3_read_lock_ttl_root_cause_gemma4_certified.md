# Gemma 4 根因:L1 读锁 TTL 在 lookup 与 transfer 之间到期

日期:2026-08-22(接 `2_`)

---

## 一、结论

困了两天的 Gemma 4-E4B 静默错答案,根因是 **`read_ttl_seconds`(默认 300 秒)的读锁在 lookup 与 transfer 之间到期**。

和 Gemma 4 的模型结构**毫无关系**。非均匀 `blocks_per_chunk = [1,1,1,1,1,2]`、`num_kv_shared_layers=18` —— 这两个此前唯一"还剩下"的嫌疑,全部无罪。这是第四个死掉的假设,也是最后一个。

判决性实验:只把 `--l1-read-ttl-seconds` 从 300 改成 86400,其余一切不变(同模型、同 chunk 32、同 6 组、同 2374 题)。

| | 修前 | 修后 |
|---|---|---|
| 失败读(`cannot be read`) | 7699 | **0** |
| `Retrieve failed` | 881 | **0** |
| flips vs pass1 | 1288 / 2374 | **0 / 2374** |
| flips vs baseline | — | **0 / 2374** |
| 合成套件 | 26/26 | 26/26 |
| 判决 | 静默错答案 | **SUPPORTED** |

`pass2_local_cached_tokens = 0`,所以 660800 个跳过的 token 确实是穿过 connector 命中的,不是 vLLM 自己的 prefix cache 伪装出来的。

---

## 二、机制

```
submit_prefetch_task  ──> l1_manager.reserve_read()      # lookup 时刻
                              └─ read_lock.lock()
                                   expiration = now + read_ttl_seconds

        ...  请求在 vLLM 的 waiting queue 里排队 ...        # 无人刷新 expiration

read_prefetched_results ──> l1_manager.unsafe_read()     # transfer 时刻
                              └─ if not read_lock.is_locked():
                                     return KEY_NOT_READABLE
```

`TTLLock::is_locked()`(`csrc/lmcache_native/ttl_lock.cpp`)是:

```cpp
return (counter > 0) && (current_time < expiration);
```

**只有 `lock()` 刷新 `expiration_ms_`。** `is_locked()` 不刷新,`unsafe_read()` 不刷新,中间任何一步都不刷新。所以只要 lookup→transfer 的间隔超过 TTL,entry 就在无人通知的情况下变成不可读 —— 而它**依然存在**,所以永远是 `KEY_NOT_READABLE`,永远不是 `KEY_NOT_EXIST`。这一点此前被当成"读锁归属问题",方向本身是对的,只是没想到"归属"是被时钟收走的。

vLLM 侧的放大器:请求进入 waiting queue 时就查 prefix,但要等 GPU block 腾出来才做 transfer。`benchmark_parity.py` 一次 `llm.chat()` 提交全部 2374 道题,于是队列深度直接把间隔推过 300 秒。

---

## 三、为什么此前所有形状分析都指错方向

**"clean through question 1056 然后 98% 坏" 不是阈值跳变,是滚动到期。** 新日志(完整服务端日志,4.8 MB,这次留下来了)显示失败从 02:45:25 一直散到 02:52:03,**6.6 分钟的参差爆发**,而不是 `2_` 里记的"19 秒内爆发"(那是 200 行过滤样本造成的假象)。每把锁按自己的 reserve 时刻各自到期,所以既没有干净的悬崖,也没有均匀的比例 —— 这个形状此前是我不敢给非均匀 block size 定罪的唯一理由,现在它反过来成了 TTL 的正面证据。

**首次失败落在 pass2 开始后 332 秒**,不是第 1 题。这是最该早点算的一个数,300 秒的旁边。

**`object_group_id=0` 占 100% 是选择效应,不是性质。** `lmcache_driven_transfer.py:1366`:

```python
for obj_group_id in range(num_object_groups):
    with read_prefetched_results(in_window_keys) as window_objs:
        if not window_objs or len(window_objs) != len(in_window_keys):
            logger.error("Some keys not found during retrieve!")
            retrieve_succeeded = False
            break          # <-- 第一个失败的 group 就跳出
```

六个 group 的锁在同一个 lookup 时刻 reserve、同时到期,所以 group 0 永远先失败,循环根本走不到 1–5。我在 `2_` 里把这条写成"读锁归属集中在一个 object group",是把一个 `break` 当成了数据特征。**教训和 `2_` 里那条同源但更进一步:不只要证明代码被执行过,还要证明你看到的分布不是控制流裁出来的。**

**Gemma 3 干净的真正原因是它跑得短。** 整个双 pass 服务端生命周期 **643 秒**,没有一把锁活得够久。此前我把它当成"多组滑窗在 chunk 16 下无罪"的证据 —— 结论没错,但理由是错的,它其实什么都没隔离出来,只是跑得快。同理,五张 chunk 16 的绿证书全是短跑。

**容量从头到尾无关**(37 GB / 280 GB 池),这条此前排除得是对的。

---

## 四、顺带修掉的第二个 bug:诊断说的是反话

`read_prefetched_results` 把 `unsafe_read` 的 `KEY_NOT_READABLE` 上报成 `reason="write_locked"`。但 `unsafe_read` **压根不看写锁** —— 它自己的 docstring(`l1_manager.py:320`)写明:"KEY_NOT_READABLE: The key is not readable (in this case, not read-locked)"。

这个标签是区分"读锁到期"和"撞上并发写"的**唯一信号**,而它一直指着一个不存在的并发写者。这是这个 bug 藏这么久的直接原因之一。

- 改成 `reason="read_lock_expired"`(仅 `l1_retrieve` 路径)。
- `store_controller` 那处**不动** —— 它走的是 `reserve_read`,那里 NOT_READABLE 确实等于 write-locked(`available_for_read()` 只查写锁)。同一个 enum 值在两条路径上语义不同,这本身值得记住。
- 更新了 metric 的 description,把三个 reason 各自的含义写清。

另加一行聚合日志。原来 7699 行 per-key `"exists but cannot be read"` 既不提锁也不提旋钮,读完整面墙依然无从下手:

```
N prefetched keys lost their read lock before transfer, so their load
returned nothing. The read lock is stamped at lookup and is not refreshed
on read, so this is what a lookup-to-transfer gap longer than
read_ttl_seconds (300s) looks like; raise it if the queue can
legitimately be that deep.
```

为读到 TTL 值,在 `L1Manager` 上加了 `read_ttl_seconds` 属性,而不是去碰 `_read_ttl_seconds` —— CLAUDE.md 禁止跨类访问私有成员。

---

## 五、修的是什么,没修的是什么

**修了:** `tests/e2e_mm/harness.py` 加 `MP_SERVER_L1_READ_TTL_S = 86400`,理由和它上面那条 reap timeout 完全一样 —— 批量 benchmark 不是在线服务,把超时变成无关项,而不是去调它。这不掩盖泄漏:`finish_read_prefetched` 依然释放每一把锁,只是不让时钟在排队中途响。

**没修,而且是故意的:** 生产侧的 TTL 行为。TTL 存在是有理由的 —— 它是防止"客户端 reserve 完就死"把内存永久钉住的安全阀。全局拉长或去掉,是拿一个失效模式换一个内存泄漏。真正的设计问题是 **retrieve 路径是否该在 transfer 时刻自己续锁,而不是继承 lookup 时刻的时间戳**。这有真实的权衡(续锁就削弱了安全阀),属于维护者的决定,不属于一次认证任务。**标记出来,不擅自实现。**

**一个重要的安抚:** 先前那个诚实上报的修复(`d43e817a`)已经把这个隐患从"静默错答案"降级成"安全失败"了。同样的 TTL 到期,现在在非 hybrid 上是 load error → 重算,在 hybrid 上是响亮打挂。这次带修复的重跑就是后者,和 `register_kv_caches` 那条多组警告预言的一字不差:

```
ValueError: too many values to unpack (expected 1)
  scheduler.py:2293  _update_requests_with_invalid_blocks
```

也就是说:**即使这个 TTL 根因没被找到,静默错答案也已经不会再发生了。** 找到根因带来的是"能用",不是"变安全"。

---

## 六、Gemma 4-E4B 证书

`SUPPORTED`。26/26 合成;MME 2374 题,pass1 与 baseline 逐字节相同(0 flips),pass2 与 pass1 逐字节相同(0 flips,预算 11.87);hit_ratio 0.957,coverage 1.0076,parse_ratio 门槛按 spec 的 0.85(它对 239 道名人/艺术品题固执地拒答,0.8993 是天花板,已实测 8/64/256 decode token 都救不回来);`pass2_local_cached_tokens = 0`。

这是**第一个非均匀 paged block size 的滑窗 hybrid**,也是此前卡住整条队列的那个模型。

---

## 七、证据归档

`/home/bo/lmcache_evidence/2026-08-22_gemma4_read_ttl/`(**仓库外**,不进 git):

- `parity_gemma-4-e4b.mp_server.log.gz` —— 7699 条失败读的完整服务端日志,已校验条数完整
- `certify_g4_refix.client.log.gz` —— 带诚实上报修复、打挂在 hybrid 上的那次
- `parity_gemma-3-4b.mp_server.CONTROL.log.gz` —— 零错误对照

上一次这个日志没留下来,只剩 200 行过滤样本,直接导致 `2_` 里把起病形状记错。**这次先归档再重跑** —— certify 的 parity 输出文件名由 `workdir / f"parity_{model_key}.json"` 决定,同一个模型重跑必然覆盖。

---

## 七点五、一个必须记下的既有 flake

回归扫描(`tests/v1/distributed/ + mp_observability/ + lmcache_native/`,`-p no:randomly`)带我的改动是 **2 failed / 1766 passed**,两条红的都是 `test_turboquant*_storage_manager_roundtrip`(`Expected 3 hits, got 0`)。单独跑全绿。

没有靠"看起来无关"下结论,而是把改动 stash 掉跑了同一条扫描:clean HEAD 也是 **2 failed / 1765 passed**(+1 正好是我新加的测试),但**红的是不同的 parametrization** —— clean 上是 `4bit_nc-0.9` + `3bit_nc-0.8`,带改动是 `4bit_nc-0.9` + `k3v4_nc-0.85`。

**同样的失败条数、每次不同的 parametrization 集合**,这就是既有 flake 的签名,也是判定它与本次改动无关的依据。看起来是 serde roundtrip 之间的跨测试状态泄漏,不是 serde 本身的 bug。已写入 memory,不在认证任务里追。

提交是在这个比对做完之后才做的,不是绕过两条红灯提交的。

---

## 七点七、待办第 1 条完成:两个场景终于能在 hybrid 上跑了

`capacity_eviction` / `preemption` 此前在每张 hybrid 证书上都写着 `known_not_covered`。原因不是"路径测不了",而是**两个场景都无条件 build `MMHarness`,而 `MMHarness` 正是唯一装不下 hybrid 的 harness**。可修,不是固有限制。

加了 `_deployment_harness` 路由器,按 spec 选路径,hybrid 走真实 MP server(容量也在那边,是 server 的 `l1_size_gb` 而不是 harness kwarg),并把 server 日志尾巴收进 metrics —— 下面两个发现全靠它才诊断得出来。

**路由暴露出三个独立缺陷:**

1. **容差单位错了。** 两个场景都用模块级 `CHUNK`(16)算 hit slack 和 bytes/token,而 harness 早就提供了 `harness.chunk`。在均匀模型上完全等价,在**每个 hybrid 上都是错的**(Gemma 4 是 32,Qwen3.8 是 784)。`capacity_eviction` 的 `bytes_per_token` 在 Gemma 4 上被放大了 2 倍,溢出断言因此毫无意义。**这不是判断题**:`ModelSpec.hybrid_block_tokens` 的 docstring 早就写明"tests read their chunk tolerance from `harness.chunk`",是这两个场景没遵守已经写下来的约定。

2. **`max_model_len` 和 block 数被耦合成 `PREEMPTION_GPU_BLOCKS * CHUNK`**,隐含假设 vLLM 的 block size 等于 LMCache 的 chunk —— 只对均匀 16-token-block 模型成立。拆成独立的 `PREEMPTION_MAX_MODEL_LEN`,数值保持 128*16 不变,让已认证模型看到完全相同的引擎。

3. **eviction 的容量是 MP 层从没答应过的数。** in-process 层把 0.05 GB 精确到字节(实测 resident 53673984 / cap 53687091),但 MP server 的宿主分配器按 **64 MB** 为单位扩张,0.05 GB 被静默向上取整成 64 MB 池(`Total allocated size: 61.25 MB, free 2.75 MB`)。eviction 于是正确地把用量约束在 64 MB,而场景拿它跟 51.2 MB 比,把一个正常工作的后端判成坏的。改成申请整数个单位。**这是修正错误的期望,不是放宽真实的约束** —— Gemma 4 现在落在 cap 的 0.992,是在 1.0 以内,而不是靠那 10% 容差。

**验证(两个 hybrid 几何完全不同,加一个非 hybrid 对照):**

| 模型 | chunk | 组数 | blocks | eviction | preemption |
|---|---|---|---|---|---|
| gemma-4-e4b | 32 | 6 | 512 | 0.992 cap / 3.8× 溢出 | 1 次,chunk 报 32 |
| gemma-3-4b | 16 | 7 | 1024 | 0.771 cap / 2.7× 溢出 | 1 次,chunk 报 16 |
| qwen2-vl-2b | 16 | 1 | 默认 | 通过,且 metrics 里**没有** server_log_tail | 通过 |

滑窗 hybrid 从 1 个可用场景变成 3 个。**chunk 报数按模型不同(32 vs 16)正是缺陷 1 已修的证据** —— 修之前两个都会报 16。非 hybrid 没有 server_log_tail,证明它仍走 in-process 分支、没起 server。

**故意什么都没给 recurrent-state hybrid**,所以没有任何模型在跑一个没为它验证过的场景。两个独立原因,都写进证书而不是留着不说:
- eviction:那边一个 object 是一整页 recurrent state(Qwen3.6-27B 约 205 MB),**比整个 cap 还大**,连一个 object 都存不下,会因为跟 eviction 无关的原因失败。
- preemption:block 池必须落在"能装一个 max-length 请求"和"装不下运行批次"之间,这个窗口由模型的 KV bytes/token 决定,**无法从 spec 推导**,没测过的 hybrid 会直接引擎起不来。

按这两个**性质**做 gate(而不是按"测过的模型清单"),新注册的模型默认就会被正确处理。

差点犯的错:我一度把 preemption 对所有 hybrid 打开,而三个 Qwen 没有测过的 `preemption_gpu_blocks` —— 那会让三个模型的套件直接挂在我刚修好的那个 ValueError 上。提交前自查抓住了。

`chunked_prefill` 仍然只在 in-process,但现在写明了理由而不是跟另外两个混为一谈:它把 batched-token 预算钉在远低于一个 prompt 的值上,而 recurrent-state 家族需要的恰恰相反(步长要宽到装下一整个 544–784 token 的 block)。**两个要求互相矛盾,是构造性的,不是管线缺失。** 滑窗 hybrid 的小 block 原则上可以,但没测,所以不开。

**`preemptions: 1`(两个模型都是)——这条如实记下。** 它过了 >0 的非空洞门槛,但不宽裕。而且这是 decode 调度的性质,不是池子余量的性质:Gemma 4 在 496 blocks 下产生同样的 1 次,所以挤池子只会把它推向 vLLM 拒绝启动的下界。写进 spec 注释,而不是绕着调参。场景的验证强度不受影响 —— 6 个输出和 6 次 replay 照样全查。

---

## 八、待办(按优先级)

1. ~~**`capacity_eviction` / `preemption` 路由 hybrid 到 MP harness**~~ —— **已完成,`b1836ce1`**,见上面第七点七节。滑窗 hybrid 已验证;recurrent-state 家族仍待两项测量(见下一条)。
2. **给 recurrent-state hybrid 测出两个数**:(a) 一个能装下 ~205 MB 状态页 object 的 eviction 容量 —— 注意同时要维持 >2× 溢出,这两个要求会互相拉扯;(b) 三个 Qwen 各自的 `preemption_gpu_blocks`,方法就是读 vLLM 拒绝启动时自己报的"estimated maximum model length"。做完这两项,三个 Qwen 也能拿到这两个场景。
3. **Qwen3-Omni** —— 第一个音频模型,八张证书全部把 audio 列为未覆盖;需要 AIR-Bench/MMAU 而非 MME,探针要重设计。
4. 搭车批次:DeepSeek-OCR、Mistral Small 3.1、MiniCPM-V 4.6、Molmo 2。
5. Llama 4 / Kimi-VL / Step-3。
6. Phi-4-multimodal —— 实质是 `extra_keys` 重构,见 [[keying-extra-keys-refactor-todo]]。
7. Whisper 明确排除。

**上报给维护者(不自己动)**:retrieve 路径是否该在 transfer 时刻续读锁。

**carry-over**:`achievable_hit_tokens` 分母、MP retrieve 上报延迟(0.3–20 s vs 服务端 3 ms)、heartbeat 连续失败策略、P5 bypass 护栏、MP race flakes。

**清理**:`backup/pre-artifact-rewrite` 是那 1.7 MB artifact blob 唯一的家,确认 fork 分支无误后删除,且永不推送。
