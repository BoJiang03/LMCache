# MP 异步 retrieve 的完成延迟,以及混合模型上"装载失败不可恢复"

日期:2026-08-21(07:30–09:00)
分支:`multi_modal`(未推送)
前置记录:`7_hit_provenance_oracle.md`(命中来源交叉校验;本条是它暴露出来的
第一个真信号)

## 起因

`7_` 里留的未完成项:`test_t0_collision_pressure[qwen3.5-2b]`(64 张不同图)
在"真装载生效"后失败,要拿断言细节。拿到的不是断言——

```
ValueError: too many values to unpack (expected 1)
vllm/v1/core/sched/scheduler.py:2293
```

引擎自己崩了。三次重跑三次同样的崩法。

## 事实一:混合模型上 vLLM 无法恢复失败的 KV 装载

崩点是 `_update_requests_with_invalid_blocks`:

```python
# TODO (davidb): add support for hybrid memory allocator
(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
```

混合模型有多个 KV cache group(实测 4 个 kernel group,其中一个是
recurrent state),`get_block_ids` 返回每组一个列表,单元素解包直接抛。
上游自己标了 TODO,所以这是 vLLM 0.23.0 的既有缺口,与 LMCache 无关。

后果比"一个测试红"大:connector 的**降级模式在混合模型上等于杀引擎**。
它报 `invalid_block_ids` 是为了让 vLLM 重算(非混合模型上这条路是对的、
测过的),但混合模型上 vLLM 接不住。而且没有更优雅的替代——装载失败后
那段 KV 是垃圾,不重算就是静默错误输出,所以"报错块"是唯一正确动作。
证书里因此显式排除混合路径的降级恢复。

## 事实二:触发链是一次超时的心跳

`error_block_ids` 只有三处赋值,全部以 `not self.is_healthy` 为前提。
实测心跳的 ping 耗时(`mp_request_ping_timing_qwen3.5-2b.log`):

```
ping healthy=True in 3.15s / 3.55s / 1.28s / 5.87s
ping healthy=False in 10.02s   <-- 窗口=心跳间隔=10s
```

一次 ping 没抢到时间片就判"服务端死了",无重试、无连续失败计数。判死之后
`get_finished` 把所有在飞的 retrieve future 排干、其 block 全进
`error_block_ids` → vLLM 收到 `invalid_block_ids` → 事实一。

服务端日志证明它没死也没 reap 这个 worker(会话清理线程只有启停两行,
活动一直持续到进程被 teardown 终止)。

## 事实三:一次真 retrieve 的完成要 0.3–20 秒,而传输只有 3 毫秒

给 connector 的 lookup/retrieve 边界打时间戳
(`mp_retrieve_latency_trace_qwen3.5-2b.log`,相对秒):

```
113.84 retrieve_submit 3-81df3bc5
120.91 retrieve_done  ['3-81df3bc5']     7.07s
121.14 retrieve_submit 4-bdde0fe5
122.67 retrieve_done  ['4-bdde0fe5']     1.53s
122.89 retrieve_submit 5-83881692
142.93 retrieve_done  ['5-83881692']    20.04s  <-- 判死后被强制排干
```

同一时刻服务端侧(`mp_retrieve_latency_server_qwen3.5-2b.log`):

```
08:14:46,949 Prefetch request completed ... external_request_id=3-81df3bc5 (1.7 ms)
08:14:47,026 Retrieved 1088 tokens in 0.003 seconds
```

请求 3 的提交对齐到 08:14:46.9,服务端 08:14:47.03 就传完了,客户端到
08:14:54 才看见——**3 毫秒的活,7 秒才被观测到**。同一形状的问题在 `7_`
里已经露过一次头:被 connector 引用的 block "要等后续某个 scheduler step
才释放,而空闲时不再有 step"。两者都指向"MP 路径的异步完成只在引擎恰好
再走一步时才被注意到"。

## 对照实验:关掉强制清缓存

同一个用例,把 `_reset_local_prefix_cache` 打成 no-op(命中记账因此退回
真空,顺带关掉来源校验),只看延迟
(`mp_control_no_forced_clear_qwen3.5-2b.log`):

**128 个请求全部 ≤0.42s(首个 34.89s 是 warmup),测试通过。**

也就是说这条延迟只在"真的装载"时存在——`7_` 之前它一直被 vLLM 自己的
前缀缓存挡着,从没被跑到过。压力用例之前"绿",绿的是没跑过的路径。

宿主环境放大了它但造不出它:实测 `load average 480 / 160 核`(机器被别的
用户超订 3 倍)。服务端 3ms、客户端 7s 的差距不是 CPU 争抢能解释的。

## 处置

- **套件**:所有 MP 引擎统一改成 60s 心跳窗口 + 300s 服务端 reap 超时
  (`harness.MP_HEARTBEAT_INTERVAL_S` / `MP_WORKER_REAP_TIMEOUT_S`,
  常量成对给出,server 命令行和 extra_config 各取一个,防止漂移)。
  这样压力用例量的是缓存行为,而不是这条延迟。
- **证书**:`HYBRID_NOT_COVERED` 增加"装载失败恢复(降级模式)"一条,写清
  上游单 KV group 的假设和"混合上装载失败即致命"。
- **README**:新增段落说明为什么套件不能制造一次装载失败,以及心跳窗口
  为什么放宽。
- **没有**去改 LMCache 核心:心跳"一次超时即判死"和 retrieve 完成延迟都
  属于 MP 竞态/延迟家族(records/2026/08/20 里已有两个证据完整的 flake),
  值得单独一个 PR,不塞进认证工作里。混合 MP 路径在这条延迟修掉前不适合
  推荐上生产,这一点写进记录。

## 影响面

心跳窗口的改动落在**所有** MP 引擎上,包括非混合模型的 T3 场景(此前用
默认 10s 跑出的 5 张证书不需要重出:放宽窗口只会去掉"误判服务端死亡"这
一类事件,不会让任何断言变松——T3 里那句"rare KV-load-failure flake"的
注释此后应该基本不再触发,如果还触发就是真的装载失败)。

## 顺带

- `isolated_cases.py` 的 T3 场景里那句注释"a rare KV-load-failure flake
  aborts the engine mid-scenario"说的就是同一家族:非混合模型上它是罕见
  flake,混合模型上变成必崩,因为强制清缓存把每个请求都变成一次真装载,
  暴露面涨了约 30 倍。
- **不要用 faulthandler 周期性 dump 去采 CUDA 进程的栈**:
  `dump_traceback_later(4, repeat=True)` 把被测进程直接 segfault 了,
  apport 又写了 1.2 GB core(已删)。改成包一层 connector 方法打时间戳,
  一次就拿到了想要的相位数据。`ptrace_scope=1` 让 py-spy 不可用,所以这类
  "包一层"的办法是这台机器上的默认手段。
- HF hub 短暂不可达会让 baseline runner 直接 exit 1;重复实验一律加
  `HF_HUB_OFFLINE=1`。

## 未完成

1. MP retrieve 完成延迟的根因(future.query() 一直 False,还是
   `get_finished` 没被调用)——需要看 `lmcache_driven_transfer` 的事件/
   dispatcher 机制;这是 MP 家族的下一个正式课题。
2. 心跳应改成"连续 N 次失败才判死"(或 ping 走独立通道),让一次慢 ping
   不再等于服务端死亡。
3. Qwen3.5-2B 证书:全套件已绿(**26 passed**,`2a522b33`,含压力用例与
   批内重复用例,来源校验全程生效);MME 全量 parity(2374 题,MP 路径,
   280 GB L1)在跑,回来后出证书。
