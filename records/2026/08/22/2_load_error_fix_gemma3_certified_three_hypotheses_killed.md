# 载入失败如实上报、Gemma 3 认证,以及三个假设被自己的数据杀掉

日期:2026-08-22(00:10–02:25,接 `1_` 之后)
分支状态(全部**本地**,未推):

```
09bc14c0 (dev @ 08-18)
 ├── fix_mp_load_error  e18e55f2   载入失败如实上报(1 commit,9 files 之外的独立 PR 候选)
 └── multi_modal        ea5a84e1   d43e817a(同一修复)+ ea5a84e1(Gemma 3 认证)
```

本条的价值主要不在"做成了什么",在**三个假设怎么死的**。第二节是重点。

## 一、真正的 bug:MP 连接器把失败 retrieve 的 block id 扔了

`LMCacheMPWorkerAdapter.get_finished` 的排空循环写的是

```python
for request_id, (r_future, _) in self.retrieve_futures.items():
```

下划线就是 `op.flat_block_ids`。future 解析为 False 时只 `logger.error`,然后请求照样进 `finished_retrieves` —— vLLM 看到一次干净的完成加载,保留已计入的 `num_computed_tokens`,不重算,模型去读那些块里的残留。

**证据链是闭合的,四段都在代码里:**

1. `unsafe_read` 返回 `KEY_NOT_READABLE` → `read_prefetched_results` yield `None`;
2. `retrieve()` 打 `Some keys not found during retrieve!`,`retrieve_succeeded = False`,并且 `MP_RETRIEVE_END` 上报 `num_tokens=0` —— **服务端自己是知道的**;
3. `RetrieveResult = bool`,`DeviceMessagingFuture` 把 `tuple[bytes, T]` 解成 `T` —— 所以 `r_result` 就是 `retrieve_succeeded`;
4. 最后一跳把它丢了。

**最有说服力的是不对称:同一个方法的"服务不健康"分支一直写的是对的**(`self.error_block_ids.update(r_block_ids)`),健康但失败这条漏了。意图本来就在。进程内连接器 `vllm_v1_adapter.py` 从来没受影响 —— 它从返回的 token mask 反算漏掉的块再上报。**所以这个洞只在 MP 路径,而 MP 路径正是认证套件和生产用的那条。**

修法:抽出 `_collect_finished_retrieves()`,让 `get_finished` 和
`get_finished_with_lazy_offload` 两份复制品不能再各自漂移(文件里本来就挂着
`TODO: There are some duplicated codes with get_finished`),在里面补上 flag。

**关键验证不是"测试过了",是"撤掉修复测试会挂"**:把生产改动单独 `git checkout` 掉,三个正向测试全挂(`assert set() == {3, 4, 9}`、`{7}`、warning 断言),两个负向对照两边都过。没有这一步,测试可能只是在断言自己。

**一个刻意接受的代价。** vLLM 的恢复路径 `_update_requests_with_invalid_blocks` 里有
`(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)` —— 只解包一个 KV 组,
上游自己挂着 `TODO (davidb): add support for hybrid memory allocator`,**0.23.0 和 0.27.1 都还是这样**。
所以混合模型如实上报会让引擎报错退出而不是重算。仍然选如实上报(崩溃是响的,错答案是哑的),
并在注册时加一条警告把这件事讲在故障之前。计数用**去重后的 `engine_group_id`**,不是
`EngineGroupInfo` 的个数 —— 后者可能因物理转移身份把一个 engine group 拆开,而 vLLM 解包的是它自己的组。
Gemma 3 实跑时这条警告确切打了一次(7 组),没误报。

## 二、三个假设,全部被数据杀掉(本条核心)

### 假设 1:"小 chunk 是触发条件" —— 被证书本身杀掉

见 `1_` 第五节已作废的那条。8 张证书里 **5 张的 chunk 是 16**,比 Gemma 4 的 32 更小,
全是 2374 题全量 MME 全绿。

**错在哪:我是从 `hybrid_block_tokens` 那一列反推的**,非 hybrid 的模型这一列是 0、
chunk 走 `LMCACHE_TEST_CHUNK_SIZE`。**我把"字段为空"读成了"不在样本里"。**
后果不只是结论错 —— 它让我把"先修存储层再加模型"排到第 0 位,理由整个是虚的。

### 假设 2:"多 object group + 高对象数" —— 被 Gemma 3 杀掉

Gemma 3 4B:7 个 KV 组、滑窗、2374 题、136 KB/token,压力全有。
**结果 1 个翻转(预算 11.87)、零失败读。** 假设不成立。

### 假设 3:"滑窗窗口算术错位" —— 被两次运行自己的数字杀掉

这是最值得记的一个,因为它**看起来**已经快成立了:lookup 保留
`[H−w, H)`(`unfold`),retrieve 读 `[num_chunks−w, num_chunks)`,两边的 H 来源不同,
像是天然会错位。

然后用两次运行自己的 token 数去算 `skip` 到底有没有非零过:

| | chunk | 每请求 chunk 数 | 滑窗 | `skip = max(0, chunks − window)` |
|---|---|---|---|---|
| Gemma 3 | 16 | 284.5/16 = **17.8** | 1024/16 = **64** | **0** |
| Gemma 4 | 32 | 279.3/32 = **8.7** | 512/32 = **16** | **0** |

**两边恒为 0。**MME 的提示词(~280 token)太短,填不满任何一个模型的滑窗。
所以那条"按窗口只锁后缀"的截断路径,**在两次运行里都一次没执行过** ——
Gemma 4 挂的原因不可能在那里,Gemma 3 的绿也不能用来给那条路径背书。

**方法论教训:在把一段代码当嫌疑犯之前,先证明它被执行过。**
我差点写出一份基于从未运行的代码的根因分析。能救回来的唯一原因是去算了每请求 chunk 数 ——
一个两行的除法,比读三百行代码有用。

### 附带排除的一条,查了但**不**报成 bug

`retrieve()` 判 `window < 0`,`fold`/`_fold_python` 判 `window <= 0`,两边对
`window == 0` 的解释相反(一个当空窗、一个当全注意力)。但
`AttnWindowDesc.__post_init__` 显式 `raise` 掉 0(`must be -1 or >= 1`),
**这条路不可达**。查清楚再决定不报,和没查过是两件事。

## 三、Gemma 4 剩下的独特之处(从服务端日志里读的硬数据)

```
Gemma 3: 7 个 kernel group,全部 tokens_per_block=16, bs=16, nh=4, hs=256
Gemma 4: group 0-4  tokens_per_block=32, bs=32, nh=2, hs=256, sw=512
         group 5    tokens_per_block=16, bs=16, nh=2, hs=512, sw=-1
```

chunk 32 下 `calculate_num_blocks`:groups 0-4 = `32*32//32//32 = 1`,
group 5 = `32*16//16//16 = **2**`。所以

- **Gemma 4 `blocks_per_chunk = [1,1,1,1,1,2]`,Gemma 3 全 1。**
- group 5 的 `hs=512` 是别人两倍 —— 正是 `EngineGroupInfo` 文档说的
  "一个 engine group 因物理转移身份被拆开"。
- 外加 `num_kv_shared_layers=18`:42 层里只有 24 层有自己的 KV。Gemma 3 一层不共享。

**Gemma 4 是唯一 per-group 块几何不一致的模型**,而这条每次转移都走,与窗口无关。

**但我不宣布它是根因,因为时间形状不对。** 每次转移都走的算术错误会从第 1 题起坏;
实测是**前 1056 题干净、之后约 98% 坏**,失败读在 200 行采样里挤在 **19 秒**内爆发。
那是"越过阈值后状态变坏",不是"算错了"。容量也排除:56 KB/token × 279 token × 2374 题
≈ **37 GB**,而 L1 给了 280 GB。

一条待查线索:客户端日志里有
`Failed to reset prefix cache because some blocks (138) are not freed yet` ——
"引用没释放"和 `KEY_NOT_READABLE`(键在、读锁没了)是同一族账目问题,但还不是证据。

失败读的错误类型分布是**纯的**:180/180 全是 `cannot be read`,零 `KEY_NOT_EXIST`,
全部 `object_group_id=0`。而 `unsafe_read` 只在
`not entry.read_lock.is_locked()` 时返回它 —— 所以是**读锁归属**,不是驱逐、不是写锁冲突。

## 四、Gemma 3 4B:SUPPORTED(第一个滑窗混合模型)

| | 值 | 门限 |
|---|---|---|
| flips pass1 vs baseline | **0** / 2374 | — |
| flips pass2 vs pass1 | **1** / 2374 | 11.87 |
| 分数 | 1715.68 / 1715.68 / 1714.93 | Δ ≤ 10 |
| hit ratio / coverage | 0.965 / 1.0056 | 0.8 / 0.95 |
| parse ratio | 0.992 | 0.9 |
| 套件 | 25 passed / 0 fail / 0 skip | — |
| 失败读 / 上报坏块 | **0 / 0** | — |

pass1 与 baseline **逐字节相同**。`pass2_local_cached_tokens = 0` ——
667520 个跳过的 token **没有一个**来自 vLLM 自己的 prefix cache,全部真的过了连接器。
**这个字段是这类跑分唯一的防伪**:开着 prefix caching,一次纯 GPU 命中的重放也会报出满额 LMCache 命中。

几何 / 配置(实测,写进 spec 注释):34 层 = 29 滑窗(1024)+ 5 全注意力 → 7 组 →
必须 hybrid manager → 仅 MP 路径;全组 block 16 → chunk 16;136 KB/token
(65536 × 34 / 16),是 Gemma 4 的 2.4 倍。**无需 `hf_overrides`,无需门限覆盖**(默认就过)。

`modalities` 只写 image 不是我缩范围:vLLM 的
`Gemma3ProcessingInfo.get_supported_mm_limits()` 返回 `{"image": None}`,没有 video 条目,
所以那 6 个 deselected 是视频探针,模型吃不下视频。**这条去查了才写,没有猜。**

两次跑套件的耗时差:1002 s → **431 s**,同机同测试,差别只是权重和编译缓存热了。
报进度时要注意这个,否则会把"热缓存"当成"变快了"。

## 五、操作性发现

**1. `pgrep` 等待循环自匹配 —— 昨天的归因是错的。**

```
until ! pgrep -f "certify.py gemma-3-4b" > /dev/null; do sleep 30; done
```

`pgrep -f` 匹配完整命令行,**等待者自己的命令行就含着这个字面串**,所以目标结束后
它仍"找到一个匹配"(自己),永不退出。实测:`pgrep -af` 三行全是等待者和探测 shell,
目标早已不存在。

`1_` 第七节把 8 个僵尸循环归给"Bash 工具 2 分钟超时截断",**至少是不完整的** ——
真机制是自匹配。症状是**该来的完成通知没来**,而我当时把它当成了正常。
已存 memory `pgrep-wait-loop-self-match`,规避:模式写 `"[c]ertify.py ..."`、
或先抓 PID 用 `kill -0`、或检查结果文件。最根本一条:**harness 已经在跟踪的东西不要手写轮询器**,
这两个循环本来就是多余的。另外**要杀父 shell,杀 `sleep` 只会进下一轮**。

**2. 长实验中途只有 tqdm 能当进度信号。**
`benchmark_parity.py` 的 `print(f"[parity] ...")` 在重定向到文件时走块缓冲,进程结束才落盘;
tqdm 写 stderr 不缓冲。我一开始按 `[parity]` 缺失判断"还没进 parity",**那个推断是错的**。

**3. artifact 归档纪律生效了。** Gemma 3 的证书/几何/parity/junit 全部 `cp` 进
`records/2026/08/22/`(被 `.git/info/exclude:19:records/` 排除),提交只含 `specs.py` 一个文件,
`git ls-tree -r HEAD | grep -cE "certificate_|parity_|suite_.*xml|records/"` = **0**。
`1_` 第一节那次 1.7 MB 历史重写没有重演。

**4. 两个套件场景根本跑不了混合模型**(已存 memory `hybrid-models-are-mp-path-only`)。
只有 `LMCacheMPConnector` 继承 `SupportsHMA`,两个场景都用 `MMHarness`:
- `capacity_eviction`:vLLM 关掉 HMA 后在自己内部崩 —— `KeyError: 'language_model.model.layers.24.self_attn.attn'`;
- `preemption`:`max_model_len = PREEMPTION_GPU_BLOCKS * CHUNK` 硬编固定 chunk 16,饿死 56 KB/token 的模型。

这就是它们出现在 `known_not_covered` 里的**具体原因**,也是我为验假设跑的两个 Gemma 4 探针
**一个数据都没产出**的原因 —— 它们死在套件限制上,不是死在 bug 上。当时没有把这个
误读成"假设被否",这点是对的。

## 待办(顺序即优先级)

1. **在跑**:Gemma 4 parity 重跑(带修复)。要两样东西 —— 完整服务端日志(起病点上下文、
   错误类型分布;上次只留了 200 行过滤样本),以及修复在 6 组模型上的实际行为
   (预期:响亮打挂引擎,而非静默错答案)。两种结果都有信息量。
2. 顺着第三节:`blocks_per_chunk` 非均匀这条(`downsample_and_stage_block_ids`、
   下溢守卫 `len(group_block_ids) < num_chunks * bpc`),但要先解释"1056 题后才坏"的时间形状。
3. 修那两个套件场景,让它们把混合模型路由到 MP harness —— 能关掉 4 张证书上的两条
   `known_not_covered`,也是抢占/驱逐路径唯一的测试入口。
4. 之后 Qwen3-Omni(第一个音频模型,8 张证书都写着 audio 未覆盖;需换 AIR-Bench/MMAU,探针重做)。
5. 确认 fork 上三条分支无误后删 `backup/pre-artifact-rewrite`(1.7 MB artifact blob 的唯一去处,永不推)。
6. 存量:`achievable_hit_tokens` 分母、MP retrieve 上报延迟、心跳连续失败策略、
   P5 bypass guardrail、MP race flake、`extra_keys` 重构。
