# 0.27.1 抽样六跑收结果:四绿一红,抢占挂死

**日期**: 2026-08-26(当天第 6 篇,接 `5_`;`5_` §四.1 已改为指向本篇)
**代码状态**: `multi_modal@9c5a7d0f`,工作树干净(本篇不含仓库代码改动)
**测量树**: `multi_modal_verify@2485fdbc` = `9c5a7d0f` + 三个 base 修复
**日志**: `records/2026/08/26/run0271/`(5 个套件日志 + 抢占空日志 + molmo2 单场景重跑 + 两个启动脚本)

## 结论先写

1. **五个抽样套件:四个全绿,一个一红。**

   | 模型 | 结果 | 用时 | `5_` 预估用例数 |
   |---|---|---|---|
   | qwen2-vl-2b | 28 passed / 4 deselected | 776.35s (12:56) | 28 ✓ |
   | qwen3.5-2b | 27 passed / 3 deselected | 895.10s (14:55) | 27 ✓ |
   | **gemma-4-e4b** | **27 passed / 3 deselected** | 1293.50s (21:33) | 27 ✓ |
   | molmo2-4b | **1 failed** / 24 passed / 6 deselected | 646.88s (10:46) | 25 ✓ |
   | qwen3-omni-30b | 30 passed / 2 deselected | 950.37s (15:50) | 30 ✓ |

   五个用例数与 `5_` 事先按 `isolated_routing.isolated_scenarios()` 谓词算出的
   预估**逐个吻合**,门控判断可复核。

2. **`gemma-4-e4b` 首测全绿。** 这是抽样里最凶的一条路 —— SLIDING_WINDOW 混合、
   **6 个 KV 组 / chunk 32 / 2 个 object group**,此前从未在任何 vLLM 版本上测过。
   因此 `5_` §五.1 的"若 gemma-4-e4b 红就立刻拉 gemma-3-4b 做受控对照"**取消**。

3. **`qwen3-omni-30b` 全绿**(唯一 audio + cross-modal),**30 个用例最多**。

4. **唯一的红:`capacity_eviction[molmo2-4b]` —— 服务端一个字节都没收到。**
   allocator 自检:`Total active allocations: 0`、`Total allocated size:
   0.000000 MB`、`Total free size: 64.000000 MB`;32 个请求的 pass-1 命中全 0。
   **尚未定性**(见 §一)。

5. **抢占在 0.27.1 上是"挂死",不是"失败"。** 1500s 外层硬杀,零输出。
   与 `3_` 的 `defer_block_free` 活锁机制一致。

## 一、molmo2-4b 那个红:已知的与未知的

### 1. 两条 failure 只有一条独立信息

报告里两条:

```
"no resident keys after the eviction traffic"
"traffic stored only ~0 bytes against a 67108864-byte cap -- eviction never
 exercised; raise EVICTION_N, or lower this model's capacity ... "
```

第二条是**派生的**:`isolated_cases.py:463-466` 算
`bytes_per_token = total_bytes / max(1, num_keys * chunk)`,`num_keys=0` 时它
恒为 0,于是 `intended_bytes=0` 必然触不到 `> 2 * capacity_bytes`。所以它给的
建议(raise `EVICTION_N`)**恰好是错的方向** —— 流量不是不够,是根本没落盘。

实测(`capacity_eviction_molmo2-4b.json`):
`resident = {num_keys: 0, total_bytes: 0, capacity_bytes: 67108864,
intended_bytes: 0}`,`capacity_gb = 0.0625`,`pass1_hits` 32 个全 0。

### 2. 两次跑是两种失败形态

* 套件里(`s_molmo2_4b.log`):子进程 **exit 1 且没写 JSON**,测试报
  "crashed before reporting";
* GPU 5 上单独重跑(`iso_molmo-evict.log`):**正常写出 JSON**,报上面两条 failure。

同一场景两种形态,说明有不确定性。**套件那次的原始 stderr 已经丢了**
(见 §三.2),没法回溯它是不是同一个原因。

### 3. 排除项:不是"molmo2 存不进"

同一个套件里 molmo2-4b 的 **24 个验收用例全绿**,含跨图隔离与命中等价 ——
存取在 40 GB 的 session L1 上完全正常。所以结论只能收窄成
"**在 64 MiB 这个档位上存不进**"。

### 4. 最可能的假设,与待判实验

`isolated_cases.py:95-106` 那段注释已经预告过这个档位问题:默认
`EVICTION_CAPACITY_GB = 0.0625` 是按"一个 cache object"选的,注释里列了实测值
(Qwen2-VL 一个对象、Gemma 3 184 MB/2.7x、Qwen3.5-2B 12 MB),并明说
"只有 27B 级混合的 ~154 MB 状态页需要更大,它们通过
`ModelSpec.eviction_capacity_gb` 自己声明 —— **例外的依据是 per-model object
size,不是 hybrid family**"。

假设:molmo2-4b 的**单个 cache object 超过 64 MiB**(它是唯一的
mm-prefix-LM,`mm_bidirectional_attention=True`,整段 media 可能不可切分),
于是每次分配都被拒,一个字节都留不下。

待判实验(**一次就能定性**):给 molmo2-4b 的 spec 加
`eviction_capacity_gb=0.5`(与两个 27B 同档)重跑该场景。

* 绿 → 就是档位问题,改 `specs.py` 一行,并把量到的对象大小补进那段注释;
* 还红 → MP 存储路径对 mm-prefix-LM 真有问题,单独查。

## 二、抢占:挂死的后果

`p_qwen2vl2b_preempt.log` **0 字节**,pytest 被外层 `timeout -s KILL 1500`
在 09:34 杀掉,整组回收。定性为挂死。

**后果得记下来**:`test_isolated_paths.py` 给子进程的是
`subprocess.run(..., timeout=2400)`。也就是说,不加外层 timeout 的话,
**每个抢占场景要占 40 分钟才变红**。当前 12 个模型里会跑 `preemption` 的那些,
在 0.27.1 上就是每个 40 分钟的纯等待。

处置仍是 `3_` §三 的未定项:要么场景层面加短超时,要么按已知上游缺陷跳过。
本篇只补一条数据:它确实不会自己失败,只会耗到超时。

## 三、过程里的事故与教训

### 1. 失败文案要分清"根因断言"和"派生断言"

第二条断言在 `num_keys=0` 时必然触发,而它带的建议指向反方向。诊断文案只应在
自己是根因时给建议 —— 否则会把人往"提高流量"上带,而真相是"一个字节没存"。
这是本篇唯一一条**可以直接改代码**的教训。

### 2. 2000 字符的 stderr tail 吃掉了 traceback

`test_isolated_paths.py:58` 报错时只带 `proc.stderr[-2000:]`。LMCache 关停要打
十几行 PeriodicThread 日志,真正的 traceback 必然被顶出窗口 —— 套件那次报的
"crashed before reporting (exit 1)" 后面跟的全是关停噪声,零信息。
子进程 stderr 应当**落盘**,不是截尾。

### 3. "ps 里没了"不等于"被杀了"

收尾时 `ps` 里找不到我的 4 个 pytest,而 GPU 0/1/2/5/6 上冒出另一个 session 的
5 个 `vllm-lazy` Qwen3-Coder 服务,我一度以为自己的跑被挤掉了。**先看日志尾巴**:
五个套件全部打了 `N passed ... in Ns`,都是正常跑完退出的。同一族错误:
用"我看不见它"代替"它的输出说什么"。

### 4. 组 kill 收不走已 reparent 的 MP 服务端

清理时找到两个孤儿,都是 ppid=1、进程组已死、只在 LISTEN 挂着、
零条 established 连接的 `lmcache.v1.multiprocess.http_server`:

| pid | 来源 | 常驻 | 占用 |
|---|---|---|---|
| 3802172 | `5_` §三.1 那次 TMPDIR 撞车被 kill 的跑 | 25 分钟 | 5.0 GB RSS + GPU 7 518 MiB |
| 1937373 | 0.25.1 bisect 那个 session(`vllm-bisect-0.25.1` venv) | **20 小时 28 分** | 5.0 GB RSS + GPU 3 610 MiB |

`setsid` + 组 kill 只收当时还在组里的进程;MP 服务端一旦 reparent 到 1 就漏出去。
**判"是不是我的"要按 venv + pgid,不能按卡号** —— 同机还有另一个 session 的
`vllm-lazy` 五个服务在跑,按卡号清会误杀。清完 GPU 3 / GPU 7 各回到 4 MiB。

### 5. 抽样的门控预估要事后对账

`5_` 事先按谓词算出 28/27/27/25/30,实测逐个吻合。这条对账让"抽样"这件事变得
可复核:数字不对就是门控读错了,而不是"跑的时候大概少了几个"。

## 四、诚实边界

1. **molmo2-4b 那个红没定性**,只排除了"molmo2 完全存不进"。§一.4 的假设未验。
2. **套件里那次 exit-1-无-JSON 的形态没有复现**,原始 stderr 已丢。
3. **跳过的 6 个模型在 0.27.1 上仍未测**,含两个 27B。
4. **`preemption` 只在 qwen2-vl-2b 上确认挂死**,其余 4 个抽样模型未跑,按 `3_`
   的机制外推(vLLM 侧开关与模型无关),**未逐模型验证**。
5. **时长数字不能当性能结论**:六跑并行共享主机,同机还有另一个 session 的
   5 个 30B 服务。
6. 测的是 verify 树(PR HEAD + 三个 base 修复),不是 `multi_modal` 单独;三个
   修复要单独走上游。
7. 仍未做:证书 schema 加 vLLM 版本字段;MME parity 重跑;`certify.py` 端到端;
   12 个模型 MP-only 重新出证。

## 五、下一步

1. ~~molmo2-4b 加 `eviction_capacity_gb=0.5` 重跑 `capacity_eviction`,定性。~~
   **已做,见 `7_` §一:是档位,0.5 GB 全绿,`93d31dfa`。附带推翻了那段注释里
   "单位是一个 cache object"的说法。**
2. ~~`test_isolated_paths.py` 的子进程 stderr 落盘。~~ **已做,`687bede3`。**
3. ~~`preemption` 场景的处置决定。~~ **已做,见 `7_` §二:场景钉
   `async_scheduling=False`,`4debe5d0`;超时改 900s 而非分钟级,依据在 `7_` §三。**
   五个套件已在 `7_` 转为 5/5 全绿。
4. 决定是否补测 qwen3.6-27b / glm-4.6v-flash(`5_` §五)。
5. 上游:`defer_block_free` 活锁的最小复现(脱离 LMCache);`#4463` 补
   "MP 不受影响"实测。
