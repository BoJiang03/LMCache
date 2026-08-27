# 证书正确性清理:三个同类缺陷,11 个模型全部重出

接 [`8_`](8_crossmodal_t25_landed_and_omni_isolated_fraction_fixed.md) 第七节的
遗留顺序第一项「五个 hybrid 证书重开」。**实际范围比那一项大**:去查 Gemma 4
那两句不成立的话时,发现根子不在那份 JSON 里,而在 `certify.py` 的硬编码上,
而且不止一处。修完之后 11 个模型全部在同一个提交上重出。

提交 **`dc1590c1`**;权威证书集在 `recert/`(见其 `INDEX.md`)。

---

## 一、三个缺陷,同一个物种

都是**证书断言了这次运行从未检查过的东西**。三个都是在已发布的证书里发现的,
不是假想。

### 1. 音频排除项是无条件的

`KNOWN_NOT_COVERED` 里那句「audio modality: the suite has no audio probes」
从 `8519c60c`(音频落地)之后就不成立了。更糟的是它对**每个**模型都发,
包括带着音频认证的 Qwen3-Omni——于是同一份文档里:

```
scope.modalities        = ["audio", "image"]
known_not_covered[2]    = "audio modality: the suite has no audio probes ..."
```

**同一份文档,隔两个块,自己打自己。** 现在只在 spec 没声明 `audio` 时才发,
并且改写了理由:这是关于**模型**的陈述,不是关于套件的——checkpoint 有音频塔
但仍然只按 image/video 认证是合法的(Gemma 4 就是)。

> 顺带更正 `8_` 里我自己写错的一句:那份证书的 `known_not_covered` 我写成
> 「为空」,实际有 5 条。**正是这个误读让上面这条矛盾多藏了一轮。**

### 2. `verdict_meaning` 硬编码 "MME parity"

Qwen3-Omni 是 MMAU 门控的。于是它的证书写着「synthetic suite + **MME**
parity green」,而它自己的 parity 块里 `benchmark: mmau`。第二处自相矛盾。
现在标签从**实际使用的报告**里读,退化到 spec 的 `parity_benchmark`,
最后才是历史默认值。

### 3. `commit` 只在启动时读一次

`8_` 里已经记过这个坑(第一份 omni 证书的 commit 指向被测树之外),但当时
只是「下次注意」。**「下次注意」不是修复。** 现在启动和写出时各读一次 HEAD,
两端都查脏树,记进 `tested_tree`,不一致就往 stderr 打警告:

```json
"tested_tree": {"commit_at_start": ..., "commit_at_finish": ...,
                "dirty_at_start": false, "dirty_at_finish": false,
                "stable": true, "note": "`commit` names the tree under
                test only when `stable` is true; ..."}
```

### 附带:`pass2_hit_coverage: 0.0` 的误导性显示(`7_` 的遗留项)

`MPTransportCounters` **只在 MP 路径**安装,所以 in-process 运行根本没有
per-request 的 lookup 长度,`achievable_hit_tokens([])` 返回 0,coverage 就
算成 `0.0`。于是一个 raw hit ratio = 1.0 的运行,证书上写着「coverage 0.0」
——**一个没测到的量,长着测到了零的脸。** 现在是 `None`;coverage 门控把
`None` 当作不可满足,而不是当作通过的零。

---

## 二、验证方式(不是「跑一遍看着没炸」)

- **门控回归**:对全部 11 份已记录的 parity 报告重算 gate,**11/11 依旧
  pass**,五个 in-process 模型从伪造的 `0.0` 变成 `null`,六个 MP 模型保持
  实测值不变。**没有任何一个判决因为这次修改而改变。**
- **矛盾扫描**:对全部 11 个 spec 检查 scope/exclusion 的自相矛盾,**0 处**。
- ruff check / format 全绿。

---

## 三、我这一轮自己犯的错(和 `8_` 完全同一条)

**我在六个认证跑着的时候改了 `certify.py` 和 `benchmark_parity.py`。**

我刚加的 `tested_tree` 守卫会正确地把这六份全标成 `stable: false`——守卫在
干它该干的事,但代价是六份不可发布的证书。`8_` 第八节第 2 条写的就是这个,
我在写完它一个小时内又犯了一次。

处理方式:全部杀掉(约 8 分钟 GPU 工作作废)→ 提交 → 在干净树上重跑。
**留着跑完再重跑要花两倍时间,而且中间那批本来就不能用。**

有一个副作用值得记:杀 `certify.py` 之后 `isolated_cases.py` 和
`lmcache.v1.multiprocess.http_server` 会**孤儿化并继续占着显存**(4 张卡上
残留 30–53 GiB)。`pkill -f 'certify.py'` 还会匹配到发出它的那条 shell 自己
(退出码 144),得按 PID 杀。这和 `pgrep-wait-loop-self-match` 那条记忆是
同一个陷阱的另一面。

---

## 四、parity 证据的来源,逐个查过

重出 11 份证书需要 11 份 parity 证据。**关键问题:Aug-21 那批 MP 报告跑在
读锁 TTL 修复之前,还能不能用?**

不能靠命中数判断——Gemma 4 那次报告的 hit coverage 是 1.0076,同时坏了
2374 题里的 1288 题。**命中数分不清健康运行和那次事故。** 唯一的判据是
failed-read 日志。

| 模型 | 证据 | 结论 |
|---|---|---|
| qwen3.5-2b | MP server 日志在,`Failed to read prefetched object` = **0** | 可复用 |
| qwen3.6-27b | 同上,**0** | 可复用 |
| **qwen3.8-27b** | **日志已丢** | **重跑全量 MME** |
| gemma-3-4b / gemma-4-e4b | 报告晚于修复 | 可复用 |
| 五个 in-process | TTL 在 MP cache server 里(`MP_SERVER_L1_READ_TTL_S`),`LocalCPUBackend` 没有读锁 | 从未暴露 |

qwen3.8 重跑的结果**反过来确认了旧报告**:flips 13 vs 12,分数在 2800 满分
上差 1.6 以内,hit ratio 0.168 vs 0.164,而且 failed reads = 0、
`read_lock_expired` = 0。旧那份是干净的——但现在这是**测出来的**,
不是猜的。

---

## 五、结果:11/11 SUPPORTED,同一个提交

| 模型 | tests | benchmark | coverage | 说明 |
|---|---|---|---|---|
| qwen2-vl-2b | 29 | MME | null | in-process |
| qwen2.5-vl-3b | 29 | MME | null | in-process |
| internvl3.5-2b | 29 | MME | null | in-process |
| qwen3-vl-2b | 34 | MME | null | +deepstack,测试数最多 |
| glm-4.6v-flash | 29 | MME | null | in-process |
| gemma-3-4b | 27 | MME | 1.0056 | 滑窗 hybrid |
| gemma-4-e4b | 27 | MME | 1.0076 | 滑窗 hybrid |
| qwen3.5-2b | 27 | MME | 1.0 | recurrent-state |
| qwen3.6-27b | 27 | MME | 1.0563 | recurrent-state |
| qwen3.8-27b | 27 | MME | 1.0586 | **全新 parity** |
| qwen3-omni-30b | 31 | **MMAU** | null | image+audio |

全部 `schema_version: 4`、`tested_tree.stable: true`、`commit dc1590c1`、
0 failures / 0 errors / 0 skips。合计 3.5 小时套件时间,分两波跑在
GPU 1/2/3/5/6/7 上(GPU 0、4 有别人的进程)。

**五个 hybrid 的测试数都涨了**,而且涨幅不一样,差别正好说明了原因:

| 模型 | 旧证书 commit | 旧 tests | 新 tests | 增量来自 |
|---|---|---|---|---|
| qwen3.5-2b | `2a522b33` | 26 | 27 | `capacity_eviction` |
| qwen3.6-27b | `da537bf0` | 26 | 27 | 同上 |
| qwen3.8-27b | `da537bf0` | 26 | 27 | 同上 |
| gemma-4-e4b | `ea5a84e1` | 26 | 27 | 同上 |
| **gemma-3-4b** | `d43e817a` | **25** | **27** | `capacity_eviction` **+** `preemption` |

Gemma 3 涨了 2 个,因为它那份证书停在 `d43e817a`,**早于** `b1836ce1`
(让 hybrid 跑 preemption)和 `782d612c`(capacity_eviction 开给所有 hybrid)
两个提交,所以两个场景都是新增的。

`capacity_eviction` 正是 Gemma 4 旧证书声称「不覆盖」的那个场景。这一句是
**自愈**的:`known_not_covered` 里场景形状的条目都从 `isolated_scenarios(spec)`
推导,所以重出即修正,不用手改。这个设计是 `782d612c` 时就做对了的,这次
验证了它有效。

> 更正:本文初稿把这里写成「三个 hybrid 从 26 涨到 27」。实际是四个
> 26→27、一个 25→27。数字记错的方向很典型——**只看了变化最常见的那档,
> 没有逐个对**,而恰好是那个不一样的(Gemma 3 涨 2)才携带「旧证书停在哪个
> 提交」的信息。

---

## 六、并行认证的可行性(顺带测到的运维事实)

6 个认证同时跑在 6 张卡上没有冲突。端口都是 PID 派生的
(`conftest` zmq `24000 + pid%1000`,`isolated_cases` `25000 + pid%5000`),
2 TB 内存里 1.69 TB 可用,HF hub 是只读共享。串行 3.5 小时的活,两波并行
约 55 分钟走完。

一个**尚未咬到但存在**的隐患:`conftest` 的 http 端口区间(25000–25999)
和 `isolated_cases` 的 zmq 区间(25000–29999)**重叠**。单次运行里两者就
同时活着,所以这不是并行引入的;并行只是把碰撞概率乘上了并发数。真碰上会
在 healthcheck 上响亮失败,不会静默。记在这里,别等它发生时重新查一遍。

---

## 七、状态

- **9 个本地提交,未推送**(`dc1590c1` 最新)。`records/` 仍被 exclude。
- 权威证书集:`records/2026/08/22/recert/`(11 份 + `INDEX.md`)。
  旧的 14 份留在各自日期目录里当历史,其中 4 份有已不成立的表述。
- 两份上报文档写好了,可以直接提:
  `escalations/1_block_pool_cache_full_blocks_crash.md`(带排除上游 vLLM
  的对照实验)、`escalations/2_read_lock_renewal_design_question.md`。
- 剩余遗留:`PREEMPTION_MAX_TOKENS` 是否放宽、MP retrieve 上报延迟、
  heartbeat 连续失败策略、P5 bypass 护栏、确认 fork 分支后删
  `backup/pre-artifact-rewrite`(`e33973a8`)。
- 模型顺序下一步:hitchhiker 批(DeepSeek-OCR / Mistral Small 3.1 /
  MiniCPM-V 4.6 / Molmo 2)→ Llama 4 / Kimi-VL / Step-3 →
  Phi-4-multimodal(实质是 `extra_keys` 重构)。Whisper 明确排除。

---

## 八、方法论

第 1、3 条和第六节的运维事实进了长期记忆:新建
`unmeasured-vs-measured-zero.md`(第 3 条)、
`parallel-certification-is-safe.md`(第六节),并把 `pkill -f` 自匹配和
杀掉认证后 GPU 孤儿进程补进 `pgrep-wait-loop-self-match`。第 2 条留在这里,
因为它是关于**文档**的,不是关于操作的。

1. **「下次注意」不是修复。** `8_` 里我把 commit 字段的问题写成了一条注意
   事项,然后一小时内又踩了同一个坑。能变成守卫的教训就不该留在文档里——
   `tested_tree` 现在会自己喊。
2. **一份自相矛盾的文档,比一份陈旧的文档更糟。** 陈旧的会随重出自愈;
   矛盾的是逻辑错误,重出多少次都还在。区别在于:陈旧的事实是**推导**出来
   的,矛盾的是**硬编码**的。所以硬编码的每一句断言都要问「什么条件下它会
   变成假的」。
3. **没测到的量不能长成测到了零的脸。** `coverage: 0.0` 和
   `coverage: null` 在 JSON 里差一个词,在阅读者眼里差「缓存什么都没干」和
   「这个数这条路径上测不出来」。缺省值省的是写代码的事,骗的是读文档的人。
4. **要判断旧证据能不能复用,得找那个能区分两种情况的信号。** 命中数在
   Gemma 4 事故里是 1.0076,和健康运行没有区别;failed-read 日志才是判据。
   日志丢了的那一个就重跑,而不是拿相邻模型的相似性推。
