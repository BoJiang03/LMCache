# audio 模态落地:Qwen3-Omni 解锁、MMAU 全量 parity 通过、keying 用负控证明

日期:2026-08-22 06:50 PDT

接 [`6_`](6_omni_audio_start_storage_moved_probe_and_benchmark_chosen.md) 第六节的六步计划。这一篇是 audio 这条线从"起点"走到"第一个模型认证就绪"的记录:1-4 步全部完成,`6_` 里两处**被实测推翻的结论**也一并更正。

代码改动落在 8 个文件(`benchmark_parity.py` / `catalog.py` / `harness.py` / `specs.py` / `certify.py` / `conftest.py` / `test_mm_acceptance.py` / `README.md`),已提交为 **`8519c60c`**;工作区干净,本地领先 `fork/multi_modal` **7 个提交**,**未推送**(等明确指令)。

---

## 一、Qwen3-Omni 在 vLLM 0.23.0 上其实能跑 —— 更正 `6_` 的封锁结论

`6_` 记的是"Qwen3-Omni-30B 起不来,退回 Qwen2.5-Omni-3B,把 30B 记为 0.23.0 阻塞"。**这个结论是错的**,需要两个东西,而且**都不是打补丁**:

| 尝试 | 结果 |
|---|---|
| 默认(audio+image) | **video** profiling 崩:`cu_seqlens_q must be on CUDA` |
| `video: 0` | 同样的崩,改在 **image** profiling |
| `image: 0, video: 0` | 换了个崩法:`Tensor on device meta` —— 把某个模态设 0 会跳过权重加载 |
| **`mm_encoder_attn_backend="TORCH_SDPA"`** | **引擎起来了** |

关键认识:**`limit_mm_per_prompt` 绕不开坏掉的 vision tower**。省略某个模态并不会禁用它(vLLM 照样 profile video),而显式设 0 又会跳过权重加载。`TORCH_SDPA` 是官方支持的配置开关(`vllm/config/multimodal.py:158`),它完全避开了那个 kernel。

第二个东西更平淡:MMAU 的 WAV 是 32 kHz,vLLM 要用 pyav 重采样,而 `av` 没装 —— `ImportError: Please install vllm[audio]`。之前的合成探针都是 16 kHz,所以从没触发过。装 `av==18.1.0` 即可,**这是 upstream 声明的 optional extra,不是本地改 vLLM**。

所以认证目标**仍然是 30B**,`6_` 里准备写的"排除"不需要了。`TORCH_SDPA` 是这个模型的**必需配置**而非调优偏好,已进 `ModelSpec.mm_encoder_attn_backend`(upstream 修复在 0.27.1 的 `qwen3_omni_moe_thinker.py:982`)。

---

## 二、几何:非 hybrid,两条独立证据

- config:thinker text tower **48 层全是 full attention**,没有 `layer_types`,`sliding_window=None`;MoE 只在 FFN,不影响 KV 几何。
- 引擎日志只有**一对** `GPU KV cache size` / `Maximum concurrency`(多组模型每组一对)。

KV = 48 层 × 4 KV heads × 128 head_dim × 2 × 2B = **96 KB/token**,与引擎自报的 56.02 GiB / 611,888 tokens = 98,300 B/token 吻合。

结论:走**in-process 路径**,不是 Qwen3.5/3.6/3.8 那种 recurrent-state hybrid,四个 isolated scenario 全部适用。~234 token/题 × 1000 题 ≈ 28 GB,在 runner 默认 40 GB 之内,**不需要 capacity override**。

---

## 三、MMAU:两个采样缺陷,一个自证虚假的假设

smoke 先在 40 题上跑出 parse_ratio 1.0 / accuracy 0.725 / determinism 1.0 —— 看着很好,但 `accuracy_by_task` **只有一个 key**。

查下去发现三件事:

1. **shard 是按 task 成段存的**,不是混的。333 sound / 333 speech / 334 music,但分段长度是 96, 333, 301, 48, 33, 189 —— 取前 40 行全是 sound。而我自己在 `load_items` docstring 里写的是"MMAU 的 shard 已经按 task 和难度混过了"。**这句话是错的,而且是被自己的输出量出来的。**
2. **静默丢行**:970 行 4 选项、**20 行 5 选项**、10 行 2 选项。原来的 `2 <= len(choices) <= 4` + `LETTERS="ABCD"` 把那 20 行悄悄扔了 —— 2% 的 benchmark 无声消失,正是 `mme_min_parse_ratio` 存在的理由那类失败。
3. parse 函数还得挡**越界字母**:2 选项的题答 "C" 不能算命中,否则把幻觉选项记成真答案。

修法:按 task **round-robin** 选行(仍然完全确定、不 shuffle,但任意长度的前缀都覆盖三个 task),`LETTERS="ABCDE"` 收下 5 选项行,音频**第二遍只读选中行**(不必解码全部 2.84 GB)。

分层后重跑 45 题,数字变了,而且变的方式很重要:

| 指标 | 只有 sound(40) | 分层(45) | 全量(1000) |
|---|---|---|---|
| parse ratio | 1.0 | 1.0 | **1.0** |
| determinism | 1.0 | 1.0 | — |
| accuracy | 0.725 | 0.689 | **66.90** |
| speech | — | 1.00 | **59.16** |
| sound | 0.725 | 0.60 | **71.47** |
| music | — | 0.467 | **70.06** |
| prompt tokens | 171-186 | 172-483 | 均值 ~234 |

**per-task 跨度 59-71(45 题样本上甚至 0.467-1.00)是 MMAU 必须分 task 记分的原因**:只看总分会把只发生在一个 task 上的回归平均掉。这也说明 45 题样本的 per-task 数字**并不代表全量** —— 又一次"小样本别外推"。

### 一个自己吓自己的中间结论

中途我怀疑 "prompt token 不随音频时长变化",因为 40 题的 token 只有 171-186 而数据集时长是 3.4-34.5 s。**错的**:那 40 题全是 sound,时长其实只有 9.3-10.0 s,0.7 s 的跨度,相关系数 +0.74,残差由题干长度解释。expansion 一直在正常工作。

同样,我一度担心 audio 太短(~181 token)会让 parity 变成空测。**也是错的**:`LMCACHE_TEST_CHUNK_SIZE = 16`(不是我以为的 256),181 token 的 achievable 上限是 176,raw hit ratio 天花板 0.97,远高于 0.8 的地板。不需要动 chunk。

---

## 四、benchmark 抽象:小抽象,不是 fork

`Benchmark` ABC 只有四个抽象方法(`load_items` / `conversations` / `parse_answer` / `scores`)加一个 `default_mm_processor_kwargs`;三个 pass、计数器、hit-coverage 算术和 gate 全部共享。

必须保证的兼容性:`certify.py` 会**吃已经录好的 report**。所以:

- `MAX_SCORE_DELTA` 从模块常量搬到每个 benchmark 上(MME 2800 分制 10 分,MMAU 100 分制 1.0),模块里那个常量**删掉**而不是留着不用。
- `parity_gate` 读 `report.get("benchmark", MME_KEY)` —— **没有 `benchmark` 字段的老 report 仍按 MME 的 10.0 判**,已验证。
- MME 的 parser 在 11 种答案形状上逐一比对,行为**逐字节不变**;pixel cap 也不变。

`engine_kwargs` 现在从 benchmark 拿 `limit_mm_per_prompt` 的 modality 和 processor kwargs 兜底,并新增 `mm_encoder_attn_backend`。

**certify.py 的两个真缺口**(不补就是错的认证):它拼 parity 命令时既不传 `--benchmark`(于是给 audio 模型跑 MME 图像基准)也不传 `--mm-encoder-attn-backend`(于是 Qwen3-Omni 根本起不来)。新增 `ModelSpec.parity_benchmark` 后两个都从 spec 走。

### 全量 1000 题 parity:通过

```
flips pass2 vs pass1      0/1000
flips pass1 vs baseline   0/1000
score delta               0.00 / 0.00
lookup hit ratio          1.000
external cached           233,240 / 233,984 tokens
parse ratio               1.0
```

三个 pass 的分数**逐位相同**(66.90;music 70.06 / sound 71.47 / speech 59.16)。

这就是 `6_` 第四节要求的标定数据:**实测抖动地板是 0**,而且包含 `baseline vs pass1` 这个跨进程、跨引擎配置的比较 —— MME 的 0.5% 当年就是从这个比较里标出来的。所以 MMAU 的闸门不是拍的:5 次 flip / 1.0 分是**建立在实测 0 之上的余量**。代码注释里那句 "PROVISIONAL, not yet measured" 已按实测改写,同时保留一句"一次实验只能给地板定上界,不能证明它永远是 0"。

---

## 五、audio oracle:在**认证目标**上重测,3B 的结论不可移植

`6_` 第三节定的 oracle(`sound_kind` = tone/noise/silence)是在 **Qwen2.5-Omni-3B** 上测的,而认证目标是 30B。这是个真缺口,补测结果直接推翻了它:

| | 3B | 30B |
|---|---|---|
| tone | ✓ | ✓ |
| noise | ✓ | ✓ |
| **silence** | ✓ | **✗ → "tone"(稳定)** |

**30B 稳定地把静音叫成 "tone",和真 tone 撞车。** 撞车正是让检测器**瞎掉**的那种失败(A 拿到 B 的缓存答案,两者答案相同时完全不可见),所以 silence 不能用。

只剩 2 个值太窄(图像那边有 6 色),于是没有直接接受,而是问"能不能更宽"。用一个六选一的问题测四个新候选:

| 刺激 | 回答 | correct | stable |
|---|---|---|---|
| tone | `tone` | ✓ | ✓ |
| noise | `noise` | ✓ | ✓ |
| silence | `tone` | ✗ | ✓ |
| beeping | `beeping` | ✓ | ✓ |
| rumble | `rumble` | ✓ | ✓ |
| warble | `warble` | ✓ | ✓ |

**去掉 silence,剩下 5 个 kind 全部 correct + stable + 两两互异。** palette 从 2 涨到 5,接近图像的 6。

注意这里的**边界**:`beeping` 作为**种类**可靠,但 beep 的**数量**不可靠(`6_` 已量过)。所以 palette 只说种类,永不说个数。

顺带否掉的:**有序对**(两段音频、按序命名)在 3B 上 **0/9 correct、6/9 distinct**。所以 audio 探针**只能单段**,`mm_limits` 里 audio 上限就是 1。(9 个里还有 1 个不稳定,但那条答案像是被 12-token 预算截断,决定性的失败是 correct/distinct,不是确定性。)

---

## 六、两个自己写出来又被自己的检查抓到的 bug

这两个都属于"不测就会静默失效",记下来。

### 1. LCG 低位做 dither —— 整类音频跨 index 逐字节相同

`audio_data_uri` 给每个 index 加 ±1 LSB 的抖动,目的就是让**同一 kind 的不同 index 有不同字节**(否则同 kind 之间的假命中原理上不可见)。写完一查:index 0/3/6 (tone) 和 2/5 (silence) **哈希完全相同**。

原因是 LCG 的经典弱点:模数是 2 的幂时,`(1103515245 * state + 12345) & 0x7FFFFFFF` 的**最低位以与种子无关的固定模式交替**。所以 `next(rand) & 1` 给每个 index 发了同一串 dither。noise 逃过一劫只因为它取的是高位。改用 bit 16 后:13 个 index 全互异,22 个同 kind 对全不同。

**这个 bug 的讽刺之处:dither 存在的唯一目的就是防止这种碰撞。**

### 2. `AUDIO_SECONDS=1.5` 让隔离断言**不可能成立**

隔离断言要求"不同音频的命中比满命中少至少 `image_span_margin` = 4 chunk = **64 token**"。而 CPU 上精确量出 Qwen3-Omni 的 audio expansion 是 **13.1 tok/s**:

| 秒 | span(token) | chunk 数 |
|---|---|---|
| 1.5 | **20** | 1.2 |
| 6.0 | 79 | 4.9 |
| **8.0** | **105** | **6.5** |

1.5 s 的 span 只有 20 token,**比要减掉的 margin 还窄** —— 无论缓存多正确,那条断言都不可能通过。改成 8.0 s(105 token)。

连带一处:`beeping` 原来是固定 3 声,拉长到 8 s 会变成"1.4 s 蜂鸣 + 6.6 s 静音",模型怎么叫它就不好说了。改成**按时长铺满**(个数从不被探测,只探种类),于是与时长解耦。

---

## 七、keying:用负控证明,不是靠读代码

`6_` 第五节把 keying 列为"整条线里唯一可能是真 bug 的地方"。

读代码的结论是乐观的:`apply_mm_hashes_to_token_ids` **与模态无关** —— 它遍历 vLLM 给的 `mm_features`(`extract_mm_features` 取 `.identifier` / `.mm_position`),不区分 image/audio。

但**读代码不是测量**。而且这里有个特别阴的空测风险:**如果 audio item 根本没进 `mm_features`,正向隔离测试会平凡通过**(没有替换可做,也就没有碰撞),一片绿而什么都没证明。图像的负控证明不了这一点。

所以加了 audio **自己的**负控:关掉 identity 替换后,clip B **必须**假命中进 clip A 的条目。

```
test_t2_audio_isolation_and_hit[qwen3-omni-30b]                PASSED
test_audio_detector_sensitivity_negative_control[qwen3-omni-30b] PASSED
2 passed, 25 deselected in 105.18s
```

负控**跳了**,所以正向断言不是空的。至此:**audio 内容确实进了 LMCache 的 cache key**,这是测出来的,不是读出来的。

catalog 的生成器也是拿**套件自己的代码**验的(`audio_probe_catalog.py` 直接 import catalog),延续"别用手写 runner 验证"那条教训:10 个 index 全部命名正确、全稳定、5 个 kind 在不同 dither 下答案一致、无跨 kind 撞车、prompt token 齐齐 181(与 CPU 量的 180 吻合)。其中 index 0 和 5 都答 "tone" —— 证明 dither 确实听不出来,同时字节又不同。

---

## 八、当前状态 / 下一步

**全量 acceptance suite 已通过:26 passed, 1 deselected**(deselect 的那个是 video,spec 没声明)。

这回答了一个我原本列为"待测"的问题:spec 同时声明了 `image`,而这个模型的 vision tower 正是坏掉那块 —— `TORCH_SDPA` 绕开了崩溃,但图像**回答质量**是否受影响此前未知。现在全部图像用例(T0.1/T0.3/T0.4 十六个 phase/T0.8/T1/T2.1/T2.2/负控……)在这个模型上全绿,所以 SDPA 不只是"能起来",视觉侧的语义探针也照样成立。

下一步:

1. `6_` 第六节剩下的第 5 步:跨模态 T2.x(只换 audio 不换 image / 交换顺序)—— 这是纯图像模型测不到的覆盖面,而这个模型两个模态都已单独验过,正是做这件事的时机。
2. 出 `qwen3-omni-30b` 的证书(acceptance 全绿 + MMAU 全量 parity 通过,两半都已就位)。
3. 未清的旧账:五个 hybrid 证书重生成(Gemma 4 的 JSON 里有两行已不成立)、`block_pool.cache_full_blocks` 崩溃上报、`4_` 第二节那个 retrieve 是否该在 transfer 时续锁的设计问题、in-process 路径上 `pass2_hit_coverage` 恒报 0.0 的误导性(`achievable_hit_tokens` 在这条路径上收到空列表;gate 正确地改用 raw ratio,但 report 里那个 0.0 读起来像失败 —— 这是既有问题,不是 audio 带来的)。

推送仍然**等明确指令**;6 个本地提交未推。

---

## 九、方法论

延续 `5_`/`6_` 那条链,这一轮加两条:

两条都已写进长期记忆(`probe-oracle-is-per-model.md`、`assertion-satisfiability-check.md`),因为它们跨任务复用,而不只是这一轮的结论。

**一、oracle 必须在被认证的那个模型上重测。** `sound_kind` 在 3B 上三项全中,在 30B 上直接撞车。探针属于"模型 × 刺激"这一对,不属于刺激本身。如果我按 `6_` 的结论直接搭套件,silence 那一档会静默失效 —— 而且是以"检测器看不见碰撞"的方式失效。

**二、断言的可满足性本身要算一遍。** `AUDIO_SECONDS=1.5` 那个 bug 不会让测试变弱,而是让它**永远不可能通过**;LCG 那个 bug 不会报错,而是让 dither 静默失去全部作用。两个都不是逻辑写错,而是**没有把量纲对一遍**:span 与 margin、随机位与种子。测试的价值等于它能失败的能力,所以"这条断言在什么情况下会失败"要和"它在什么情况下会通过"一起算。
