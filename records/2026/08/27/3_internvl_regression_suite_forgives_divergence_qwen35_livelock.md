# 重发旧证书翻出一张假绿:InternVL3.5-2B 在 0.27.1 上读路径失败;套件看得见却原谅了它;qwen3.5-2b 活锁

日期:2026-08-27(接 `2_`,同一段会话)
代码状态:`multi_modal@c64eae85`,工作树干净。本篇新增一个提交:

| commit | 内容 |
|---|---|
| `c64eae85` | 把被救回的逐字节背离记进证书;`flip_asymmetry_p` 的公平硬币论证 |

用户在本段中途给了一条改变优先级的指令:**"我会在把所有模型都支持一遍后再
重新全跑一遍。现在我们先让每个 model 都变 supported。"** 所以"为一致性重跑"
一律推后,当前唯一目标是把每个模型推到 SUPPORTED。

## 一、头条:重发第一个模型就翻出一张假绿

`2_` 第五节把"重发 9 张旧证书"从"记账"改判成"跨大版本复验"。第一个跑完的
就证明了改判是对的。

**InternVL3.5-2B,0.23 上三段逐字节完全相同,0.27.1 上读路径塌了:**

| | 0.23(旧证书 schema 4) | 0.27.1(今天) |
|---|---|---|
| pass2 vs pass1 答案翻转 | **0** | **44** |
| 解析翻转 | 0 | **86** |
| 退 / 进 | — | **35 / 9**,p = **5.3e-05** |
| parse ratio(p1 → p2) | — | 0.9368 → **0.9065**(门限 0.02,超了) |
| 总分 base / p1 / p2 | 1696.61 / 1696.61 / **1696.61** | 1707.12 / 1707.12 / **1657.20** |
| pass1 vs baseline | 0 | **0,逐字节相同** |
| 合成套件 | 绿 | **绿(46 项 0 失败)** |
| 判决 | SUPPORTED | **NOT_SUPPORTED** |

三条硬事实把范围锁死在读路径上:pass1 与无 LMCache 基线**逐字节相同**;
hit coverage 1.0008,缓存确实命中;掉的 49.92 分全在 pass2。

**这不是调预算能过的。**它同时挂在方向门(p=5.3e-05)和 parse ratio delta
(0.0303 > 0.02)上,放宽翻转计数对这两条都不起作用。

注意两个变量同时变了(vLLM 0.23→0.27.1,以及我们自己的代码),所以还不能
说是 LMCache 的缺陷 —— 定性靠第三节的控制实验。

## 二、套件不是瞎,是**看见了然后原谅了**

这是本篇最有价值的发现,而且它推翻了我自己半小时前的说法。

我先说的是"套件的图全是 448×448(`catalog.py:51`),所以对切图模型结构性
失明"。翻 certify 日志才发现更准确的事实:**逐字节 oracle 触发了 10 次**,
分布在 T0.2(碰撞压力)、T0.4(chunk 边界相位)、T0.5(混合流量):

```
[T0.2 replay] hit-path text diverged from miss-path but semantic probe passed
[T0.5 MM A repeat] t05-A: exact baseline mismatch but semantic probe passed
```

`harness.check_text` 的策略原文:

> exact match against the plain-vLLM baseline is required. If the exact match
> fails but the semantic probe still passes, the step passes with a warning
> (GPU nondeterminism); if the probe also fails, this is cross-image
> contamination and the step fails hard.

这个逃生口有正当理由(±1 bf16 量子确实存在,会改措辞不改答案),但它是
**无条件**的:救多少次都行,而且只留在 pytest 的 warning 里,证书里一个字
都没有。跨模型看,这个计数是有区分度的:

| model | 套件里被救回的逐字节背离 | MME parity |
|---|---|---|
| internvl3.5-2b | **10** | FAIL,35/9,p=5.3e-05 |
| mistral-small-3.1-24b | 0 | PASS,4/4 |
| qwen3-vl-2b | 0 | PASS,2/6 |

**一个 16 分钟的套件跑出来的计数,和一个 1.5 小时 parity 的结论一致。**

`c64eae85` 把它记进证书:`harness.record_byte_divergence()` 在三个救援点各
写一条 JSONL(`baseline` / `replay_extracted` / `replay_probe`),
`certify.run_suite` 用 `LMCACHE_MM_E2E_DIVERGENCE_LOG` 收集,汇总成
`byte_divergences` + `byte_divergence_kinds` 进 `suite` 块。**只记录不设门**
—— 一个失败模型不构成阈值标定,理由写在 docstring 里了。环境变量不设时
(单跑 pytest)一行不写,行为不变。验证:三条写入正确、取消变量后不再追加、
`test_parity_gate.py` 17 项全过、ruff 干净。

## 三、控制实验(在跑,GPU 2)

pass2 相对 pass1 唯一还没被排除的差异是 **batch shape**。`run/control_run.py`:
纯 vLLM 跑完整 MME,`max_num_seqs` 分别 256 / 32,全程不碰 LMCache,
`control_compare.py` 按同一套 reg/imp 逻辑比。对象从 DeepSeek-OCR 换成
**InternVL** —— 效应大 10 倍,而且有"0.23 上 0 翻转"这个干净历史基线。

- 复现单向 → vLLM 0.27.1 数值行为,读路径无罪,门要重设计;
- 纯 vLLM 基本 0 翻转 → **LMCache 读路径真缺陷,发版阻断级**。

实现上吸取了 `../26/15_` 那次六个引擎塞一个进程 OOM 的教训:一个进程一个
引擎,各自把答案连同 ground truth 落盘,比较步骤不再重新加载数据集
(MME 加载一次 ~12 分钟)。

顺带记一个之前的猜测**被否掉**:mm 哈希本身是对的。
`apply_mm_hashes_to_token_ids` 把每个图像占位 token 换成
`(mm_hash, 位置)` 派生的 31-bit 值,`get_token_ids` 也正确处理了 prompt /
decode 分界。所以"占位 token 数相同 → 键相同 → 命中别人的图"这条在设计上
排除,套件的 `test_t0_collision_pressure`(issue #3301 那次 16-bit 截断的
回归测试)也专门压过。

## 四、qwen3.5-2b 活锁,以及我编排里的一个更糟的洞

**症状**:pass2 里 07:47 之后 19 分钟日志零增长。537 个线程 536 睡、
**1 个跑满一个核**(实测 1016 ticks / 10s = 101%),GPU 87 GB 占着但利用率
0%,MP server `active_sessions: 0`、对象数 15 秒不变,进度条停在
`2%|41/2374`(39→41 用了 14 秒然后静止)。自旋活锁,不是在算。

前置线索是 07:42:20 的 `Failed to reset prefix cache because some blocks (8)
are not freed yet`。**这条本身是预期的** ——
`harness.reset_vllm_prefix_cache` 专门处理过:公共 API 拒绝就强制清索引,
docstring 里连"4 of 12405 measured"都记了。所以它不是 bug,但 4 分钟后进
pass2 就死了。qwen3.5-2b 是唯一的 hybrid(`hybrid_block_tokens=544`,
`recurrent_state`),也是唯一开着 vLLM prefix caching 跑的,和 backlog 里
`defer_block_free` preemption 活锁那条对得上。

**清理时暴露了我自己的洞**:杀掉挂死进程后,chain 看到 pid 退出就直接去
certify —— 而盘上根本没有 parity report。那一步已经起了 pytest 和两个
baseline 进程,再晚几分钟就会产出一张**没有 parity 依据的证书**。全部杀干净
(GPU 7 回到 4 MiB),TERM 杀不掉的 MP server 补了 KILL。

重写成 `run/chain3.sh`,两条教训写进脚本注释:

- **失速看门狗**:parity 日志超过 `STALL_S=1200` 秒不增长就杀掉并报告。
  阈值按最长合法静默期定 —— MME 加载父进程 ~12 分钟、baseline 子进程再
  ~12 分钟。
- **certify 前检查 report 存在**,不存在就跳过并明说。

处置:先重试一次判"偶发还是必现"。同一点再卡就是确定性的,得动配置绕开
(它 `gpu_memory_utilization` 只有 0.6,抬高或限 `max_num_seqs` 都能压掉
抢占压力)。

## 五、当前台账

```
key                      sch verdict        vllm    gate   stable tests
deepseek-ocr               8 PROVISIONAL    0.27.1  None   True      45
gemma-4-e4b                8 SUPPORTED      0.27.1  True   True      44
glm-4.6v-flash             2 SUPPORTED      None    True   None      29
internvl3.5-2b             8 NOT_SUPPORTED  0.27.1  False  False     46
mistral-small-3.1-24b      8 SUPPORTED      0.27.1  True   True      45
qwen2-vl-2b                8 SUPPORTED      0.27.1  True   True      46
qwen2.5-vl-3b              8 SUPPORTED      0.27.1  True   True      46
qwen3-vl-2b                8 SUPPORTED      0.27.1  True   False     46
qwen3.5-2b                 3 SUPPORTED      None    True   None      26
qwen3.6-27b                3 SUPPORTED      None    True   None      26
qwen3.8-27b                3 SUPPORTED      None    True   None      26
```

在跑:GPU 2 InternVL 控制(seqs=256)、GPU 3 molmo2-4b parity → qwen3.6-27b、
GPU 7 qwen3.5-2b 重试 → glm-4.6v-flash → qwen3.8-27b。
树里没有 gemma-3-4b / molmo2-4b / qwen3-omni-30b 的证书(只在 `../22/all12/`)。

**`stable: False` 的两张(internvl3.5-2b、qwen3-vl-2b)**:是我上午那段
`flip_asymmetry_p` docstring 让工作树变脏期间跑的。已量化过,污染概率实际为
零 —— 全部改动是 10 行纯 docstring,无一行可执行代码;套件跑的
`test_mm_acceptance.py` + `harness.py` 根本不 import `benchmark_parity`;
`git checkout` 后 `git status` 为空,证明脏量被完整测量。**这是出处问题不是
正确性问题**,按用户指令推到最后统一重跑时自然解决。发现后立刻把补丁存成
`pending_flip_null_docstring.patch` 并还原了树,后续证书不再被波及。

## 六、我这段犯的错

1. **"解析翻转这一列是完美分离的"说早了。** 我拿 5 个模型排了"切图 ⇔ 有解析
   翻转"的表,把 gemma-4-e4b 标成"待查(pan-and-scan)"就发出去了。查完
   `Gemma4ImageProcessor` 是固定 280 soft tokens、无任何切图键 —— **不切图
   却有 15 个解析翻转**,这条相关直接被证伪。**待查项不该出现在结论句里。**
   剔掉后还站得住的只有方向单向性(2 个切图模型单向 vs 4 个不切图的全平)。
2. **先说"套件因为图都是 448×448 所以看不见",其实它看见了。** 正确说法是
   有信号但被无条件降级成 warning。差别很大:前者要加用例,后者要把已有信号
   接出来 —— 后者便宜得多,而且就是 `c64eae85` 做的事。
3. **chain 在 parity 没产出 report 的情况下照样 certify。** 差点产出一张没有
   parity 依据的证书。已在 `chain3.sh` 修掉。
4. **第一次读 tick 差算错了一个数量级。** 1018 ticks/10s 我念成"约 10 个核",
   实际是 101% 一个核。结论没变(都是在自旋),但差点让我按"多核忙 = 在做
   memcpy"去解释,方向是反的。

## 七、下一步(顺序即优先级)

1. **InternVL 控制实验的结论** —— 决定 InternVL / DeepSeek 这条是改门还是修
   代码。这是"让每个 model 都 supported"的关键路径上唯一的岔路口。
2. **qwen3.5-2b 重试结果** —— 偶发就继续,必现就动配置绕开活锁。
3. 剩余重发跑完:molmo2-4b、glm-4.6v-flash、qwen3.6-27b、qwen3.8-27b、
   gemma-3-4b;qwen3-omni-30b 单独处理(音频 + 30B,MMAU 在 0.27.1 上从没验过)。
4. MiniCPM-V 4.6:上游 `minicpmv4_6.py` 要融合 `qkv_proj`、权重给分开的
   `k_proj`,配置绕不开(证据 `../25/vllm_upgrade/minicpm.json`)。
5. **等用户决定**:Tier 4 的 Llama 4 / Kimi-VL / Step-3 权重要不要下到
   `/raid/data/hub`(共享盘写入)。Kimi-VL 是唯一能补 MLA 覆盖的。
6. `/var/crash/_usr_bin_python3.12.1016.crash`(89 MB,属主 bo)要不要删。
7. 全部 supported 之后按用户指令统一重跑一遍。
