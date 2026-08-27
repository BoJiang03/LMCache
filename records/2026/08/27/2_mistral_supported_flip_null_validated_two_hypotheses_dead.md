# Mistral 拿到 SUPPORTED,清单见底,方向门的零假设被四个模型证实,DeepSeek 的两个解释全死

日期:2026-08-27(接 `1_`,同一段会话)
代码状态:`multi_modal@39b10ec3` + 一处未提交改动(`benchmark_parity.py` 的
`flip_asymmetry_p` 文档串,内容见第三节)。

本篇的重点是第三节:**方向门的公平硬币零假设不是拍脑袋,是有四个模型的实测
背书的**;以及第四节:DeepSeek-OCR 的单向掉分,两个候选解释都被自己的数据
否掉了。

## 一、Mistral Small 3.1 24B:SUPPORTED

MME parity 06:43 收工,证书 06:57 签发,`certificate_mistral-small-3.1-24b.json`,
`schema_version: 8`,`runtime.vllm.version: 0.27.1`。

| 通道 | 翻转 | 退/进 | 一侧 p | 预算 |
|---|---|---|---|---|
| pass2 vs pass1(命中路径) | 8 / 2374 = 0.34% | 4 / 4 | 0.637 | 11.87(默认 0.5%) |
| pass1 vs baseline(噪声通道) | 5 | 1 / 4 | 0.969 | — |

三段 parse ratio 全是 0.9983,分差 +1.08 / +1.43,hit coverage 1.0004,
套件 45 项 0 失败。**唯一一个不需要放宽翻转预算就过默认门的大模型**,
用的是 `1_` 第五节定下的 260 GB L1 + `isolated_gpu_utilization=0.75`。

## 二、清单见底

`../25/7_` 第四节那份"支持模型清单"是权威版本:Supported 12 个(认证序)
+ Queued 3 个。今天之后:

| 项 | 状态 |
|---|---|
| Queued 的 DeepSeek-OCR | PROVISIONAL,卡在第四节 |
| Queued 的 Mistral Small 3.1 24B | **SUPPORTED** |
| Queued 的 MiniCPM-V 4.6 | 仍红,见下 |
| Tier 4(Llama 4 / Kimi-VL / Step-3) | 权重不在 `/raid/data/hub`,要下载,**等用户决定** |

MiniCPM-V 4.6 在 0.27.1 上复验仍是同一个上游错(证据
`../25/vllm_upgrade/minicpm.json`):

```
ValueError: There is no module or parameter named 'k_proj' in
MiniCPMV4_6ViTWindowAttentionSelfAttn. The available parameters ...
are: {'qkv_proj.bias', 'out_proj.weight', 'out_proj.bias', 'qkv_proj.weight'}
```

vLLM 的 `minicpmv4_6.py` 建的是融合 `qkv_proj`,checkpoint 发的是分开的
`k_proj`。**是上游模型代码和权重对不上,不是我们能靠配置绕开的**,跳过成立。

所以清单上唯一没被堵住的存量活,是那 9 张旧证书。

## 三、方向门的公平硬币零假设,被四个模型证实了

这一节是本篇最有价值的部分,而且它起源于我的一个**错误猜想**。

### 猜想

`flip_asymmetry_p(reg, imp)` 拿公平硬币做零假设。我看到 DeepSeek-OCR 的
逐题清单后提出:这个零假设可能写错了。理由听上去很硬 —— 如果一次扰动只是
把某道边界题的答案重掷一次,而模型在这批题上本来的正确率是 q,那么**退步
的比例自动就是 q**,不需要任何腐坏。q=0.85 的模型翻 11 题就该退 9 题,门
会把正常模型判成腐坏。

### 实测

把五个有逐题记录的模型在两条通道上都重算一遍(`run/analyse_direction.py`,
纯 CPU,不用引擎):

| model | MME 正确率 | pass2 vs pass1 | 退 / 进 | p(公平硬币) | p(按实际正确率) |
|---|---|---|---|---|---|
| deepseek-ocr | **0.495** | 11 答案 + 7 解析 | 8 / 3 | 0.1133 | 0.1064 |
| gemma-4-e4b | 0.664 | 1 答案 + 15 解析 | 0 / 1 | 0.5000 | 1.0000 |
| mistral-small-3.1-24b | 0.845 | 8 | 4 / 4 | 0.6367 | 0.9967 |
| qwen2-vl-2b | 0.832 | 18 | 10 / 8 | 0.4073 | 0.9988 |
| qwen2.5-vl-3b | 0.849 | 24 | 13 / 11 | 0.4194 | 0.9999 |

(五个模型的 pass1 vs baseline 通道:除 mistral 的 5 个翻转外全是 0 翻转,
即 pass1 与无 LMCache 基线逐字节相同。)

**猜想是错的,而且错得有信息量。** qwen2.5-vl-3b 整体正确率 84.9%,按猜想
它的 24 个翻转该是 20 退 4 进,实测 13 / 11;qwen2-vl-2b 83.2% 对应 10 / 8;
mistral 84.5% 对应 4 / 4。三个高正确率模型的翻转方向全部贴着 50/50,离各自
的整体正确率极远。

结论是一句可以写进代码的话:**翻转不落在随机题上,只落在模型本来就拿不准的
边界题上,而边界题按定义就是 50/50 —— 与模型整体正确率无关。**所以公平硬币
是对的零假设,门不用改。这条已经补进
`benchmark_parity.py:flip_asymmetry_p` 的文档串(未提交),把三个模型的实测
数字写在里面,免得下一个人重走这段弯路。

顺带:这张表也说明**为什么这个反驳只能靠 DeepSeek 之外的模型做出来**。
DeepSeek-OCR 自己的正确率是 0.495,公平硬币对它几乎精确(两个 p 值 0.113
vs 0.106 基本重合),拿它自己永远验不出零假设对不对。

## 四、DeepSeek-OCR:两个解释都死了,单向掉分还站着

事实回顾(`1_` 第四节):run1 8 退 1 进(p=0.0195),run2 8 退 3 进
(p=0.1133),各自都在门内;**合并 16 / 4,p=0.00591 < 0.01**,而门没有跨跑
记忆,所以两跑都合法通过。baseline 与 pass1 逐字节相同,合成套件 45/45 全绿
(含逐字节重放 oracle)。

### 假设 A:tiling —— 否

DeepSeek-OCR 是套件里第一个带分块视觉塔的模型(`candidate_resolutions`、
`tile_tag: 2D`、`global_view_pos: head`),而合成套件的图全是同一尺寸,所以
猜测 MME 的杂尺寸照片会走出套件从不生成的 prompt 布局。

用已存的逐题答案做零成本验证(`run/analyse_flips.py`,从 data URI 的 base64
前 32 字符解 PNG IHDR 拿宽高,不解码图像):

```
distinct image sizes=735  most common=[((683,512),106), ((564,240),64), ...]
flipped        n=  11 longest median= 768 mean=1092 [178..4288]  MPx median=0.39
unchanged      n=2356 longest median= 768 mean=1205 [155..8688]  MPx median=0.39
  perm-test longest edge  p=0.2649
  perm-test megapixels    p=0.4216
  perm-test aspect        p=0.9458
flip rate by longest edge:
  <=384  n= 114 changed= 3 rate=2.63%      641-1024 n=760 changed=9 rate=1.18%
  385-640 n= 736 changed= 3 rate=0.41%     >1024    n=764 changed=3 rate=0.39%
```

翻转题和未翻转题的最长边中位数一模一样(768 vs 768),三项置换检验全平,
分桶翻转率没有阈值形状,翻转从 178×80 一路铺到 4288×2848。**几何解释不了。**
(n=18 功率确实低,但连倾向都看不到。)

### 假设 B:成分效应 —— 否,见第三节

模型正确率 0.495,公平硬币对它就是对的零假设,校正后 p=0.106,没有改变
任何结论。

### 一个之前漏看的事实

11 个答案翻转里,yes→no 5 个、no→yes 6 个 —— **答案方向本身是平的**,偏的
只有"对→错"这一维。而这 11 题 pass1 恰好对了 8 题。也就是说这批题的翻前
正确率是 73%,而模型整体只有 49.5%。翻转**优先落在模型答对了的题上**,这
和第三节四个模型的规律(翻转落在 50/50 边界题上)不一致 —— 这是目前唯一
指向"不是普通噪声"的证据。

### 下一个控制实验(已排队,GPU 2)

pass2 相对 pass1 唯一还没被排除的差异是 **batch shape**:命中缩短 prefill,
请求批次组成随之改变,而 0.27.1 上批次形状本身就能推动一个 bf16 量子
(`../26/15_` 已用纯 vLLM 复现过)。

`run/control_batchshape.py`:**把 LMCache 整个摘掉**,纯 vLLM 跑两遍完整
MME,`max_num_seqs` 分别 256 和 32,按同一套 reg/imp 逻辑比。

- 若单靠 batch shape 就复现单向 → 这个偏属于模型 + benchmark,不属于读路径,
  DeepSeek-OCR 可以按 SUPPORTED 发,门也不用加跨跑记忆。
- 若纯 vLLM 是 50/50 → 偏确实来自 LMCache 读路径,必须查到底,不发证书。

**在这条结论出来之前,DeepSeek-OCR 保持 PROVISIONAL。**

## 五、9 张旧证书重发:是复验,不是记账

`1_` 第八节把这条列在"存量未动",理由是"0.23 上是绿的,重发是记账"。
今天重新看了一遍,**这个定性是错的**:

| 证书 | schema | runtime |
|---|---|---|
| qwen2-vl-2b / qwen2.5-vl-3b / gemma-4-e4b / mistral-small-3.1-24b | 8 | 0.27.1 |
| glm-4.6v-flash / internvl3.5-2b / qwen3-vl-2b | 2 | 无 runtime 字段 |
| qwen3.5-2b / qwen3.6-27b / qwen3.8-27b | 3 | 无 runtime 字段 |
| gemma-3-4b / molmo2-4b / qwen3-omni-30b | 只在 `../22/all12/` 归档里,树里没有 | |

这 9 张过的是**旧门**(没有方向门、没有答案/解析翻转拆分)、跑的是**旧
runtime**(0.23),而 PR 要为这些模型背书。重发是真复验。

按由小到大排的三条串行队列(每卡一条,跑完自动接下一个,`run/chain.sh`):

| GPU | 队列 |
|---|---|
| 2 | qwen3-vl-2b → **DeepSeek batch-shape 控制** → gemma-3-4b |
| 3 | internvl3.5-2b → molmo2-4b → qwen3.6-27b |
| 7 | qwen3.5-2b → glm-4.6v-flash → qwen3.8-27b |

qwen3-omni-30b(音频 + 30B)留到最后单独处理 —— 它的 MMAU 在 0.27.1 上
从没验过。

## 六、资源回收(用户当面问的)

用户问"有没有 task 或空闲服务忘了回收"。查下来:

- **确实漏了一个**:`apport`(pid 1082419)在啃 2026-08-26 20:27 那个
  python3.12 core dump,**跑了 10 小时 18 分,占满一个核、34 GB RSS**。
  发 TERM 时它 RSS 已经是 0,说不清是被杀的还是自己刚好收尾,总之已消失。
  它落下的 `/var/crash/_usr_bin_python3.12.1016.crash`(89 MB,属主 bo)
  还在共享目录里,**没动,等用户发话**。
- GPU 2 / 7 已确认 4 MiB,`../26/15_` 里那次杀错 pgid 漏下的 87.5 GB 已清干净。
- 跑完的 monitor(parity / certify)当场 TaskStop 掉了,没有留空转的 tail。

教训进第七节。

## 七、我这段犯的错

1. **提出了一个自己就能证伪的猜想,还先讲给了用户。** 第三节那个"退步多是
   因为翻前本来就对得多",在提出的当下就有五份逐题记录躺在盘上可以查,
   应该先算再说。代价是一轮无谓的往返 —— 虽然算完的结果比猜想本身有用。
2. **几何分析第一版取不到尺寸。** `dims()` 找的是 `image` / `image_path` /
   `img` / `path` / `file`,而 MME item 的 key 是 `image_uri`;脚本没报错,
   只是所有 `size=None`,打印出"no dimensions available"就跑完了。**一个
   全部取空却不报错的探针,比报错的探针更贵** —— 它看起来像是做过了。第二版
   加了显式的 `raise ValueError`。
3. **"重发 9 张旧证书是记账"这个定性写进了 `1_` 第八节。** 实际是跨大版本
   复验(0.23 → 0.27.1)加换门。定性错会直接改变优先级排序,这次是往低了排。

## 八、下一步

1. 三条队列跑完(约 3–6 小时),逐个签 schema 8 证书。
2. **DeepSeek batch-shape 控制的结论**,决定 PROVISIONAL 还是 SUPPORTED,
   也决定门要不要加跨跑记忆。
3. `benchmark_parity.py` 那处文档串改动待提交。
4. **等用户决定**:Tier 4 的 Llama 4 / Kimi-VL / Step-3 权重要不要下到
   `/raid/data/hub`(共享盘写入)。Kimi-VL 是唯一能补 MLA 覆盖的。
5. `/var/crash/_usr_bin_python3.12.1016.crash` 要不要删。
6. 存量:qwen3-omni-30b 的音频 / MMAU 在 0.27.1 上从没验过;PR 里提一嘴
   0.27.1 的 `defer_block_free` preemption 活锁。
