# 三张 schema-8 证书落地,清单往下推两个模型,以及 DeepSeek-OCR 读路径上一个重复出现的单向掉分

日期:2026-08-27(接 `../26/15_`,跨零点连续会话)
代码状态:`multi_modal@39b10ec3`,工作树干净。本篇不新增提交,以下四个提交
是本段会话的全部产出:

| commit | 内容 |
|---|---|
| `888e23a2` | qwen2.5-vl-3b 的 MME 翻转预算放宽到 1.5% |
| `5d6e6d93` | 注册 DeepSeek-OCR;`ModelSpec.chat_template` 及五处穿线 |
| `a8a174c9` | 注册 Mistral Small 3.1 24B |
| `39b10ec3` | Mistral 的 `isolated_gpu_utilization` 抬到 0.75 |

本篇的重点在第四节。前三节是把 `../26/15_` 那批工作收尾,第四节是新发现的、
**至今唯一一个方向可疑的模型**。

## 一、头三张 0.27.1 + MP 上的证书

`../26/15_` 里发出去的三跑全部收工,证书已签发:

| 模型 | 套件 | 翻转 (pass2 vs pass1) | 退/进 | 一侧 p | 预算 | 结果 |
|---|---|---|---|---|---|---|
| qwen2-vl-2b | 46 项 0 失败 | 18 | 10 / 8 | 0.407 | 23.74 (1.0%) | SUPPORTED |
| gemma-4-e4b | 46 项 0 失败 | 1(另 15 parse) | 0 / 1 | 1.0 | 11.87 (默认) | SUPPORTED |
| qwen2.5-vl-3b | 46 项 0 失败 | 24 | 13 / 11 | 0.419 | 11.87 → **35.61** | SUPPORTED |

三张都是 `schema_version: 8`、`tested_tree.stable: true`、
`runtime.vllm.version: 0.27.1`、路径 `LMCacheMPConnector + MP cache server`。
**这是头三个在 0.27.1 + MP 上拿到的证书。**

两件值得单独记的:

**gemma 那一跑证实了 answer/parse 拆分是承重的。** 它 pass2-vs-pass1 合计
16 次翻转,其中 **15 次是 parse 翻转,真答案翻转只有 1 次**,而且是变对。
gemma parse ratio 只有 0.896(它话多,常不吐干净 yes/no),那 15 次是"没解析
出来"的题在动。拆分之前这跑会按 16 撞 11.87 的预算**直接判红**。

**qwen2.5-vl-3b 首跑判红,红在数量不在方向。** 24 次翻转 = 1.011%,超默认
0.5% 预算,但 13 退 11 进、p=0.419,和另外两个一样干净的双侧;parse 翻转 0,
parse ratio 三段全 1.0,pass1-vs-baseline 0 翻转。用户定 0.015。注意
**照搬 qwen2-vl-2b 的 0.01 修不好**(23.74 < 24),而"看见红了把线抬到刚好够"
正是要避免的做法,所以 spec 注释里写的是"预算坐在实测值之上,不是坐在它上面",
并把方向门在 36 次翻转时需要 26 次同向(72%)这个数写进去,让买来的余量有价钱。

三次真实数据也把 p=0.01 的标定验了:0.407 / 1.0 / 0.419,全离阈值 40 倍以上。
门没有卡到噪声。

## 二、模型清单在哪、以及一条被写错的前提

用户指出"model list"是更早之前定的顺序表,要翻 records 才有。在
`../22/1_` §五(按覆盖维度排,不按热度排),`../25/1_` 更新过实测状态:

| # | 模型 | 0.27.1 上的状态 |
|---|---|---|
| 1 | **DeepSeek-OCR** | 已解锁(0.23 上首次解码 SIGFPE) |
| 2 | **Mistral Small 3.1 24B** | 已解锁(0.23 上 pixtral processor 缺 `fetch_images`) |
| 3 | MiniCPM-V 4.6 | **仍挂**,阻塞从 processor 挪到权重装载(`k_proj` mapper 没被应用) |
| 4 | Llama 4 / Kimi-VL / Step-3 | 未动,权重也不在 `/raid/data/hub` 里 |
| 5 | Phi-4-multimodal | 实质是 `extra_keys` 接口重构 |
| 6 | Kimi K3 / ERNIE-4.5-VL 单独立项;Whisper 明确不做 | |

**`../22/10_` §二写的"DeepSeek-OCR 是 MLA + MoE(`use_mla: true`、
`kv_lora_rank`)"是错的**,并据此把它排成这批最值钱的一个、"套件第一个 MLA
模型"。直接读 checkpoint(snapshot `9f30c71f`)的 config:

```
use_mla = False          (顶层与 language_config 都是)
kv_lora_rank = None      q_lora_rank = None
qk_rope_head_dim = 0     v_head_dim = 0
```

活引擎复核一致:`ModelConfig.use_mla` False,KV spec 上 `use_mla` 也是 False。
和 Gemma-4 `use_bidirectional_attention: null` 那次是**同一类错误** ——
按架构族(DeepseekV2)推断,没读这个 checkpoint 自己的 config。

它实际带来的新维度:**套件第一个 MoE**(64 routed + 2 shared)、**第一个 MHA**
(10 个 q head、10 个 kv head,已认证的 12 个全是 GQA)、**第一个 tiled 视觉塔**。
**MLA 这个维度目前没有任何在手候选**,证书里"未覆盖 MLA"会继续挂着;真正的
MLA 候选是清单第 4 档的 Kimi-VL(DeepSeek-V3 架构)。

## 三、DeepSeek-OCR:没有 chat template,以及为什么一个 spec 字段就够

**实测:processor 和 tokenizer 上都没有 chat template**,repo 里也没有任何
template 文件。它的对话格式以 Python 形式活在 `conversation.py` 里
(`sft_format = deepseek`,roles `<|User|>`/`<|Assistant|>`,sep `\n\n`)。

我最初判断这会牵扯 harness / catalog / baseline_runner 三处、要长一条
raw-prompt 分支。**这个判断是错的**,四种提法逐个实测之后:

| 提法 | 合成红图 | MME 式 yes/no | 能否用于 MME |
|---|---|---|---|
| 原始 `<image>\n{问题}` | `Red` | `Yes` / `No` | 能 |
| 它自己的 `<\|User\|>:` / `<\|Assistant\|>:` SFT 格式 | `1` | "The image is a vibrant blue color," | **不能**,改成描述图片 |
| 模板不发图像标记,让 vLLM 自己替换 | — | — | 不能,`AssertionError: Failed to apply prompt replacement` |
| **手写模板,只发图像标记和文本、不发角色标记** | `Red` | `Yes` | **能** |

最后一行的关键是:该模板经 `llm.chat` 渲染出的 **token id 序列与原始 prompt
逐个相同**(比的是完整 id 列表,不是长度),salt 走 system message 时多 9 个
token、答案不变。所以套件不需要第二条提示路径,只需要一个新的 spec 字段。

**一个承重的细节:模板里 `<image>` 后面那个换行不能被吃掉。** 第一版写成字面量
`<image>\n`,被相邻标签的 `{%- -%}` 空白控制吃掉一个 token(286 → 285),模型
就从 `Yes` 变成先把问题复述一遍(`Is the image blue?Yes, the`)。改成表达式
`{{ '<image>\n' }}` 才稳。**一个 token 的差别改变了模型行为**,这条值得记。

实现:`ModelSpec.chat_template`,按 `chat_template_kwargs` 的同一条路径穿到
每个引擎(harness 两个 chat 调用 + baseline spec dict、baseline_runner、
benchmark_parity 的 CLI/`run_batch`/自举 baseline 子进程 argv、
certify 的 `parity_command`)。`_validate_prompt_shape` 也改成**优先用 spec 的
模板** —— 它要校验的是这次跑真正发出去的 prompt 形状,而不是模型自带的。

实测几何(活引擎读):1 个 KV group、FullAttentionSpec、block 16、12 层、
80 KiB/page → **60 KB/token**;640×480 照片 283 token、1540×1540 是 703,
**不需要 pixel cap**;两图 prompt 与单图共享 275/276 token,`media_prefix_stable`
成立。全量 MME 的 KV 在 40–100 GB 之间,给 120 GB。

**套件首跑 45 项全过,0 失败**,四个调度场景全覆盖、无 exclusion。但这个"全绿"
有一半是我给的:Molmo 2 当初首跑 28 项全红,是因为它的模板把图像提到 salt 前面
且不收 system role;而 DeepSeek-OCR **没有模板,模板是我写的**,我自然写成
salt 在前、图像在后。套件的三条前提是被按满足的方式构造出来的,不是模型天然
满足。证书覆盖的是"用这个模板驱动的 DeepSeek-OCR",不是它的官方用法。

## 四、DeepSeek-OCR 读路径上一个重复出现的单向掉分(未结)

两次独立的全量 MME parity:

| | 翻转 | 退 / 进 | 一侧 p | baseline | pass1 | pass2 |
|---|---|---|---|---|---|---|
| run1 | 9 | **8 / 1** | 0.0195 | 835.51 | 835.51 | **817.28**(−18.23) |
| run2 | 11 | **8 / 3** | 0.1133 | 835.51 | 835.51 | **821.57**(−13.94) |
| 合并 | 20 | **16 / 4** | **0.00591** | | | |

对照另外三个模型:0.407 / 1.0 / 0.419。**这个形状是独一份的。**

**单看每一跑都合法通过门**(0.0195 和 0.1133 都在 0.01 之上,翻转数也都在
11.87 预算内),**合起来越线**。

三条把范围收窄的硬事实:

1. **baseline 与 pass1 总分完全相同(835.51),0 翻转、0 parse 翻转** ——
   写入路径逐字一致,差异只可能来自**读**。
2. **两次掉分同向**(−18.23、−13.94),不是随机游走;命中率两次 0.9888/0.9889,
   parse ratio 两次都是 0.9486,parse 翻转两次都是 7 —— 各项都高度可重复。
3. **但合成套件 45 项全过,包括逐字节重放的 oracle。**

要同时解释第 3 条和前两条,当前假设是**分块(tiling)**:DeepSeek-OCR 是套件
里第一个 tiled 视觉塔(`candidate_resolutions: [[1024,1024]]`、`tile_tag: 2D`、
`global_view_pos: head`),**套件的合成图全是同一尺寸,只产生一种分块布局**;
MME 照片尺寸各异,会产生套件从未生成过的 prompt 布局。一个依赖分块的缺陷正好
能同时满足"套件全绿"与"MME 单向掉分"。

**处置:DeepSeek-OCR 保持 PROVISIONAL,不发 SUPPORTED。**

正在做的零成本验证:用已存的逐题答案(`parity_deepseek-ocr.answers.json`)
把翻转题与未翻转题按图像几何对比,看翻转是否集中在特定尺寸上。这一步不需要
再跑引擎。**本篇写完时该分析仍在跑**(MME 1097 张图的装载本身要几分钟)。

另一条独立的解释方向没有排除:这个模型 MME 总分只有 835(qwen2.5-vl-3b 是
2208),parse ratio 0.9486 —— 它是 OCR 模型不是 VQA 模型。若正确答案是个小
目标,随机扰动"从对变错"的机会多于"从错变对";但按 ~55% 的正确率算,期望的
退/进比约 1.2,离实测的 4.0 还很远,所以这条解释不够。

## 五、Mistral Small 3.1:路由决定了哪个 processor,以及 24B 的显存墙

**实测几何**:`PixtralForConditionalGeneration`、1 个 KV group、block 16、
40 层、64 KiB/page → **160 KB/token,套件里最宽的 KV**(此前是 Molmo 2 的
144 KB)。chat template 在 processor 上,收 system role,合成图答 `Red`。

**pixel cap 加不上,但也不需要 —— 这是测出来的,不是省略。** 第一版用同一个
进程连起六个引擎扫 `size.longest_edge`,每档报出完全相同的 token 数,我据此
猜"vLLM 按进程缓存 processor,那六个数字不是六次测量"。**这个猜测是错的**:
改成一个 cap 一个进程重测,default / 756 / 512 三档**仍然完全相同**
(640×480 → 446,1540×1540 → 3094)。

真正原因:这个 repo 同时带 `consolidated.safetensors` + `params.json`,
**vLLM 认 mistral 格式优先于 `config.json`**,把架构从 config 里的
`Mistral3ForConditionalGeneration` 解析成 `PixtralForConditionalGeneration`
(`model_type: pixtral`),走 `pixtral.py` + `MistralCommonPixtralProcessor` ——
它从 `params.json` 的 `vision_encoder.max_image_size` 取尺寸,**不看 HF 的
`size` kwarg**。离线直接调 HF 的 `PixtralImageProcessor` 是认的
(756 档把 1540² 从 3082 压到 758),两条路的 processor 根本不是同一个。
`config_format="hf"` 能把它掰回 `Mistral3ForConditionalGeneration`,但那是
另一条代码路径,换了等于证另一个东西。

结论是不需要 cap:模型自己的 1540 像素上限已经把最坏照片压到 3094 token,
在 8192 上下文里放得下。`mme_max_local_cpu_gb=260`(2374 × ~446 × 160 KB
≈ 175 GB,留余量;箱子 2 TB RAM,1.4 TB 可用)。

**首跑 NOT_SUPPORTED,43 过 2 挂**,两个挂点同一个原因:

```
ValueError: No available memory for the cache blocks
```

挂在 `capacity_eviction` 和 `mp_connector` 两个隔离场景 —— 它们各起独立引擎,
用 `isolated_gpu_utilization` 的 0.35 默认 = 这张卡上 50 GB,而 24B 的 bf16
权重就是 48 GB,一个 KV block 都放不下。spec 字段文档里早写着这个坑
("27B in bf16 needs 0.37 before a single KV block"),三个 27B 都设了 0.75。
抬到 0.75 后**套件 45 项全过,PROVISIONAL**,四个调度场景全覆盖。全量 parity
在跑。

## 六、方向门的一个缺口:没有跨跑记忆

第四节直接暴露的:**两跑各自合法通过,合并后越线**。门是按单跑评的,
`flip_asymmetry_p` 只看这一份报告里的退/进。

不急着改。要改的话有两个方向,各有代价:

- **证书累计同一模型的历史 parity 报告再评方向** —— 能抓到这种情况,但引入
  "证书依赖历史"这一新性质,而当前证书的卖点正是自足(一份报告 + 一棵树 +
  一次套件)。
- **要求同模型两跑并合并评判** —— 诚实但把每个模型的 parity 成本翻倍。

在 DeepSeek-OCR 这条查清楚之前不动门:现在改,等于在不知道信号真假的情况下
调灵敏度。

## 七、我这段犯的错

1. **杀 A 用了启动器的 pgid,而它已经 exec 掉了。** `ps -o pgid= -p <启动器pid>`
   返回空,`kill` 被跳过,我却报告了"已 drop"。下一条消息里才发现进程还活着、
   GPU 还占着 87 GB。**教训:要杀的是活着的那个 pid 的 pgid,不是我记下的启动
   时的 pid。**
2. **探针的 TMPDIR 用了 scratchpad 那条长路径**,vLLM 的 engine-core IPC socket
   撞 `sockaddr_un` 的 107 字符上限(那条路径本身就 ~105)。parity/certify 没
   踩到是因为它们走 `configure_environment()` 关掉了 V1 多进程,压根不建那个
   socket。**教训:凡是自己起引擎的脚本,TMPDIR 要短。**
3. **探针读 `engine_core.engine_core` 前没设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`**,
   拿到 `SyncMPClient` 直接 AttributeError。
4. **六个引擎塞进一个进程扫 cap** —— 第三个就 OOM(`del llm` 并没把显存还回去),
   而且让我对着六个相同的数字编了一个错误解释(见第五节)。**教训:一个配置
   一个进程,不要在同一进程里连起引擎。**
5. **三张证书 23:24 跑完,我 23:53 才发现** —— 后台 monitor 盯的是 parity 日志,
   没把 certify 日志加进去,白等半小时。后面每个长跑都挂了 monitor。
6. **高估了 chat template 的工作量**(说要动三处、要长 raw-prompt 分支),
   实际一个 spec 字段。**教训:先测四种提法再估工作量,不要先估。**

## 八、下一步(顺序即优先级)

1. **DeepSeek-OCR 的单向掉分定性** —— 先看几何相关性分析的结果;若翻转确实
   集中在某类尺寸上,就用固定尺寸的 MME 子集重跑做对照。这条不结,它不发
   SUPPORTED。
2. **Mistral Small 3.1 的 parity 收工后签证书。**
3. MiniCPM-V 4.6:阻塞在上游权重装载,不是我们能配置绕开的,跳过。
4. 清单第 4 档(Llama 4 / Kimi-VL / Step-3)的权重都不在 `/raid/data/hub`,
   开始之前要先定下载哪个 —— Kimi-VL 是唯一能补上 MLA 覆盖的候选。
5. 存量未动:9 张旧证书重发 schema 8(0.23 上是绿的,重发是记账不是覆盖,
   已按用户意见降级到新覆盖之后)、方向门的跨跑记忆、PR 里提一嘴 0.27.1 的
   preemption 活锁。
