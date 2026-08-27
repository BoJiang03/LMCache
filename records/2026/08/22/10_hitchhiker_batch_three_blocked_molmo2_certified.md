# 搭车批(hitchhiker):四个里三个连引擎都起不来,第四个把套件的隔离不变量顶穿了

日期:2026-08-22
最终状态:**12/12 SUPPORTED**,全部落在 `0040c6bd`,证书在
`records/2026/08/22/all12/`

本次提交:

| commit | 内容 |
|---|---|
| `22f8a92e` | 注册 Molmo 2-4B;spec 级 `trust_remote_code` |
| `62dae0f1` | chunked prefill 的排除按模型属性路由,不再按 hybrid family |
| `a3c5ba42` | case 身份从文本 salt 挪进媒体本身;prompt 形状活体校验 |
| `624bc261` | Molmo 2 的 preemption 池尺寸(算出窗口再实测) |
| `0040c6bd` | template 在 processor 上时从 processor 读(差点弄坏 omni) |

## 一、这批的预期与实际

`records/2026/08/22/1_` §三写这批的价值是**样本量**:
"四个里有一个意外挂了比四个都过更有信息量"。

实际是**三个挂了**,而且三个都挂在 LMCache 之前 —— vLLM 0.23.0 的
processor 层,连引擎都起不来。第四个(Molmo 2)起来了,第一次跑套件
**28 项全红**,然后把套件里三处一直没被触发的假设逐个顶了出来。

产出:**一张新证书 + 三处套件修复 + 三份精确的阻塞原因 + 一个已定性的
上游 config 缺陷**,以及 11 张重新签发的证书。

## 二、三个阻塞,逐个的确切原因

全部在 8×H200 / venv `vllm-lazy`(vLLM 0.23.0,transformers 5.15.0)上实测。

### 1. DeepSeek-OCR(`deepseek-ai/DeepSeek-OCR`,6.7 GB)

架构 `DeepseekOCRForCausalLM`,**MLA + MoE**(`use_mla: true`、
`kv_lora_rank`、`n_routed_experts: 64`)—— 本来是这批里最有价值的一个,
它会是套件第一个 MLA 模型,KV 布局与已认证的 12 个全都不同
(LMCache 侧有 MLA 支持:`vllm_service_factory.py` 的
`mla_enabled` / `validate_mla_config`,`use_mla` 时 kv shape 第一维取 1)。

实测:带 `trust_remote_code` **能加载**(权重、MoE config、vision tower
都过),然后**第一次 generate 就死**,退出码 **136 = 128+8 = SIGFPE**。

- 走 chat template 的图片请求:进度条停在 0%,GPU 利用率 0%,CPU 19%,
  卡了 10 分钟以上。
- 绕开 chat template、用它自己的 `<image>\nFree OCR.` 原始 prompt:同样。
- **纯文本、完全不带图片**:同样 SIGFPE 退出(这一次没有任何人给它发信号)。

最后一条是关键:**与多模态无关,与 LMCache 无关**,是这个模型在这版 vLLM
上的解码路径本身就崩。日志最后一行永远是
`Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel`。
没有进一步定位(venv 里没有 pip,装不了 py-spy)。

### 2. Mistral Small 3.1(`mistralai/Mistral-Small-3.1-24B-Instruct-2503`,48 GB)

引擎初始化时死在 dummy mm 输入上,链式异常的真实原因是:

```
transformers/processing_utils.py:738 in prepare_inputs_layout
    images = self.image_processor.fetch_images(images)
AttributeError: 'MistralCommonImageProcessor' object has no attribute 'fetch_images'
```

transformers 5.15 的 `ProcessorMixin.__call__` 会走 `prepare_inputs_layout`,
它要求 image processor 有 `fetch_images`;vLLM 0.23.0 自带的
`MistralCommonImageProcessor`(`vllm/transformers_utils/processors/pixtral.py`)
没有这个方法。**上游已修**:vLLM main 上该类已经定义了 `fetch_images`
(注释写 "Copied from Transformers (Apache-2.0)")。

`--tokenizer-mode mistral` 不解决(试过,同样失败)。

> 更正一处我自己的误判:我最先用 `from_pretrained` 手工复现,撞到的是
> `MistralCommonPixtralProcessor.__init__` 里
> `tokenizer.transformers_tokenizer` 在 transformers 5.15 的
> `TokenizersBackend` 上不存在 —— 那是**另一条构造路径**的另一个错误,
> 不是引擎实际撞的那个。引擎是构造成功、**调用**时才失败。原因是我把
> `traceback.format_exc()` 截了尾部 4000 字符,而链式异常的原始 cause 打在
> **最前面**,正好被截掉。改成同时保留 head 才看到真因。
> **教训:截断 traceback 要留头,不是留尾。**

### 3. MiniCPM-V 4.6(`openbmb/MiniCPM-V-4.6`,2.6 GB)

先说一个意外发现:**它是 recurrent-state hybrid**。config 里
`text_config.model_type = qwen3_5_text`,`layer_types` 是
3×`linear_attention` : 1×`full_attention` 的 24 层 —— 建在 Qwen3.5 文本塔上。
如果能跑,它会是第四个 GDN 系 hybrid,而且只有 2.6 GB,是所有 hybrid 里
最便宜的一个(对 `escalations/1_block_pool_cache_full_blocks_crash.md`
那个崩溃来说,一个便宜的复现器很有用)。

实测阻塞:

```
vllm/model_executor/models/minicpmv.py:545  MiniCPMVProcessingInfo.get_hf_processor
    vendored_processor = MiniCPMVProcessor(
        image_processor=hf_processor.image_processor,
        tokenizer=hf_processor.tokenizer,
    )
vllm/transformers_utils/processors/minicpmv.py:61
    self.version = image_processor.version
AttributeError: 'MiniCPMV4_6ImageProcessor' object has no attribute 'version'
```

`MiniCPMV4_6ProcessingInfo` 继承了 4.5 时代的 `get_hf_processor`,没有
override;transformers 5.15 的 `MiniCPMV4_6ImageProcessor` 不带 `version`
字段(repo 的 `preprocessor_config.json` 里也没有)。**上游已修**:main 上
签名变成 `MiniCPMVProcessor(..., version=self.get_model_version())`,
version 从 config 推。

### 小结:vLLM 升级现在背着三个模型

| 模型 | 阻塞层 | main 上是否已修 |
|---|---|---|
| Mistral Small 3.1 | vLLM 自带 pixtral image processor 缺 `fetch_images` | **已修** |
| MiniCPM-V 4.6 | vLLM 自带 minicpmv processor 读 `image_processor.version` | **已修** |
| DeepSeek-OCR | 首次解码 SIGFPE(纯文本也崩) | 未知,需升级后重测 |

两个"已修"是同一族:**vLLM 0.23.0 自带的 processor shim 没实现
transformers 5.15 要求的接口**。这把"升级 venv 的 vLLM"从
"Qwen3.8 的前置"抬成了**背着三个模型**的一步。

## 三、Molmo 2-4B:SUPPORTED,以及它顶出来的三处套件缺陷

`allenai/Molmo2-4B`,19 GB,`Molmo2ForConditionalGeneration`。
最终 26 项全过,0 失败/错误/跳过,MME gate 通过。

引擎侧实测(活引擎上读,不是从 config 推):

- **1 个 KV cache group**(`FullAttentionSpec`,block 16,36 层,
  64 KiB/page)→ **144 KB/token**,是套件里所有 in-process 模型中最宽的
  KV。不是 hybrid,走 in-process 路径。
- 1540×1540 的图片 → prompt 一共 **770 token**,已经落在别的 spec 要靠
  pixel cap 才能压到的 ~768 image token 预算里,所以它**不需要**
  `mme_mm_processor_kwargs`。
- 全量 MME 2374 题 × ~770 token × 144 KB ≈ **263 GB** KV,40 GB 默认容量
  会在 pass-2 复访前把每一条都淘汰掉、hit gate 在 ~0 失败,所以
  `mme_max_local_cpu_gb=340.0`。
- config 需要 `trust_remote_code`(带 `auto_map`,transformers 5.15 直接
  拒读),尽管 vLLM 原生实现了 `molmo2.py`、根本不跑 repo 自己的建模代码。

MME parity(全量 2374 题,in-process):

| | |
|---|---|
| flips | 2(pass1 vs baseline)、1(pass2 vs pass1),预算 11.87 |
| 分数 | 1894.95 / 1895.95 / 1895.20,delta 1.0 与 0.75,预算 10 |
| lookup hit ratio | 1.0 |
| baseline parse ratio | 1.0(2374 条全部以 yes/no 开头,8 token 预算够) |
| coverage | `null` —— in-process 没有分母,今早那个修复正常生效 |

### 缺陷 1:套件假设每个模型都接受 system role

Molmo 2 的 chat template 对 system message 抛
`jinja2 TemplateError: Conversation roles must alternate user/assistant/...`。
套件每个请求都带一条 system message —— 而它不是装饰:**per-case salt 在里面**。

修法:`supports_system_role=False` 时把同一段文本(含 salt)折进 user
message 最前面。**但这一步并没有达到它的目的**,见缺陷 3。

### 缺陷 2:chunked prefill 的排除按 hybrid family 路由,而它不是 family 的属性

`isolated_cases.py chunked_prefill` 把 `max_num_batched_tokens` 钉在 128。
Molmo 2 上引擎直接拒启:

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (8134)
is larger than max_num_batched_tokens (128).
```

`vllm/platforms/cuda.py:241` 对 `model_config.is_mm_prefix_lm`(**图像 span
双向注意力**)的模型强制 `disable_chunked_mm_input = True`,
`encoder_cache_manager.py:299` 随后在 budget 低于模型最坏 mm item 时拒启。
于是 chunked prefill 对这一类模型**结构上不可能**,和 align 模式对
recurrent-state hybrid 是同一个形状:**小到能切开 image span 的 budget 会
让引擎起不来,大到能起来的 budget 切不开 span。**

而套件把这个排除**挂在 hybrid family 上**:

- `isolated_routing.isolated_scenarios`:`family is NONE` 就跑 chunked prefill。
- `certify.known_not_covered`:非 hybrid 直接 early return,chunked prefill
  的排除文本写在 return **之后** —— Molmo 2 这种"非 hybrid 但被排除"的模型,
  证书会**悄悄漏掉一条它确实有的限制**。
- `scope.scheduling` 是同一缺陷的镜像:`_PREFILL_REGIME[NONE]` 写死
  "single-step and chunked prefill",这只在"每个单组模型都跑了这个场景"
  的前提下为真。

三处现在都读 `isolated_scenarios`。这和 08-22 上午那次证书修复是**同一族
缺陷的对偶**:那次是"断言了没验证的事",这次是"漏掉了确实存在的限制"。
省掉一条真限制,读起来和"没有这条限制"一模一样。

### 缺陷 3(最重要的一条):case 隔离不变量依赖 chat template 把文本排在媒体之前

第一次跑完套件是 16 失败 / 8 通过,其中 15 个是 T0.4 的 phase 1–15:

```
assert b1.lookup_hits(752) <= a2.lookup_hits(784) - image_span_margin(64)
```

phase 0 **通过**,1–15 全挂。测量给出了确切答案:

| 比较 | 共享的原始 token 前缀 |
|---|---|
| `t01-A` vs `t01-B`(两张不同的图) | **787 / 787**(全同) |
| `t04-p0-B` vs `t04-p1-B`(同图,不同 phase) | **762** |
| `t12-A` vs `t12-A-q2`(同图,不同问题) | 767 |
| `t22-A` vs `t22-AC`(加一张图) | **1** |

762 // 16 = 47 个整 chunk = **752** —— 正好是观测到的命中数。所以
phase p 的 B 请求复用的是 **phase p-1 的 B**(同一张图),不是 A。
**没有串图**,是同一张图的合法复用;phase 0 通过只因为它跑在最前面。

根因:Molmo 2 的模板把 `<|image|>` 提到 `<|im_start|>` **之前**,于是
**per-case salt 不再位于 prompt 开头**,image span 成了前 ~750 个 token
且跨 case 字节相同。缺陷 1 的折叠救不了它 —— **内容顺序不归调用方管**。

修法:**case 身份挪进媒体本身**。`catalog.case_media_bits` 把 salt 的 CRC
按和 index 相同的方式混进合成内容(图片/视频帧多一块角落图案,音频并进
dither 种子),三个模态都覆盖。case 内不变、case 间不同。text-first 模型下
返回 0,字节与 11 个已认证模型此前测过的**完全一致**(已验证)。

并且加了一道活体校验 `MMHarness._validate_prompt_shape`:渲染同一个探针
请求两遍(带图/去图),diff 出媒体标记位置再与 salt 位置比,与 spec 声明
不符就 `RuntimeError`。抄的是 `_validate_block_size` 的模式 —— 用模型属性
而不是模型名单路由,**声明错了两个方向都会响**。

改完:16 失败 → **1 失败**,而且 MP 路径上 T1.2 那个 0 命中也一起消失了。

### 剩下的两件,都是"窗口/前提"而不是缺陷

- **T2.2 partial sharing** 结构上不适用:`t22-A` 与 `t22-AC` 只共享 1 个
  token —— Molmo 2 的 processor 排的是整个图像**集合**的布局,不是逐项追加。
  按 chunked prefill 的同一套路做成 `media_prefix_stable=False` 的 deselect
  + 证书排除条目,in-process 和 MP 两条路径都盖到。
- **preemption 空转**(0 次抢占)。原因不是 KV 宽度而是 **prompt 的 token
  数**:池子按 block 计,所有模型都是 2048 token,而 Molmo 2 一个图像请求
  787 token = 50 block,是 Qwen 系的十倍左右,128 block 下只有两个请求能
  同时驻留,decode 增长(每个 7 block)永远撑不破。窗口
  = [6×50 = 300, 6×57 = 342),取 320,实测 1 次抢占、0 失败。

## 四、顺带定性的一个上游 config 缺陷:Gemma 4-E4B 的图像 span 在按因果跑

用 `create_model_config()` 把 12 个注册模型的 `is_mm_prefix_lm` 全测了一遍
—— 只有 **molmo2-4b** 和 **gemma-3-4b** 是 True,**gemma-4-e4b 是 False**。

Gemma 4 当初被选进来有一半原因就是验"图像 token 双向 mask"
(vLLM #40106,见 `records/2026/08/21/2_`)。往下查到确切原因:

`Gemma4ModelArchConfigConvertor.is_mm_prefix_lm` 只在
`text_config.use_bidirectional_attention == "vision"` 时返回 True。而:

| checkpoint | `text_config.use_bidirectional_attention` |
|---|---|
| `google/gemma-4-E4B-it` | **`null`** |
| `gg-hf-am/gemma-4-E4B-it-assistant` | **`null`** |
| `google/gemma-4-12B-it` | `"vision"` |
| `google/gemma-4-31B-it` | `"vision"` |

也就是说**不是"Gemma 4 不用双向"**,而是 **E4B 这个 checkpoint 的
config 把这个字段显式写成了 null,而它的两个同代兄弟写的是 "vision"**。
后果:vLLM 对 E4B 的 image span 按**因果**跑,对 12B/31B 按双向跑。
vLLM 自己 `arg_utils.py:2531` 的注释里还把 Gemma 4 当作 prefix-LM 的
典型例子。

这与 LMCache 无关(parity 是同配置对比,证书照样成立),但它正是 #40106
描述的那种"被静默忽略"。已写进 gemma-4-e4b 的 spec 注释。**值得单独上报
上游(google 的 config 还是 vLLM 的默认)。**

顺带:gemma-3-4b 是 True,它的 chunked prefill 本来就被 hybrid 那条排除了,
场景集合不变,但证书现在写**两条独立原因**而不是一条 —— 这就是重跑 11 个
模型的直接理由之一。

## 五、结果:12/12 SUPPORTED @ `0040c6bd`

证书在 `records/2026/08/22/all12/`(取代 `recert/`)。全部
`schema_version: 4`、`tested_tree.stable: true`、0 失败/错误/跳过、
parity gate 通过。11 个复用已记录的 parity 报告(`--parity-report` 会用
当前代码重新评 gate,不是照抄旧结论),Molmo 2 用今天新跑的。

覆盖到的架构维度:纯 attention、DeepStack、两个 hybrid family、
in-process 与 MP 双路径、image/video/audio 三模态加 cross-modal 配对,
现在再加 **mm-prefix-LM(图像 span 双向注意力)** 与 **media-first 模板**。

**一个非确定性失败,记下来而不是抹掉。** `qwen2-vl-2b` 这一轮第一次跑挂在
`test_t0_chunk_boundary_phases[1]`:

```
t04-p1-B: LMCache reported 20928 hit tokens but vLLM only skipped 16
on the connector's account (0 locally cached)
```

几百 token 的 prompt 报 20928 次命中不是缓存行为,是计数器 delta 捞进了
窗口外的东西。重跑 29/29 通过,同一检查在本轮另外 11 个模型和今早 11 个
模型上都通过。所以按 stats 路径的 flake 归档 —— **但归档了**,因为一个会
偶发出错的计数器,也可能在一次没人重跑的运行里出错。

## 六、我自己犯的三个错

1. **scratch 文件名遮蔽了 setuptools 要 import 的模块。** 下载脚本命名为
   `dl.py`,放在 probe 脚本同目录(即 `sys.path[0]`);
   `setuptools/command/build_ext.py:72` 有一行 `import dl`,于是 vLLM 的
   架构 inspect 子进程 import 到了**我的脚本**,而它在 import 时就读
   `sys.argv[1]` → `IndexError`。表现是三个完全不同的架构报**同一句**
   "failed to be inspected",看起来像三个模型都不被支持。改名即好。
   **教训:会成为 `sys.path[0]` 的目录里,别用短的通用名给脚本命名。**
2. **又一次把 `rm` 和后台启动写进同一条复合命令。** 第一行
   `cd X && ls && rm -f probe_*.log && nohup A &`,整条被 `&` 放到后台,
   它的 `rm` 在第二行启动的 B 之后才执行,把 B 刚建的日志删了。这和
   `parallel-certification-is-safe` 里记的 `cd X && VAR=y && nohup ... &`
   是同一个坑的另一面。
3. **新加的校验差点弄坏一个已认证模型。** `_validate_prompt_shape` 用
   `tokenizer.apply_chat_template`,而 **Qwen3-Omni 的 chat template 挂在
   processor 上,tokenizer 上是 None** —— 校验会在 harness 启动时对它直接
   抛 `RuntimeError`。在上车前把校验逻辑离线跑了一遍全部 12 个模型才发现。
   **教训:一个"帮忙检查别人"的机制,自身必须先在全部现有对象上跑通;
   而且它不能有能力弄坏一个它本来无话可说的模型。**

## 七、下一步

1. **升级 venv 的 vLLM** —— 现在背着 Mistral Small 3.1、MiniCPM-V 4.6
   两个"已在 main 修好"的模型,加 DeepSeek-OCR 的重测,再加原有的
   Qwen3.8。这已经是性价比最高的一步。
2. **上报 Gemma 4-E4B 的 `use_bidirectional_attention: null`**(见 §四)。
3. 模型顺序:Llama 4 / Kimi-VL / Step-3 → Phi-4-multimodal(实质是
   `extra_keys` 重构)。Whisper 明确排除。
4. `escalations/` 下两份仍待你提交。
5. 仍未推送:本地已排队 14 个提交,等明确指令。
