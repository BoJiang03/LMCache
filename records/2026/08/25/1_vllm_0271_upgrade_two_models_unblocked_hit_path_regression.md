# 升级 vLLM 0.23.0 → 0.27.1:两个模型解锁,但已认证的套件在新版本上塌了

日期:2026-08-25
分支:`multi_modal` @ `0040c6bd`(工作树全程干净,本次**没有改动任何仓库代码**)
接 [`../22/10_`](../22/10_hitchhiker_batch_three_blocked_molmo2_certified.md) 第七节
第 1 项「升级 venv 的 vLLM」。

## 结论先写

1. **升级本身完成,目的达到了一半以上。** 新建 venv `/home/bo/venvs/vllm-mm`
   跑 vLLM **0.27.1**(PyPI 最新)。0.23.0 背着的三个模型里,
   **DeepSeek-OCR 与 Mistral Small 3.1 解锁**,MiniCPM-V 4.6 仍挂,但挂的位置
   比以前深。
2. **代价是:12 张证书在 0.27.1 上不成立。** 同一个提交、同一棵 lmcache 树、
   同一个 harness,`qwen3.5-2b` 从 27 passed 变成 4 failed,
   `qwen2-vl-2b` 变成 **27 failed / 2 passed**,失败形态是命中路径吐空串和乱码。
3. 所以 **`vllm-mm` 目前不能用来认证**。刚解锁的两个模型也就还认证不了。
   升级这一步买到了两个模型,同时买到了一个必须先弄清楚的回归。
4. **回归与多模态无关,也与本分支无关。** 一个纯文本模型(Qwen3-0.6B,路径上
   没有一行 mm 代码)在 0.27.1 上命中准确率 **0/16**,同一个探针在 0.23.0 上
   **15/16**,两边装载的 token 数一样。把 lmcache 换成 `dev@09bc14c0` 重跑,
   数字和逐字输出都一模一样;换成**今天的上游 HEAD** `c1ef01b9` 并重新完整
   编译 native 扩展,还是一样。坏的是 base 在 vLLM 0.27.1 上的通用 KV 装载
   路径,mm 套件只是最先撞上它的东西。
5. **根因定位到 KV cache 内存布局(第六节)。** vLLM 0.27 把 K/V 融进最后一维、
   head 轴前移、张量不再连续,而 `kv_cache_layout` 两版都仍报 `NHD`。

证据在 `vllm_upgrade/`。

---

## 一、环境:为什么是新建 venv,而不是就地升级

`vllm-lazy` 的 editable lmcache 指向的是
`/home/bo/LMCache-worktrees/lazy_offloading`,不是本工作树 —— 就地升级会一起落到
那条在跑的分支上。所以新建,两个老 venv 一行没动。

| | vllm-lazy(认证用) | vllm-mm(新) |
|---|---|---|
| vLLM | 0.23.0 | **0.27.1** |
| torch | 2.11.0 | 2.13.0 |
| torchvision | 0.26.0 | 0.28.0 |
| transformers | 5.15.0 | 5.15.1 |
| numpy | 2.2.6 | 2.2.6(装完手工对齐) |

otel 一族被 lmcache 的 `<= 1.40.0` 上限拽回 1.40.0,和 vllm-lazy 一致;
`pip check` 干净。

**lmcache 的解析没有用 editable install** —— `pip install -e .` 在这个 venv 里
建不起来 C 扩展(缺 `Python.h`,没有 python3.12-dev)。改成往 site-packages 放
一个 `_lmcache_multi_modal_worktree.pth` 指向本工作树,效果一样且不需要编译。
套件本来也不依赖安装:conftest 把 repo root 插进 `sys.path`,harness 给子进程
设 `PYTHONPATH`。

**HF 缓存约定照旧:每次跑都带 `HF_HUB_CACHE=/raid/data/hub`。**

## 二、三个阻塞模型的实测(全部在 0.27.1 上)

| 模型 | 0.23.0 上的阻塞 | 0.27.1 实测 |
|---|---|---|
| **DeepSeek-OCR** | 首次解码 SIGFPE(纯文本也崩) | **正常解码**,解锁 |
| **Mistral Small 3.1** | 引擎初始化崩(pixtral 缺 `fetch_images`) | **引擎起来 + 颜色探针答对 `Red`**,解锁 |
| MiniCPM-V 4.6 | processor 读 `image_processor.version` | processor 已修,**改挂在权重装载** |

DeepSeek-OCR 是这批里最值钱的:MLA + MoE,KV 布局和已认证的 12 个全都不同,
会是套件第一个 MLA 模型。

MiniCPM-V 4.6 的新阻塞:

```
ValueError: There is no module or parameter named 'k_proj' in
MiniCPMV4_6ViTWindowAttentionSelfAttn. The available parameters are:
{'qkv_proj.bias', 'out_proj.weight', 'out_proj.bias', 'qkv_proj.weight'}
```

`minicpmv4_6.py:651` 那个类**自带** `hf_to_vllm_mapper`
(`.q_proj/.k_proj/.v_proj → .qkv_proj`)和自己的 `load_weights`,但装载这批权重
的路径上 mapper 没被应用。这是上游的权重装载缺陷,不是 config 能绕的。

## 三、静态兼容性勘察(0.23.0 vs 0.27.1 逐个符号对)

先做了一遍不花 GPU 的勘察,结论是 spec 里那些实测常数**应该**能平移:

| 面 | 0.23.0 | 0.27.1 | 影响 |
|---|---|---|---|
| `MultiModalFeatureSpec.identifier` / `.mm_position` | ✓ | ✓ | 键路径不变 |
| `PrefillStats` 那 5 个字段 | ✓ | ✓(多一个 `num_cache_creation_tokens`) | 命中来源 oracle 不变 |
| `mamba_cache_mode` / `align` | ✓ | ✓ | hybrid 设置不变 |
| `is_mm_prefix_lm` | ✓ | ✓ | chunked prefill 路由不变 |
| gemma4 `getattr(config, "global_head_dim", config.head_dim)` | 坏 | **一样坏** | Gemma 4 的 `hf_overrides` 照旧需要 |
| `(req_block_ids,) = ...get_block_ids(req_id)` | 坏 | **一样坏**(TODO 注释逐字未动) | hybrid 上"装载失败即致命"仍然成立,60s 心跳窗口照旧要 |
| `block_pool.cache_full_blocks` 里的 `assert blk.block_hash is None` | 有 | **没了** | `escalations/1_` 那个崩溃点在 0.27.1 上已不存在,上报前要在新版本上重跑复现 |

**勘察结论是对的,但它只覆盖 API 形状,没覆盖行为。** 真正的问题在下一节,
静态勘察一点没看出来。

## 四、回归:A/B 对照

同一提交 `0040c6bd`、同一棵 lmcache 树(0.23.0 侧用 `PYTHONPATH` 指过来)、
同一个 harness、各自独占空闲 GPU。

| 模型 | vLLM 0.23.0 | vLLM 0.27.1 |
|---|---|---|
| `qwen3.5-2b`(GDN hybrid,MP 路径) | **27 passed / 0 failed**(560s) | **4 failed / 23 passed**(786s) |
| `qwen2-vl-2b`(纯 attention,in-process) | **29 passed / 0 failed**(747s) | **27 failed / 2 passed**(3081s) |

0.23.0 那一列是今天同机重跑的,不是抄证书:27 与 29 分别和 `../22/all12/INDEX.md`
里 qwen3.5-2b、qwen2-vl-2b 的项数**逐个对上**。

失败形态不是计数器对不上,是**命中路径的输出坏掉**:

```
# qwen2-vl-2b(in-process)
[T0.1 different image B] got=''                            baseline='Blue'
[T0.2 replay]            got='念 Ivycourtounter apedバー a'  reference='Purple'

# qwen3.5-2b(hybrid / MP)
[T0.2 replay]     t02-5:  got='white'          reference='yellow'
[T0.4 phase 3]  t04-p3-A: got='white'          baseline='green'
[T3 t22 AC]      t22-AC:  got='yellow, yellow' baseline='green, yellow'
```

baseline 是**同版本同配置**的裸 vLLM,所以分歧只能出在 LMCache 这一侧;
miss 路径对得上、hit 路径吐空串和乱码。

**一个还解释不了的不对称:** 纯 attention 挂 27 项,hybrid 只挂 4 项。
按"命中路径整体坏了"的说法,hybrid 应该挂得更多而不是更少。这条不猜,
留给下一步。

## 五、文本模型对照:问题不在多模态代码路径上

思路是用户给的:跑一个纯文本模型比准确率。准确率不变就说明问题在 mm 路径,
掉了就说明在通用 KV 路径。

探针 `vllm_upgrade/text_accuracy_probe.py`:Qwen3-0.6B,16 个案例,每个案例一篇
约 450 token 的文档,文档中间写死一个颜色,问题问这个颜色。量三个数 —— 裸 vLLM
基线、LMCache 冷跑(本地算)、LMCache 命中(从 cache 装载)。**不 import
`tests/e2e_mm` 的任何东西**,这条路径上没有一行 mm 代码。

有效性闸门(上一轮的教训:探针的红色必须先在已知全绿的版本上证明自己不红):

- 每个命中案例必须报 `num_external_cached_tokens > 0`,否则判 `valid: false`,
  不出结论;
- 每条 prompt 单独发(batch=1),冷热两趟 batch 组成一致 —— 否则光是数值抖动
  就能自己翻票;
- 冷跑必须先和裸 vLLM 逐字相同。

| | vLLM 0.23.0 | vLLM 0.27.1 |
|---|---|---|
| 裸 vLLM 基线准确率 | 0.9375 | 0.8125 |
| LMCache 冷跑准确率 | 0.9375 | 0.8125 |
| **LMCache 命中准确率** | **0.9375** | **0.0** |
| 每次命中的外部装载 token | 448 / 450 | 448 / 450 |
| 本地前缀缓存命中 token | 0 | 0 |
| 冷跑 == 裸基线 | 16/16 | 16/16 |

两版的基线本身不同(0.9375 vs 0.8125),那是 vLLM 自己在一个 0.6B 模型上的版本
漂移,和 LMCache 无关;要比的是每一列内部那三行。

0.27.1 上 16 个案例**一个都没答对**,输出形态和 mm 套件里的一模一样:

```
案例 0  truth=violet   冷跑=' violet\n\nAnswer:\nThe stone kept in box'
                       命中='  "The answer: turquoise\n\nThe answer t'
案例 4  truth=indigo   命中='  "!\n\n"!!" \n\n!!" \n\n"!'
```

**结论:问题不在多模态代码路径上。** 装载确实发生了(两版都是 450 里的 448 个
token 走外部通道),装进来的 KV 是错的。多模态套件只是最先撞上它的东西。

还有一个必须记下来的校准:**"命中输出与冷跑逐字相同"这个指标在已知全绿的
0.23.0 上也只有 0.6875。** 答案词永远对,分歧全在答案后面那段续写 —— KV 走一趟
CPU 再回来,末位比特变了,接近平局的 token 就会翻。所以逐字相等是噪声主导的
统计量,不能拿来下结论;能下结论的是准确率。套件的 oracle 本来也是按答案比的,
不是按整段文本比的。

顺带纠正第七节第 5 条:那句被撤回的"LMCache 的装载路径在 0.27.1 上整体坏了",
方向是对的。但撤回本身也是对的 —— 那个探针在 0.23.0 上同样报红,它的红不携带
信息。现在这个探针在 0.23.0 上是绿的,它的红才算数。区别不在结论,在有没有对照。

副产品:复现这个 bug 的成本从"一个 50 分钟的 mm 套件"降到"一个 6 分钟的 0.6B
文本探针"。二分 0.24/0.25/0.26 因此只剩下载 wheel 的成本,不用每版跑一遍套件。

### 五之二、基线对照:本分支是否有份?

问题是用户提的:不用这个分支的 lmcache,还坏不坏。

把同一个探针跑在 `dev@09bc14c0`(本分支的祖先提交)上,`PYTHONPATH` 指向一个
detached worktree `LMCache-worktrees/dev_base`;native `.so` 从本工作树拷过去,
因为两棵树之间 **没有一行 C 源码差异**。补齐 2×2:

| LMCache 树 | vLLM 0.23.0 命中准确率 | vLLM 0.27.1 命中准确率 |
|---|---|---|
| `dev` @ `09bc14c0` | **0.9375** | **0.0** |
| `multi_modal` @ `0040c6bd` | **0.9375** | **0.0** |

不只是数一样 —— 逐个案例比对,两棵树的冷跑输出 16/16 逐字相同,命中输出也
16/16 逐字相同,两个 vLLM 版本上都是。

**本分支与这个回归无关。** 静态上也对得上:分支对 in-process 连接器的唯一改动
(`vllm_v1_adapter.py:1417`)整段在 `if mm_hashes and mm_positions:` 里面,纯文本
请求根本进不去;`utils.py` 那 97 行全是 mm hash 替换;其余生产代码改动都在 MP /
distributed 路径上,不在这条 in-process 路径上。

所以要修的是 **base 对 vLLM 0.27.1 的支持**,不是这个分支。

## 六、根因:vLLM 0.27 换了 KV cache 的内存布局

先把范围收干净。三棵 lmcache 树,同一个探针,同一个 vllm-mm venv,vLLM 0.27.1:

| lmcache 树 | native 构建 | 命中准确率 |
|---|---|---|
| `multi_modal` @ `0040c6bd` | 8/19 旧构建,无 `cuda_ops` | 0.0 |
| `dev` @ `09bc14c0`(本分支祖先) | 同上 | 0.0 |
| **`dev` @ `c1ef01b9`(今天的上游 HEAD)** | **全新完整构建,含 `cuda_ops`** | **0.0** |

三者输出逐字相同。所以既不是分支,也不是我们那份一周前的快照,也不是"native
扩展没编全"—— 上游今天的 HEAD 一样坏。顺带排掉的还有:

- **旧 `.so` 的 torch ABI**:`lmcache_native.so` 的未定义符号里 **一个 torch/c10
  符号都没有**(`libc10` / `libtorch_cpu` 只是链接残留的 `DT_NEEDED`)。它根本不碰
  torch 的 C++ ABI,torch 2.11 编的拿到 2.13 上用没有 ABI 面。
- **attention backend 换了**:两版都是 `FLASH_ATTN` + FlashAttention 3。

然后把 vLLM 交给 LMCache 的 KV cache 张量打印出来,答案就在上面:

| | vLLM 0.23.0 | vLLM 0.27.1 |
|---|---|---|
| 张量 shape | `[23205, 2, 16, 8, 128]` | `[23089, 8, 16, 256]` |
| stride | `[32768, 16384, 1024, 128, 1]` | `[32768, 256, 2048, 1]` |
| contiguous | True | **False** |
| vLLM 自报 `kv_cache_layout` | NHD | NHD |
| LMCache 检测出的格式 | `NL_X_NB_TWO_BS_NH_HS` | `NL_X_NB_BS_NH_CS` |

**K/V 那根轴没了。** 0.27.1 把 K 和 V 融进最后一维(256 = 2×128),head 轴挪到
block_size 前面,而且张量不再连续。两版的 `kv_cache_layout` 却都仍然是 `NHD` ——
这个 hint 描述不了这个差别。这正好解释症状:token **数量**装对了,**字节**是错的。

这条也顺手解掉了上一节留的 torch 混淆变量:KV 张量的形状是 vLLM 的 attention
backend 分配的,和 torch 版本无关。变量是 vLLM。

LMCache 的检测本身**不是**瞎猜:按 stride 的物理顺序看,0.27.1 的内存排布确实是
`[NB, BS, NH, CS]`,检测选的就是它。所以下一个问题不是"认错了布局",而是
**`*_CS` 这条路径有没有把 CS=2×head_dim 正确地拆回 K 和 V** —— 枚举里同时存在
`NL_X_NB_BS_NH_TWO_HS`(K、V 各占半)和 `NL_X_NB_BS_NH_CS`(一整块),选错哪一个
都会让 token 数对、内容错。检测代码在
`lmcache/v1/gpu_connector/kv_format/detection.py:69`,格式分派在
`lmcache/v1/platform/torch_ops.py:826-847`。**这一步还没做,等指令。**

证据:`vllm_upgrade/kvlayout_*.json`、`kvfmt_*_detection.txt`、
`kv_layout_dump.py`。

### 六之二、轴序假设被证伪

最新的 lmcache **有**这个新布局的支持代码,不是没做:`specs/nl_x_nb_bs_nh_cs.py`
写的就是 "CS == 2 * head_size, K/V packed",老的 `*_TWO_HS` 两个 spec 都标了
DEPRECATED "superseded by NL_X_NB_BS_NH_CS",检测器里那个 rank-4 分支的注释是
"Blocks-first fused K/V is the only rank-4 vLLM layout"。所以这是**新写的适配
代码里的缺陷**,不是缺功能。

检测器自己承认中间两根轴 NH/BS 分不出来,靠 `kv_layout` 这个 hint 决定,而 hint
默认是 NHD("Connectors do not specify a kv cache layout, defaulting to NHD")。
于是拿 `VLLM_KV_CACHE_LAYOUT=HND` 强制走另一条分支:

| 0.27.1 配置 | 检测出的格式 | 命中准确率 |
|---|---|---|
| 默认 | `NL_X_NB_BS_NH_CS` | 0.0 |
| `VLLM_KV_CACHE_LAYOUT=HND` | `NL_X_NB_NH_BS_CS` | 0.0 |

**两次的坏输出 16/16 逐字相同。** 格式换了,坏法一模一样 —— 如果是轴序解释错,
换个解释应该错得不一样。所以轴序不是变量,这个假设死了;坏的东西在装载路径上
更靠里、且与布局解释无关的地方。

顺带又排掉两个:失败的那几次用的是**真的 `cuda_ops`**(dev_head 编了,日志里
没有 fallback 警告),不是 torch fallback;而 dev_base 没编 `cuda_ops`、走
fallback,一样坏。

0.23.0 的 HND 对照报错了,但原因无关:dev_base 没编 `cuda_ops`,非 CUDA fallback
明确不支持 HND ——
`NotImplementedError: HND layouts ... are not supported in the non-CUDA fallback`。
这本身是个独立的小发现。

**还没分清的一个岔路:** lmcache 的 fused 布局支持是照着上游 CI 真正在测的 vLLM
**0.28.1.dev** 写的,0.27.1 stable 的变体可能不一样。如果 0.28 nightly 上是绿的,
这就是"版本覆盖缺口"而不是"bug",上报方式完全不同。要分清得装一个 0.28 nightly
再跑一遍同样的探针。**等指令。**

## 七、我自己犯的错(六个,都是自找的)

前四个直接拖慢了进度,后两个更严重 —— 它们污染了结论。

1. **用 pip 而不是 uv。** 整个 8.5 GB 的 venv 是 pip 一个包一个包装的。
   uv 快一个量级,而且能从全局 cache 硬链接复用。是用户提醒才想起来的。
2. **没先看现成的 venv 就动手建。** `vllm-ci-compat` 本来就已经是
   0.27.1 + torch 2.13 + transformers 5.15.1,我造了个近乎重复的出来。
   事后用 `hardlink` 去重,**省回 7.55 GiB**(vllm-mm 8.7G → 1.2G)。
   不动 ci-compat 本身是有理由的(它是只读对照),但这个理由当时没说出口,
   也不构成"重新下载一份 torch"的理由。
3. **`pgrep -f "vllm-mm/bin/pip"` 匹配到了它自己的 shell wrapper。** 于是
   `while pgrep ...; do sleep; done` 在 pip 早就结束之后还在空转,连着三次
   等待超时,纯浪费。判断"某进程还在不在"的 pattern 必须排除自己。
4. **`pkill` 把自己那条命令一起杀了**(exit 144),同一族问题。

下面两个是要记住的:

5. **信了一个没做对照的探针。** 我写了个纯文本 KV round-trip 探针,在 0.27.1 上
   看到 `miss != hit`、hit 输出是乱码,就直接对外说"LMCache 的装载路径在
   0.27.1 上整体坏了"。**跑了 0.23.0 对照才发现它在 0.23.0 上一样挂** ——
   而 0.23.0 是套件全绿的版本,所以坏的是我的探针配置,不是任何一个 vLLM。
   结论撤回了。讽刺的是最终 harness 给出的结论方向和它一致,但**这不能算它
   蒙对了**:一个在已知全绿的版本上也报红的探针,它的红色不携带任何信息。
   **教训:一个新探针在用来下结论之前,必须先在已知good的配置上跑出绿色。**
6. **GPU 重复占用,毁掉了第一次对照。** 0.23.0 对照被我排到 GPU 6,而我自己的
   探针还占着那块卡(38.73 GiB free / 需要 48.93),于是它死在显存上,报出来的
   `2 failed, 25 errors` 和被测行为毫无关系。更糟的是那两个探针**写完 JSON
   之后进程没退**,两块卡各压着 ~100 GB。排任务之前必须真的看一眼
   `nvidia-smi`,而不是看自己的记忆。

## 八、下一步

2. **定位回归。** 已经定到布局层(第六节)。下一步不是二分 vLLM,而是读
   `detection.py` 和 `torch_ops.py` 里 `*_CS` 这条分派,确认 CS=2×head_dim 是否
   被正确拆回 K/V。真要二分,0.24/0.25/0.26 现在也只要每版跑一次 6 分钟的文本
   探针。**等明确指令再开跑。**
   两个 detached worktree 先留着:`dev_base`(`09bc14c0`)、`dev_head`
   (`c1ef01b9`,已按 torch 2.13 编好 native 扩展)。
3. **证书应该记 vLLM 版本。** 现在 `tested_tree` 只钉了 commit,12 张证书都没写
   自己是在哪个 vLLM 上量的 —— 而今天证明了这正是能让整套结论翻盘的变量。
   重新认证时应该进 schema。
4. `escalations/1_` 上报前要在 0.27.1 上重跑复现:那个 assert 在新版本上没了。
5. DeepSeek-OCR(MLA + MoE)和 Mistral Small 3.1 已具备注册条件,但要等回归解决。
6. 仍未推送:本地 14 个提交(本次没有新增代码提交)。
