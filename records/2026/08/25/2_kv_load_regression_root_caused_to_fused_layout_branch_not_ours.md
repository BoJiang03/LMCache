# 0.27.1 回归定界:base 的 KV 装载缺陷,与本分支无关,与多模态无关

日期:2026-08-25(下午,接本日 [`1_`](1_vllm_0271_upgrade_two_models_unblocked_hit_path_regression.md))
分支:`multi_modal` @ `0040c6bd`(工作树全程干净,本次**没有改动任何仓库代码**;
全部工作是探针、对照实验和两个 detached worktree)

## 结论先写

1. **回归与本分支无关。** 三棵 lmcache 树(本分支、分支祖先 `dev@09bc14c0`、
   **今天的上游 HEAD `dev@c1ef01b9`** 重新完整编译)在 vLLM 0.27.1 上坏法
   **逐字相同**:纯文本命中准确率全部 0/16。
2. **回归与多模态无关。** 复现件是 Qwen3-0.6B 纯文本探针,路径上没有一行 mm
   代码。mm 套件只是最先撞上它的东西。
3. **直接诱因是 vLLM 0.27 换了 KV cache 内存布局**(K/V 融合进末维、head 轴
   前移、张量非连续),而 lmcache **有**这个布局的适配代码(`*_CS` 格式族,
   老格式已标 DEPRECATED)—— 所以这是**新适配代码里的缺陷**,不是缺功能。
4. **但缺陷不在轴序解释上**:强制 HND 让检测结果换了一个格式,坏输出仍然
   16/16 逐字相同。真正的病灶还没找到,在装载路径更深处。
5. 已认证的 12 张证书(全部量于 0.23.0)**不受影响**,今天两个同机对照
   (qwen3.5-2b 27/27、qwen2-vl-2b 29/29)与证书逐项吻合。
6. 被挡住的是**扩张**:DeepSeek-OCR 和 Mistral Small 3.1 在 0.27.1 上已经
   model-ready,但整个套件(14 个模型一视同仁)在 0.27.1 上不可用。

证据全部在 [`vllm_upgrade/`](vllm_upgrade/)。

## 一、复现件:带有效性闸门的纯文本探针

用户的思路:跑一个纯文本模型比准确率,准确率不变则问题在 mm 路径,掉了则在
通用 KV 路径。答案是后者。

`text_accuracy_probe.py`:Qwen3-0.6B,16 个案例,每案例约 450 token 文档中间
埋一个颜色,问答案。量三个数:裸 vLLM 基线 / LMCache 冷跑 / LMCache 命中。
闸门(上午的教训:探针必须先在已知 good 配置上证明自己是绿的):

- 每个命中案例必须 `num_external_cached_tokens > 0`,否则判 `valid: false`;
- batch=1,冷热两趟 batch 组成一致;
- 冷跑必须与裸基线逐字相同。

| | vLLM 0.23.0 | vLLM 0.27.1 |
|---|---|---|
| 裸基线准确率 | 0.9375 | 0.8125 |
| 冷跑准确率 | 0.9375 | 0.8125 |
| **命中准确率** | **0.9375** | **0.0** |
| 外部装载 token/命中 | 448/450 | 448/450 |

装载数量对,内容错。两版基线本身的差(0.9375 vs 0.8125)是 vLLM 自己在 0.6B
上的漂移,与 LMCache 无关。

一个校准值:**"命中输出与冷跑逐字相同"在已知全绿的 0.23.0 上也只有 0.6875**。
答案词永远对,分歧全在答案后面的续写(KV 过一趟 CPU,末位比特变,近平局
token 翻)。逐字相等是噪声主导的统计量,不能作为判据;判据是准确率。

## 二、定界:三棵树、两种构建、两条 ops 路径、两个轴序,全部同坏

| 变量 | 取值 | 0.27.1 命中准确率 |
|---|---|---|
| lmcache 树 | `multi_modal@0040c6bd` / `dev@09bc14c0` / `dev@c1ef01b9`(上游今日 HEAD) | 0.0 / 0.0 / 0.0,**输出逐字相同** |
| native 构建 | 8/19 旧构建(无 `cuda_ops`)/ 今日全新构建(含 `cuda_ops`) | 都 0.0 |
| ops 路径 | 真 `cuda_ops` / torch fallback | 都 0.0 |
| 轴序 hint | 默认 NHD(检测出 `NL_X_NB_BS_NH_CS`)/ 强制 HND(检测出 `NL_X_NB_NH_BS_CS`) | 都 0.0,**输出 16/16 逐字相同** |

顺带排掉的硬证据:

- **旧 `.so` 的 torch ABI 嫌疑死了**:`lmcache_native.so` 未定义符号里一个
  torch/c10 符号都没有(`libc10`/`libtorch_cpu` 只是链接残留 DT_NEEDED)。
- **attention backend 没变**:两版都 FLASH_ATTN + FlashAttention 3。
- **torch 版本混淆变量解掉**:KV 张量形状由 vLLM 的 backend 分配,与 torch 无关。

## 三、机制层:vLLM 0.27 的 KV cache 布局

| | vLLM 0.23.0 | vLLM 0.27.1 |
|---|---|---|
| 张量 | `[23205, 2, 16, 8, 128]`,连续 | `[23089, 8, 16, 256]`,**非连续** |
| stride | `[32768, 16384, 1024, 128, 1]` | `[32768, 256, 2048, 1]` |
| vLLM 自报 layout | NHD | NHD(hint 描述不了这个差别) |
| lmcache 检测 | `NL_X_NB_TWO_BS_NH_HS` | `NL_X_NB_BS_NH_CS` |

K/V 轴没了,融进末维(256 = 2×128)。lmcache 的适配代码是**真实存在且新写的**:
`specs/nl_x_nb_bs_nh_cs.py`("CS == 2 * head_size, K/V packed"),老的 `*_TWO_HS`
标 DEPRECATED,检测器 rank-4 分支注释"Blocks-first fused K/V is the only rank-4
vLLM layout"。检测选对了物理布局(按 stride 验算过),轴序假设也被 HND 实验
证伪 —— **病灶在更深处,未定位。等指令再挖。**

独立小发现:非 CUDA fallback 不支持 HND
(`NotImplementedError: HND layouts ... are not supported in the non-CUDA fallback`)。

## 四、模型支持口径的三次纠正(都记下来,别再犯)

1. **"DeepSeek-OCR / Mistral 在 0.27.1 上不被支持"是错的说法。** 缺陷是全局的,
   在同版本上把已认证的 qwen2-vl-2b 打到 27 failed。这两个模型本身已 model-ready,
   被挡的是全部 14 个模型的套件。
2. **"0.23.0 上这两个模型不被 LMCache 支持"也是错的说法。** Mistral 的阻塞已
   静态验证为 vLLM 自己的 shim 缺 `fetch_images`(0.23.0 的
   `transformers_utils/processors/pixtral.py` 无此方法,0.27.1 的第 51 行有),
   崩在引擎 init 的 dummy mm 输入上,LMCache 还没被调用。裸 0.23.0 就起不来。
3. **DeepSeek-OCR 的 SIGFPE 至今没做过"无连接器"对照**,不能像 Mistral 一样断言
   是 vLLM 侧。前日记录也只写了"未知,升级后重测"。要断言,先跑一次裸 0.23.0。

正确口径:升级前两个模型**不可认证**(引擎起不来/崩),升级后**可认证但被全局
回归挡住**。

## 五、影响与下一步(全部等指令)

- 12 张 0.23.0 证书**不受影响**;两个今日同机对照与证书吻合。
- **最高性价比的下一步:装 vLLM 0.28 nightly 跑同一个探针**(上游 CI 实际钉的是
  `0.28.1.dev202608250650`)。绿:是"0.27.1 版本覆盖缺口",且可以直接在 0.28 上
  认证 DeepSeek-OCR、Mistral,再把 12 个模型全部重认证(证书 schema 应加
  vLLM 版本字段);红:证明缺陷波及上游自测版本,上报分量完全不同。
- 备选:直接挖 `*_CS` 装载路径(kernel 侧),或二分 0.24/0.25/0.26(每版 6 分钟)。
- 留存的基础设施:`dev_base`(`09bc14c0`)、`dev_head`(`c1ef01b9`,已按
  torch 2.13 编好含 `cuda_ops` 的完整 native)两个 detached worktree;
  6 分钟文本复现件 `text_accuracy_probe.py`;布局勘察件 `kv_layout_dump.py`。
- 仍未推送:本地 14 个提交(本次没有新增代码提交)。

## 六、本场没犯新错,但有两笔要记

1. 上午撤回的那句"装载路径整体坏了"方向被证实,但撤回依然是对的:没有对照的
   探针,红色不携带信息。今天的探针先在 0.23.0 上跑绿,红色才算数。
2. 差点把"全局缺陷"说成"两个模型的缺陷",又差点把"vLLM 的 shim 缺陷"说成
   "LMCache 不支持"。归因要落在测过的那一层,不能落在最先观察到症状的那一层。
