# 模型优先级修订(2026-08 最新发布核查)

日期:2026-08-21
分支:`multi_modal` @ `61e23812`
前置记录:`1_internvl35_2b_certification_supported.md`、`../19/1_multimodal_support_investigation.md`
触发:用户指出 "qwen都已经出到3.8了,我怀疑你没跟上最新的进展" —— 属实,
08-19 的 18 模型列表在 Qwen 谱系上过时(当时就漏了 2 月的 Qwen3.5)。
用户随后指示:"继续按照顺序推进。你不要问我意见……自己推进"
(视为 InternVL3.5-2B 隐式签收 + 全程自主授权)。

## 核查结论(web 检索,2026-08-21)

- **Qwen 谱系**:Qwen3-VL(2025-10)→ Qwen3.5(2026-02,397B-A17B 起,
  后补 27B/9B/4B/2B)→ Qwen3.6(2026-04,27B/35B-A3B 原生视觉,
  vLLM ≥0.19 可跑)→ **Qwen3.8(2026-08-03 Max 2.4T MoE;08-14 开源
  27B,Apache 2.0)**。架构质变:自 3.5 起全家混合注意力,3.8-27B 的
  64 层中 **48 层 Gated DeltaNet 线性注意力(常数 recurrent state,
  非逐 token KV)**,16 层全注意力,且保留 DeepStack。
- **Gemma**:Gemma 4 已于 2026-03-31 发布(E2B/E4B/12B/31B/26B-A4B),
  统一多模态、无独立视觉编码器;vLLM 有活 bug **#40106**:
  `use_bidirectional_attention="vision"` 被静默忽略按因果跑。
  原列表 #5 Gemma 3 应替换为 Gemma 4。
- **GLM**:开源视觉线最新仍是 GLM-4.6V(已认证);GLM-5V-Turbo(04-01)
  是 API 模型,GLM-5.3(08-14)不带视觉。#3 选型仍对。
- **InternVL**:无 4;3.5 仍是主线(InternVL-U 是理解+生成一体的
  另一方向)。#4 选型仍对。

## 对排序的实质影响

1. **recurrent state 缓存从长尾(原 #18 Kimi K3 单独立项)跃升为
   最高优先级独立项目**:线性注意力层的 state 不在 paged KV 里,
   LMCache 逐 token KV 模型覆盖不了;这已是 Qwen 旗舰全家的主流架构。
2. **Qwen3-VL(纯全注意力 + DeepStack)仍是正确下一步**:隔离验证
   DeepStack 的最干净入口,不需升级 vLLM,结论惠及 3.5/3.6/3.8 全系。
3. vLLM 0.23.0 撑不到 Qwen3.8(3.6 可跑);认证 3.8 需升级 venv。

## 修订后顺序

1. **Qwen3-VL 小杯(2B)**:DeepStack 验证 + 认证 ← 当前进行中
2. 混合注意力/recurrent state 立项调研(Qwen3.5/3.6/3.8 + Kimi K3)
3. Gemma 4-E4B(双向注意力验证,对照 vLLM #40106)
4. Qwen3.6-27B(0.23 可跑)→ 升级 vLLM → Qwen3.8-27B
5. 搭车行随时插队:DeepSeek-OCR、MiniCPM-V、Mistral Small 等

## 经验

- 优先级列表要带"核查日期",隔周即需重验——2026 年 VLM 发布节奏
  以周计,08-19 的调研 08-21 已过时一档。
