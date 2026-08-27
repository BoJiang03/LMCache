# 多模态模型支持调查(LMCache multi_modal 工作启动)

日期:2026-08-19(会话始于 08-18)
分支:`multi_modal`(基于 LMCache dev @ 09bc14c0);另建 `multi_modal_repro` 用于存放 repro/测试脚本(不合入 dev,仅供 PR reviewer 参考)。

## 会话内容

1. 从 origin 拉取最新 dev(e8f93818..09bc14c0),创建 `multi_modal` worktree 和 `multi_modal_repro` 分支。
2. 调查前沿多模态模型、vLLM 支持情况、LMCache 支持现状与问题。

## 核心发现

### LMCache 多模态现状(dev @ 09bc14c0)

- 支持机制:vLLM v1 in-process connector 和主 MP connector 上,`apply_mm_hashes_to_token_ids`(`lmcache/integration/vllm/utils.py:171-187`)把 placeholder 区间的 token ID 覆写为 mm_hash 派生值,再走常规 token 哈希。
- **致命缺陷:`hex_hash_to_int16`(`utils.py:143-168`)把 mm_hash 截断成 16 bit**,整个 placeholder 区间填同一个值 → 每张图只有 16 bit 熵。同 prompt + 同分辨率场景下,~300 张不同图即 ~50% 概率哈希碰撞 → **静默串图(false hit,返回别的图的 KV)**。对应社区 issue LMCache #3301(已被 stale 关闭未修)。
- 正确修法的通道已存在但是死代码:`token_database.py` `_hash_tokens` 的 `extra_keys` 参数(`:269-295`)全仓库无调用者。vLLM 自己的 prefix cache 就是把完整 `(mm_hash, offset)` 作为 block hash extra key。
- LMCache 自己的 EC engine(encoder 输出缓存,`lmcache/v1/ec_engine.py`)用的是完整 64-bit 哈希——KV 路径的 16-bit 是未回头修的早期权宜。
- **完全无多模态处理的路径(100% 必然串图)**:`lmcache_mp_connector_0180.py` / `_0201.py`(拿到了 vLLM 的 block_hashes 却只做边界检查)、SGLang 集成、TRT-LLM 集成、token 寻址 SDK/CLI。
- 当 vLLM 发请求级标识符(`chatcmpl-xxx-image-0`)而非内容哈希时,多模态请求永远不命中(安全但缓存失效)。
- 文档 `docs/source/recipes/multimodal_models.rst` 声称"自动处理、无需配置",与实际不符。
- 无端到端多模态正确性测试(只有 unit test)。

### vLLM 侧(最新 v0.27.1,2026-08-11)

- connector API 的 `NewRequestData.mm_features` 携带 `identifier`(mm hash)+ `mm_position`(本地 vLLM 0.23.0 已确认)。
- 主线多模态全部是 decoder-only 注入;encoder-decoder 仅剩 Whisper,Mllama 已移除 → 不需要考虑 cross-attention KV。
- Encoder 输出缓存仅引擎本地,跨实例共享/offload 是生态空白(E/P/D 分离 RFC 均停滞)→ EC engine 的差异化机会。

### 架构分类(decoder layer 是否被动过)

- 多数模型(InternVL、Kimi-VL、Pixtral、DeepSeek-VL2、MiniCPM-V、Molmo、Step-3 等):decoder 结构不变,仅输入端 embedding 注入。
- 结构不变但层内计算变了:Qwen 系 M-RoPE/Interleaved-MRoPE、GLM 3D-RoPE(每层 attention 的 q/k)、Gemma 3 图像 token 双向 mask。
- 真动了 decoder 内部:**Qwen3-VL/Qwen3.5 DeepStack**(多中间层注入,存 paged KV 之外的 side buffer,vLLM 自身有 bug #41485)、Phi-4-mm(模态 LoRA)、ERNIE-4.5-VL(模态隔离 MoE 专家)、**Kimi K3**(KDA 线性注意力 recurrent state,非逐 token KV)。

## 支持优先级列表(热度×易支持度,从高到低)

1. Qwen2.5-VL(极高/低)
2. Qwen3-VL dense+MoE(极高/中低,DeepStack 需验证)
3. GLM-4.5V / 4.6V(高/低)
4. InternVL 3.5 / Intern-S1(高/低)
5. Gemma 3(高/中低,双向 attention 需验证)
6. Qwen3.5(高上升/中低)
7. DeepSeek-OCR / OCR-2(高/低,单图收益小)
8. Mistral Small 3.1 / Ministral 3(中高/低)
9. MiniCPM-V 4.5/4.6(中高/低,收益小)
10. ERNIE-4.5-VL(中/低)
11. Llama 4(中/中)
12. Kimi-VL / K2.5(中/低)
13. Step-3(中/低)
14. Qwen3-Omni(中/中)
15. Molmo 2(中/低)
16. Phi-4-multimodal(中低/中,LoRA 入 key)
17. Gemma 4 Unified(中低/中高)
18. Kimi K3(高/**高**,建议单独立项)
— Whisper:明确不支持(cross-attn KV)

关键结构:#1–#4 及所有"搭车"行共享同一修复——完整 mm_hash 经 `extra_keys` 进 chunk hash;修一次点亮约 12 个模型。需单独设计的只有 DeepStack 验证、Gemma 双向 attention 验证、K3。

## 计划的工作(未开始)

1. 修 16-bit 碰撞:完整 mm_hash 走 `extra_keys` → `_hash_tokens`(与 vLLM 对齐)。
2. `_0180`/`_0201` MP connector 补多模态替换(或直接消费 vLLM block_hashes)。
3. 修正文档过度承诺;补 e2e 多模态正确性测试(两张不同图不串、同图命中),按优先级 #1–#5 建验证矩阵。
4. 碰撞复现脚本放 `multi_modal_repro` 分支。
5. 中期:EC engine 的 MP/远端支持;DeepStack 行为验证。

## 环境备注

- 本地 vLLM 0.23.0(`/home/bo/venvs/vllm-lazy`),落后最新版(0.27.1)四个版本;跑 K3 或最新 connector 验证需升级。
