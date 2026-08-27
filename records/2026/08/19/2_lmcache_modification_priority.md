# LMCache 多模态修改项排序(由模型优先级列表推导)

日期:2026-08-19
分支:`multi_modal`
前置记录:`records/2026/08/19/1_multimodal_support_investigation.md`(模型优先级列表及所有 file:line 依据)

## 会话内容

把 18 个模型的支持优先级列表按"每个模型需要 LMCache 改什么"归并去重,得到 LMCache 修改项的执行排序。

## 修改项排序

**P0 — 修 16-bit mm_hash 碰撞(核心修复)**
完整 mm_hash 不经 `hex_hash_to_int16` 截断,作为 `extra_keys` 传入 `token_database.py` 的 `_hash_tokens`(参数已存在、目前无调用者),与 vLLM block hash 设计对齐。一次点亮 ~12 个模型:Qwen2.5-VL、GLM-4.5V/4.6V、InternVL 3.5、DeepSeek-OCR、Mistral Small、MiniCPM-V、ERNIE-4.5-VL、Llama 4、Kimi-VL、Step-3、Molmo 2。改动点:`lmcache/integration/vllm/utils.py`、`vllm_v1_adapter.py`、`lmcache/v1/token_database.py`。

**P1 — 正确性测试 + 碰撞复现**
e2e 测试(不同图不串、同图命中、chunk 边界跨 placeholder)放 `multi_modal`;#3301 碰撞复现脚本放 `multi_modal_repro`。与 P0 同一交付。

**P2 — MP connector(`_0180`/`_0201`)补多模态处理**
当前这两条路径对 MM 请求 100% 必然串图(拿到 vLLM `block_hashes` 只做边界检查)。复用 P0 的替换或直接消费 block_hashes。

**P3 — DeepStack 验证/支持(解锁 Qwen3-VL、Qwen3.5)**
side buffer 在 paged KV 之外(vLLM bug #41485);验证 LMCache 命中跳过 prefill 后 side buffer 缺失的影响,结论三选一:天然安全 / 额外缓存 side buffer / bypass。

**P4 — Gemma 3 双向图像 attention 验证**
图像 token 双向 mask 与 chunk 边界的相互作用;不安全则限制对齐或 bypass。

**P5 — 文档修正 + 安全护栏**
`multimodal_models.rst` 撤掉过度承诺;SGLang、TRT-LLM、token 寻址 SDK 等未支持路径检测到 MM 请求即 bypass + 告警,把静默串图变为不命中。护栏成本低,可与 P0 同批做。

**P6 — Phi-4-multimodal:LoRA ID 进 CacheEngineKey**
`extra_keys` 注释已预留 LoRA ID,是 P0 机制的自然延伸。

**P7 — SGLang / TRT-LLM 真正的 MM 支持**(按需)

**P8 — 中期:EC engine 跨实例/远端共享**(生态空白、差异化机会,不阻塞 KV 正确性)

**P9 — Kimi K3 KDA state**(recurrent state block 非逐 token KV,单独立项)

## 关键结构

- **P0+P1+P2 是一个整体交付**:修复 + 证明 + 全路径覆盖,合入后优先级列表 #1、#3、#4、#7–#13、#15 立刻可用。
- P3、P4 是独立验证型任务,各解锁一个高热度模型系。
- 其余为长尾/中期。

## 下一步(已确认方向)

先在 `multi_modal_repro` 写 #3301 碰撞复现脚本坐实串图,再在 `multi_modal` 做 P0 修复。
