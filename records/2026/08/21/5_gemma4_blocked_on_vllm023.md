# Gemma 4 在 vLLM 0.23 上不可加载:推迟到 venv 升级批次

日期:2026-08-21
分支:`multi_modal` @ `0c911591`
前置记录:`2_model_priority_revision_aug2026.md`(修订顺序第 3 项)

## 事实

- vLLM 0.23 有完整的 gemma4 家族实现(gemma4_mm / gemma4_unified /
  gemma4_mtp),E4B/12B 权重不设 HF 门禁,下载正常。
- **E4B 与 12B 都加载失败**,两段式:
  1. transformers 5.15 的异构保护:`AmbiguousGlobalPerLayerAttribute-
     Error: 'head_dim' is a per-layer attribute`——vLLM 0.23 的
     `model_arch_config_convertor.get_head_size` 做全局
     `getattr(head_dim)`。裸 config 无 per_layer_config、head_dim
     全层 256,该保护可用 `hf_overrides={"allow_global_per_layer_
     attribute_access": True}`(顶层+text_config 双写)绕过;
  2. 绕过后崩在权重装载:`Attempted to load weight (torch.Size([512]))
     into parameter (torch.Size([256]))`——checkpoint 存在 vLLM 0.23
     的 gemma4 实现没有建模的逐层异构维度(E4B 与 12B Unified 同症),
     不是配置问题,是实现版本落后于最终 checkpoint 格式。
- E4B 侦察到的备用情报:无 `use_bidirectional_attention` 标志
  (双向注意力验证的载体只能是 31B / 26B-A4B);滑窗混合
  (42 层 full+sliding, 窗 512);图像 token 固定 280
  (`vision_soft_tokens_per_image`,无需像素预算 kwarg);带音频塔
  (若认证,将是首个 audio 模态模型,探针机制需新设计)。

## 决定

Gemma 4 认证推迟到 **vLLM venv 升级批次**(与 Qwen3.8-27B 同波,
升级本身另立项:参考 memory `vllm-lazy-venv-build` 的构建坑)。
当前 venv 上继续可做的:mm×hybrid 套件改造(Qwen3.5-2B →
Qwen3.6-27B)、搭车行 spec(DeepSeek-OCR、MiniCPM-V 等)。
