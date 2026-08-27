# qwen2-vl-2b 升 SUPPORTED + 支持检测机制设计收口

日期:2026-08-19
分支:`multi_modal` @ `4f2bc199`(本段无新代码提交,工作树干净)
前置记录:`5_certification_completeness_review.md`

## 会话内容

1. qwen2-vl-2b 全量 MME parity 完成并重签证书;2. 回答用户"支持检测设计完了吗"。

## qwen2-vl-2b 全量 MME(2374 题)

**最强形式 PASS**:baseline / pass1(miss)/ pass2(hit)三组逐分一致,总分 1966.06;双向翻转 0/2374;pass2 命中率 1.000。报告归档本目录 `mme_full_qwen2-vl-2b.json`(3B 的也已归档为 `mme_full_qwen2.5-vl-3b.json`)。

重签后:**qwen2-vl-2b = SUPPORTED @ 4f2bc199**(证书本目录)。至此两模型双 SUPPORTED,且 2B 全程零新增测试代码——横向流程(specs.py 一行 + `certify.py <key> --run-parity`)端到端验证成立。

## 设计收口结论(答用户问)

**机制设计已完成闭环**:定义(四项职责 + 失效模式→探测器映射表)→ 两层检测(合成套件定位 / MME 认证)→ 自证(negative control + oracle 经 2B 事件归因考验)→ 结论产出(certify.py → 证书,CI 可用)→ 可扩展性(2B 零代码认证)。

**剩余是覆盖实现,不是设计**(均在证书 known_not_covered 挂载点上):
- T3 部署路径(MP connector / TP>1 / 远端后端)——矩阵已定义"同一套 T0+T1 按 path 重跑",依赖 P2 代码先做出来;是唯一会改变证书 scope 结构的项。
- video/audio(T2.3)——矩阵有定义,spec.modalities 与证书 scope 已预留。
- 特殊架构挂件(DeepStack / Gemma3 / Phi-4 LoRA)——extra_suites 字段预留。
- 抢占重算——仅 MME 统计性覆盖,确定性测试难做,已如实写进边界。

## 下一步(等用户拍板)

- P2:`_0180`/`_0201` MP connector 补 MM(生产主路径,T3 第一条新 path)。
- 横向继续铺模型(流程已验证)。
- pin count bug + allocator 层守恒对账(单独立项)。
