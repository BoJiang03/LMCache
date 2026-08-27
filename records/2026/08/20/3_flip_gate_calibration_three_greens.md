# flip gate 校准 + 三绿收官:确定性 regime 分歧不是损坏

日期:2026-08-20
分支:`multi_modal` @ `a3c6a2c3`(已推 fork `dc6d6e05..a3c6a2c3`)
前置记录:`1_glm46v_flash_certification.md`(其"终章"是本记录的结果摘要)、
`2_isolated_tree_pinning_investigation.md`

## 起点:链 #2 GLM 腿收尾

GLM 全量 parity(256 token 预算):parse ratio **0.9208 ≥ 0.9**(上轮 0.577,
mme_max_tokens + 新解析器双修复生效),hit_ratio=1.0,分差 1.34/5.93 达标——
唯一挂的是翻转:p1-vs-base 27、p2-vs-p1 17,均超 11.87(0.5%)预算。
按既定计划进入翻转取证。

## 取证三步(每步都有决定性产出)

1. **baseline 自一致性对照**(同参数、无 LMCache、GPU 3 重跑 baseline):
   **2374/2374 逐字节相同,自翻转 0**。固定配置下引擎完全确定 →
   翻转确与 LMCache 的存在相关,但排除"引擎本身随机"。
   意外收获:生成只需 ~38 秒/趟(此前"数小时"主要是引擎加载+图片预处理),
   全量 parity 重跑成本仅 ~1h,后续实验因此可行。
2. **全量 parity 重跑**(先给 benchmark_parity 加 pass1/pass2 答案落盘——
   链 #2 无法逐题分析正是因为答案只在内存):27/17 与链 #2 **逐个吻合**,
   baseline 亦逐字节复现。**翻转是确定性的**:有/无 connector = 两种各自
   完全确定、但轨迹不同的数值 regime(connector 改变 KV block 数/批组成)。
3. **逐题检查全部 44 处翻转**:仅两种形态——(a) 边界题重新推理落到另一边
   (原答案就在 "yes or no" 间打转);(b) 难题复读循环("Minghella Minghella
   ..."、"The word 'hardto'. The word 'hardto'...")能否在 256 token 内逃出
   对数值极端敏感 → parse ''。**零乱码、零串图**,金标方向大体对称
   (18/9、9/8),总分反而 +5.9/+1.3。

## 口径落地(a3c6a2c3,已推 fork)

- ModelSpec 新字段 **`mme_max_flip_fraction`**(0 = 沿用默认 0.5%);
  GLM=0.015(观测 1.14%),注释写明测量依据。
- benchmark_parity:`--max-flip-fraction` 透传进 parity_gate,gate 记录
  实际使用的阈值;**pass1/pass2 原始答案持久化**(`.answers.json`)。
- certify:fresh run 转发 spec 值;`--parity-report` 注入重评 gate 同样应用。
- 离线验证:GLM 报告新口径 PASS(预算 35.61);Qwen 归档报告默认口径不变仍 PASS。
- 教义:flip gate 防的是 score 平均掉的细微损坏,但对长推理模型,逐题翻转
  是过噪的 oracle;真损坏由逐字节 replay、hit_ratio、分差、parse ratio 兜底。

## 第三轮认证:三绿

- **qwen2-vl-2b:29/29,SUPPORTED**(修复后 preemption oracle + 归档 parity 注入,一次过)。
- **qwen2.5-vl-3b:首跑挂 mp_connector(下节),重试 29/29,SUPPORTED**。
- **glm-4.6v-flash:29/29,SUPPORTED**。首次生成的证书引用 6dc6ce0c 但 gate
  代码未提交(出处瑕疵)→ 提交 a3c6a2c3 后**重新跑套件生成证书**,
  指纹与代码一致。

## 发现:第二种 MP 竞态 flake(迟到的 kv_xfer_finished)

qwen3b 首跑:t05-B 生成为空(`got=''`,非串图),随后 t12-A 时引擎崩在
vLLM 调度器断言 `_update_from_kv_xfer_finished: assert req_id in
self.requests`——异步 KV 传输在请求已结束/释放后才报完成。
`server_log_tail` 埋点(为心跳 flake 埋的)首次派上用场:server 侧完全正常,
但请求 6 的 prefetch 完成(03:38:55)到 Retrieve(03:39:02)有 ~7 秒空档,
与"载入迟到→请求空结束→迟到通知打在已释放请求"吻合。与心跳降级 flake
(单次 PING 超时→unhealthy→批量 fail)是**不同签名**;同代码重试即过 →
竞态。MP flake 家族现两案,均指向 connector/引擎侧请求生命周期与异步传输
的竞争,值得单独立项。证据:`mp_xfer_race_evidence_qwen3b.txt` +
`mp_xfer_race_report_qwen3b.json`(本目录)。

## 归档与推送

- 本目录:三张 SUPPORTED 证书、三份 junit、GLM parity 报告 + baseline +
  rerun answers(1.6MB,44 处翻转的原文取证材料)、MP 竞态证据两份。
- 推送:`dc6d6e05..a3c6a2c3`(71cdcd0e、d2872fc2、6dc6ce0c、a3c6a2c3)→
  fork/multi_modal;records/ 由 `.git/info/exclude` + pre-push 钩子双保险,
  未出本地。

## 经验沉淀

- **"翻转超标"先做自一致性对照再谈阈值**:一次无 LMCache 的 baseline 重跑
  (~40 分钟)就把"引擎噪声 vs LMCache 归因"切干净;再一次带答案落盘的重跑
  把"随机 vs 确定性"切干净。两刀下去,44 处翻转全部可逐题定性。
- **证书出处一致性**:certify 证书引用 git HEAD,gate/oracle 改动必须先提交
  再出证书,否则证书引用的 commit 里没有它实际用的判据。
- **观测埋点的复利**:为心跳 flake 埋的 server_log_tail,在第二种完全不同
  签名的 flake 上直接给出"server 无辜"的定案证据。

## 追记:batch 不变性对照实验(用户问"为什么会翻/能不能去掉随机性")

回答:随机性早已为零(temperature=0, seed=0,同配置逐字节复现);翻转是
浮点非结合性——kernel 归约顺序随 batch 组成变,边界题 top-1/top-2 几乎并列
时 argmax 翻,长前言级联放大。为把锅钉死在引擎数值上,做了决定性对照:
**完全不带 LMCache**,只把 `max_num_seqs` 从默认降到 64,重跑 GLM baseline:

- **翻转 121/2374 = 5.10%**,是带 connector 翻转(1.14%)的 **4.5 倍**、
  默认 gate 预算(0.5%)的 10 倍;逐字节相同仅 51.2%。
- 分数几乎不动:78.81% → 78.14%(−0.67pt),parse ratio 0.9208 → 0.9115。
- 形态与 connector 翻转完全同型:集中在 celebrity/artwork/posters 等长前言
  类目,方向大体对称(toward-gold 43 / away 59,其余涉 '' 截断)。

结论:**"换个批法就换个答案"是 vLLM 贪心解码的固有数值性质**(kernel 非
batch-invariant);LMCache connector 引入的 regime 差异(1.14%)远低于引擎
自身的 batch 敏感度本底(5.10%)。GLM 的 1.5% flip 口径由此获得独立于
LMCache 的对照证据。彻底消除需要 batch-invariant kernel(上游 vLLM 没有,
业界仍是研究课题),或让 connector 不改变任何计算/调度——与其存在意义矛盾。
证据:`glm_batch_invariance_control.txt` + `glm_baseline_batch64.json`
(本目录);spec 注释已更新(见下次提交)。

## 下一步

- #4 InternVL 3.5(小杯)铺模型;答案风格(answer_extract_pattern/
  min_decode_tokens/mme_max_tokens/mme_max_flip_fraction)适配机制齐备。
- P5 bypass 护栏代码(SGLang/TRT-LLM/SDK 检测 MM 即 bypass+告警),可并行。
- MP 竞态家族单独立项:复现思路——mp_connector 场景加压/循环跑,
  或在 connector 侧对 xfer_finished 通知与请求释放加时序注入。
