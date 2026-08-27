# batch 不变性对照实验:GLM flip 口径获用户签收

日期:2026-08-20
分支:`multi_modal` @ `1129b2e2`(本地提交,未推送;fork 上是 a3c6a2c3)
前置记录:`3_flip_gate_calibration_three_greens.md`(其追记即本实验的结果摘要)

## 起点:用户对 GLM "parity PASS" 的三连追问

本段是用户对 GLM-4.6V-Flash 认证结论的质询与最终签收,三个问题层层递进:

1. **"parity PASS?"** — 澄清口径:两个 Qwen 是默认 0.5% flip 预算干净 PASS;
   GLM 是翻转超标(27/17 > 11.87)→ 三步取证 → 文档化模型级口径 1.5% 后
   PASS。四道 gate(hit_ratio=1.0、分差 +1.34/+5.93、parse 0.9208、翻转)
   逐项列表呈现,并给出备选(降回 PROVISIONAL / 更严口径)。
2. **"你不能把推理关了?"** — 已经关了:spec 里 `enable_thinking=False`
   (specs.py:117),套件与 parity 全生效;当初弃 GLM-4.1V-9B-Thinking 选
   4.6V-Flash 正是为此。但关的只是 `<think>` 结构块,GLM 仍用散文写长推理
   前言再给盒装答案——模型风格,模板开关去不掉。改 prompt 强制短答案会
   偏离 MME 标准口径且压不掉根因(边界题最后一个 token 照样能翻),未采用。
3. **"为什么会翻?你就不能把随机性去掉?"** — 随机性本来就是零
   (temperature=0, seed=0,同配置逐字节复现已证)。翻转机理:浮点非结合
   性 → kernel 归约顺序随 batch 组成变(vLLM kernel 非 batch-invariant,
   FlashAttention split-K 拆分随 batch/长度变)→ logits 差 1e-6 量级 →
   边界题 top-1/top-2 几乎并列时 argmax 翻 → 贪心解码级联,长前言放大
   200+ 倍机会。这同时解释了短答案 Qwen 翻得少、长推理 GLM 翻 ~1%。

## 对照实验(用户批准):只改 batch,不带 LMCache

设计:**完全无 LMCache**,与归档 baseline 唯一差别是 `max_num_seqs`
默认 → 64(scratchpad `glm_baseline_batch64.py`,复用 benchmark_parity 的
engine_kwargs/run_batch,不动仓库代码)。GPU 3,~40 分钟。

结果(对比归档 `parity_glm-4.6v-flash.baseline.json`,同 parse_yes_no 口径):

| 指标 | 带 connector | 只改 batch(无 LMCache) |
|---|---|---|
| 翻转 | 27/2374 = 1.14% | **121/2374 = 5.10%** |
| 逐字节相同 | — | 仅 51.2% |
| 总分 | +1.34 | −0.67pt(78.81%→78.14%) |
| parse ratio | 0.9208 | 0.9115 |

形态与 connector 翻转完全同型:集中在 celebrity(33)/artwork(24)/
posters(16) 等长前言类目,方向大体对称(toward-gold 43 / away 59),
一半以上涉复读截断的 ''。分数几乎不动 → 全是边界题两可摇摆。

**结论:"换个批法就换个答案"是 vLLM 贪心解码的固有数值性质;connector
的 1.14% 远低于引擎自身 batch 敏感度本底 5.10%(4.5 倍差距)。GLM 的
1.5% flip 口径由此获得独立于 LMCache 的对照证据,实际相当保守。**
彻底消除需要 batch-invariant kernel(上游没有,业界研究课题)。

## 落地

- 证据归档本目录:`glm_batch_invariance_control.txt`(对比全文)+
  `glm_baseline_batch64.json`(812K 答案原文)。
- specs.py GLM flip 注释补上对照数字;记录 3 加追记。
- 提交 `1129b2e2` "test(e2e_mm): batch-invariance control evidence in GLM
  flip-budget note"(ruff 全绿;本地,未推送)。

## 用户签收 + 总结交付

用户:"ok。这个模型过了。" — **GLM-4.6V-Flash SUPPORTED 正式签收**。
随后按用户要求交付了总结(相对 dev merge-base 核实):

- **已支持 3 模型**:Qwen2.5-VL-3B、Qwen2-VL-2B(默认口径)、
  GLM-4.6V-Flash(1.5% 口径 + 对照背书)。
- **核心库改动仅 6 文件 +236/−99**:utils.py 16-bit→31-bit mm_hash keying
  (修跨图串缓存,点亮 ~12 个搭车模型)、vllm_v1_adapter.py 抢占尾 token、
  三个 MP connector 补 MM 替换、包内单测。零测试专用改动。
- **测试设施 ~3800 行新建**:29 场景合成套件、MME parity 四 gate、
  certify.py 证书、适配全 spec 化(6 个字段,铺新模型只加一个条目)。

## 经验沉淀

- **对照实验是归因的终点**:三步取证(自一致/复现/逐题)证明了"确定性、
  良性",但只有"无 LMCache 只改 batch 翻 5.1%"这一刀把"是不是 LMCache
  的锅"变成了可量化的否——而且成本只有 40 分钟。阈值争议优先做对照,
  不做口舌辩护。
- **用户的质询顺序就是证据链该有的顺序**:PASS 口径 → 能不能关推理 →
  为什么翻/能不能去掉。每一问都对应一个本应提前备好的证据;下个模型
  认证时把对照实验直接纳入标准流程(翻转超默认口径即跑 batch 对照)。

## 下一步

- #4 InternVL 3.5(小杯)铺模型(用户已知晓,待启动)。
- 待推送:`1129b2e2`(注释提交,可与下批工作一起推 fork)。
- P5 bypass 护栏代码;MP 竞态家族立项(两案证据在 records/2026/08/20/)。
