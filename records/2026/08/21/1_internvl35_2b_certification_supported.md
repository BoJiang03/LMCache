# InternVL3.5-2B 认证:SUPPORTED(第 4 个模型)

日期:2026-08-21(工作实际发生于 08-20 06:00–07:45)
分支:`multi_modal` @ `61e23812`(本地提交,未推送;fork 上仍是 a3c6a2c3)
前置记录:`../20/5_keying_extra_keys_refactor_todo.md`
用户指令:"接下来继续按顺序推进模型支持与测试"(全程自主推进)

## 选型与铺模型(一个 spec 条目,零测试代码)

- **`OpenGVLab/InternVL3_5-2B-HF`**(transformers 原生导出,
  `InternVLForConditionalGeneration` → vLLM `interns1` 实现,image+video)。
  OpenGVLab 原格式(`InternVLChatModel`)带自定义 config 代码需
  trust_remote_code,套件引擎不传,弃用——**零基建改动的选型**。
- 模板无 thinking 开关(InternVL3.5 思考模式需显式 system prompt 开启,
  默认直答),无需 GLM 式 `enable_thinking` 适配。
- MME 图像预算:`max_patches=2`(448² tile × 256 token + 缩略图 =
  768 token),与 Qwen `max_pixels`/GLM `size.longest_edge` 同一预算。
- spec 提交 `b55633af`。冒烟:图像/视频两模态路径均通。

## 预检发现:2B 模型的感知怪癖

12 探针预检(6 色板 × 图像/视频)9/12:把 red 色板稳定叫 **"Pink"**
(图像视频一致)、orange 视频叫 "Brown";且短答案后跟复读垃圾
(`'Pink Pink Pink...'`、`'.tk.tk.tk'`)。预判风险:regime 分歧时
probe 兜底会误伤。按证据优先,先跑真实套件再决定适配。

## 套件两轮:probe 缺口修复(`ca751f56`)

- 第 1 轮 28/29:唯一失败 T0.2 replay,t02-8 hit 路
  `'redTürkiye Türkiye...'` vs miss 路 `'redsquare.me/wp...'`——
  **色词都是 "red",分歧全在答案后的复读垃圾**(miss/hit 两 regime
  数值分歧,GLM 同族现象)。根因:pressure 请求是全套件唯一
  `expected_probe=()` 的图像请求族(eviction/preemption 都带色词
  probe),divergence 无援救只能硬比字节。
- 修复:补上 `image_color_name` probe,与其余请求族同一政策。检测力
  不降:串图会答出**另一张图的颜色** → probe 照样硬失败;计数器裕量
  仍是 T0.2 主检测器。确认过 pass1 不做 probe 强检(check_text 仅
  baseline 路径),改动无新失败面。
- 第 2 轮 **29/29**:3 处 replay 分歧(t02-8 red、t02-33/63 green)
  全被正确色词援救——良性 regime 漂移定性成立。

## parity 两轮:GQA-8 容量错配(`61e23812`)

- 第 1 轮 FAIL,失败形态前所未见:**hit_ratio=0.013**,其余全部完美
  (0 翻转 ×2、分差 0.00、parse 0.9351)。零翻转+零分差 = pass2 全在
  重算而非命中——结构性 miss,非损坏。
- 定因(数据实锤):Qwen3-1.7B 骨干是 **GQA-8**:28 层 × 8 KV 头 ×
  128 维 = **112 KB/token**,是已认证 GQA-2 模型(Qwen 28–36 KB/token)
  的 3-4 倍。全量 MME ≈2374 × ~1000 token × 112KB ≈ **250GB KV**,
  冲爆 parity 运行器 40GB 默认缓存;pass2 按存储序重放,LRU 顺序扫描
  在重访前驱逐每一条 → 命中率归零。机器 2TB 内存(1.6TB 空闲)。
- 修复:新 spec 字段 **`mme_max_local_cpu_gb`**(0 = 默认 40GB),
  穿 certify → benchmark_parity `--max-local-cpu-gb` →
  configure_environment;InternVL 设 280GB。README spec 字段清单同步。
- 第 2 轮 **满分 PASS**:hit_ratio=**1.000**,双向 0/2374 翻转,
  分差 0.00——且是**默认 0.5% 翻转口径**,无需 GLM 式模型级校准。

## 最终认证

`certify.py internvl3.5-2b --parity-report ...`:套件随 HEAD `61e23812`
重跑 29/29,**verdict=SUPPORTED**,证书引用 commit 与全部判据代码一致
(出处纪律:三个改动均先提交再出证)。

## 证据归档(本目录 ../20/ 与本记录同批)

`certificate_internvl3.5-2b.json`、`suite_internvl3.5-2b.xml`、
`parity_internvl3.5-2b.json` + `.baseline.json` + `.answers.json`、
`parity_internvl3.5-2b_40gb_FAIL.json`(40GB 失败对照)——均在
`records/2026/08/20/`。

## 经验沉淀

- **失败指纹三兄弟凑齐了**:GLM 挂翻转(长推理放大数值分歧)、
  InternVL 挂命中率(宽 KV 冲容量)、Qwen 全默认干净过。
  "0 翻转 + 0 分差 + 近零命中"= 容量/驱逐问题;
  "命中正常 + 翻转超标"= 数值 regime 问题。互为对照,归因即查表。
- **InternVL 的 0 翻转反向背书 GLM 口径**:同一套 gate 下短答案模型
  字节稳定,证明 GLM 1.5% 口径确实是"长前言放大"而非 gate 太松。
- **spec 字段的边界又划对了一次**:两个修复都是"检测器口径对齐模型
  风格/资源形态",零 lmcache/ 核心改动——支持第 4 个模型依然只是
  "加一个 spec 条目 + 校准检测器",测试设施的模型无关性成立。
- 预检(12 探针冒烟)物有所值:提前预判了 probe 误伤方向,套件失败
  时 10 分钟内定性,不用从头猜。

## 下一步

- **等用户签收 InternVL3.5-2B**(惯例:每模型用户过目后再进下一个)。
- 待推送(用户发话才推):`1129b2e2` → `755272e1` → `b55633af` →
  `ca751f56` → `61e23812`。
- 后续候选:#2 Qwen3-VL(需 DeepStack 加测套件)、#5 Gemma3(需双向
  注意力加测)、P5 bypass 护栏代码、MP 竞态家族立项。
