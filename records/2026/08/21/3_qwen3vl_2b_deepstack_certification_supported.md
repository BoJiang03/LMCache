# Qwen3-VL-2B 认证:SUPPORTED(第 5 个模型,P3 DeepStack 验证闭环)

日期:2026-08-21(03:00–04:15)
分支:`multi_modal` @ `0c911591`(本地提交,未推送)
前置记录:`2_model_priority_revision_aug2026.md`
用户指令:"继续按照顺序推进。你不要问我意见……自己推进"

## 意义:第一个不能只靠 spec 条目过关的模型

Qwen3-VL 是 P3(DeepStack 验证)的载体:视觉塔多尺度特征
(ViT 层 5/11/17)经 **paged KV 之外的逐 step side buffer** 注入
LLM 层 0-2。风险路径 = 命中边界落在图像 span **中间**:vLLM 从
span 内恢复 prefill,payload 必须按正确偏移散射。`extra_suites`
机制(specs.py 预留的字段)由此首次落地。

## P3 验证结论(实证)

**天然安全**:注入效果烘焙进存储的 KV;被跳过的前缀不需要 side
buffer,恢复段由 vLLM 正常重注入。证据(`deepstack_resume_kv_*.json`):
span 中间恢复重存的 KV vs 原始全 prefill KV,相对 Frobenius 距离
0.02–0.04(bf16 recompute regime 噪声级);致盲对照(注入清零)
爆到 0.55–0.70,**15–25 倍分离**。

## 关键发现:输出 oracle 对 DeepStack 故障全盲

预检(`deepstack_sensitivity_*.json`):**完全关闭注入,12 探针
答案零字节变化**(logprob 差仅 ~1e-5),而注入特征三级范数高达
248/304/374(正对照证明注入活跃)。因此加测套件放弃输出比较,
改用 **KV 级 oracle**:
1. 外科驱逐已存请求的尾部 chunk(`harness.evict_resident_keys`,
   新公共辅助),replay 命中恰好 = 保留 chunk×16,强制 span 内恢复;
2. 重存 KV vs 原始 KV 克隆逐 chunk 比较,阈值双向 >3 倍裕量
   (NORMAL≤0.15,BLIND≥0.30);
3. TD.4 负对照在套件内常驻,oracle 失灵即红灯。

`test_deepstack.py` TD.1–TD.4:单图两切深、双图第二 span、视频
span、负对照,5 用例(TD.1 参数化 ×2)。

## 踩坑:pytest 文件序打破 GPU 独占假设

套件第 1 轮 31/34:三个隔离场景子进程崩
"Duplicated timeseries in CollectorRegistry"。根因与模型无关——
`test_deepstack.py` 字母序排在 `test_isolated_paths.py` 前,session
引擎(gpu 0.6)先建且贯穿全程,隔离子进程再要 0.6 挤爆,引擎重试
初始化时撞出指标重复注册。以前"隔离场景先跑、独占 GPU"纯属文件名
巧合。修复(`0c911591`):conftest 显式稳定排序,无 session 引擎
的测试一律先跑。手跑单场景通过 + 顺序对照(昨日 InternVL XML)
实锤归因。

## 认证结果

- 套件第 2 轮 **34/34**(29 基础 + 5 DeepStack),唯一 replay 分歧
  是空格级('red' vs ' red'),探针援救。
- MME parity **满分**:hit_ratio=1.000,双向 0/2374 翻转,分差
  0.00,parse 0.9979(历史最佳),**默认 0.5% 口径**,280GB 容量
  (GQA-8 112KB/token,同 InternVL)。
- `certify.py --parity-report` → **verdict=SUPPORTED**,证书 commit
  = HEAD `0c911591`,出处一致。

## 提交(本地,未推送;fork 仍在 a3c6a2c3)

- `2c153ea8` DeepStack 加测套件(requires_extra_suite 机制 +
  harness 外科驱逐辅助 + test_deepstack.py + README)
- `847da467` Qwen3-VL-2B spec 注册(新式 size 处理器 kwarg:
  786432px≈768 token;280GB parity 容量)
- `0c911591` 测试排序修复(引擎子进程测试先行)

## 证据归档(本目录)

certificate / suite xml / parity(+baseline/answers)/ precheck /
deepstack_sensitivity / deepstack_resume_kv,均带 `qwen3-vl-2b` 名。

## 经验沉淀

- **"探针测不出的故障类,用状态级 oracle"**:输出不敏感 ≠ 无法检测;
  KV 直比 + 自校准负对照给了 15-25 倍信噪比。该模式可复用于 Gemma 4
  双向注意力验证(比较 KV 而非答案)。
- **隐式顺序依赖会被新文件名引爆**:资源互斥(GPU 独占)必须显式
  编码进 collection 排序,不能靠字母序。
- 12 探针预检 + 灵敏度实验前置,让加测套件第一版就带对了 oracle,
  没有走"输出比较假绿"的弯路。
- Qwen3-VL-2B 感知质量迄今最佳:12/12 探针全对,parse 0.9979。

## 下一步

- 待推送(用户发话才推):`1129b2e2`→`755272e1`→`b55633af`→
  `ca751f56`→`61e23812`→`2c153ea8`→`847da467`→`0c911591`
- 修订顺序的下一项:**混合注意力/recurrent state 立项调研**
  (Qwen3.5/3.6/3.8 + Kimi K3;先摸 LMCache hybrid allocator 现状)
- 之后:Gemma 4-E4B(双向注意力,KV oracle 模式复用)
