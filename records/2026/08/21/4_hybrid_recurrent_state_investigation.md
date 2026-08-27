# 混合注意力/recurrent state 立项调研:缺口是"多模态×混合",不是混合本身

日期:2026-08-21
分支:`multi_modal` @ `0c911591`
前置记录:`2_model_priority_revision_aug2026.md`(修订顺序第 2 项)、
`3_qwen3vl_2b_deepstack_certification_supported.md`

## 核心发现:主线已把"混合模型 KV 缓存"做完了

`docs/source/mp/hybrid_models.rst`(298 行,含验证流程)+
`lmcache/integration/vllm/kv_cache_groups.py` + `.claude/skills/
hybrid-benchmarking`(已在 Qwen3.6-27B/GDN 上验证过性能阶梯):

- **支持面**:Qwen3.5/3.6 全系、Qwen3-Next、Kimi-Linear、**Kimi K3**
  ——修订顺序里"K3 单独立项"和"recurrent state 立项"的核心已由主线
  交付。GDN 层的 conv+SSM state 被注册为**不透明页**,按 align 模式
  (vLLM `--mamba-cache-mode align`,仅在 block 边界快照 state)当作
  "单块滑窗"处理(`_is_mamba_align_spec`:命中长度 L 只需最后一块在)。
- **约束**:统一 block size N 模型相关(vLLM 启动日志打印;
  Qwen3.5-0.8B=544,Qwen3.6-27B=784,K3=768);LMCache chunk_size
  必须 = N 的倍数;server 需 `--separate-object-groups`;
  `--max-num-batched-tokens ≥ N`。
- **仅 MP connector 路径**:`LMCacheMPConnector` 向 vLLM 声明混合
  支持;进程内 `LMCacheConnectorV1`/`vllm_v1_adapter` 零 mamba/hybrid
  代码。
- **非位精确**:GDN 后端不支持 batch-invariant 模式,缓存/新算两 run
  只能做分数级比较(文档明说,CI 用 `hma_lm_eval` gsm8k 分数对齐)。

## 缺口(文档原文):多模态未验证

hybrid_models.rst caveat:"Several of these models are
**vision-language** … The validated, supported path is **text** KV
caching; **image/video KV caching is not validated**."
—— 这正是本项目(多模态支持认证)的地盘。

## 模型侧事实(HF config 核查)

- Qwen3.5-2B/4B/0.8B、Qwen3.6-27B 全是 `Qwen3_5ForConditionalGeneration`
  原生 VL(视觉塔标配)。
- Qwen3.5-2B:24 层 = 6 全注意力 + 18 GDN;kv_heads=2, head_dim=256。
- **Qwen3.5/3.6 砍掉了 DeepStack**(`deepstack_visual_indexes: []`,
  2B 与 27B 皆空)——DeepStack 是 Qwen3-VL 一代的特性,我们刚好在
  它退场前完成了验证(TD 套件对 Qwen3-VL 家族仍有效)。
- 本地 vLLM 0.23:有 `qwen3_5.py` 实现 + `mamba_cache_mode`
  (默认 "none",align 可用)——**不需要升级 venv** 即可做
  Qwen3.5-2B 的 mm×hybrid 验证(Qwen3.8 才需要升级)。

## 立项定义:mm×hybrid 认证(Qwen3.5-2B)

待回答的问题:
1. **mm_hash keying 在 MP 路径 × 混合模型上是否成立**:chunk=N
   (544+)时整张图(~196 token)常落在单个 block 内;图像身份必须
   进 block hash,否则同形不同图在 block 粒度上碰撞。
2. **GDN state 页的内容正确性**:block 边界的 state 快照汇总了含
   图像 token 在内的全部历史;key 不含 mm 身份 ⇒ 串图,且 state 页
  "不透明",一旦串了没有任何 KV 级征兆——检测只能靠输出。
3. 套件适配:基础矩阵搬到 MPHarness(现在只有 T3 在 MP 路径)、
   chunk_size 每模型化(16 → N)、字节 replay oracle 降级为
   探针/分数级(非位精确)。

## 可行性实探结果(qwen35_mm_hybrid_probe.json,一次通过全绿)

Qwen3.5-2B,N=544,MP server(chunk=544 + separate-object-groups)+
`LMCacheMPConnector` + `mamba_cache_mode=align` + prefix caching
(两 pass 间 `reset_prefix_cache()` 只清 vLLM 本地缓存):

- pad 400 词把图像 span 推过第一个 block 边界(总提示 639 token);
- A 首跑 0 命中(干净 miss);**同 pad 异图 B 零假命中**——mm 身份
  确实进了混合路径的 block key(identifiers 记录到完整 mm_hash,
  P0 keying + P2 MP 修复在混合路径自然生效);
- A 重放命中恰好 544(一个 block,存储 4 keys = 2 请求 × 2 对象组
  [全注意力 KV + GDN state]),答案探针正确(red/blue)。

结论:**上游标注"未验证"的 mm×hybrid 基本行为实测正确,走"适配
套件后认证"路线**,无需先修核心。

## 立项:e2e_mm 套件"混合模式" + Qwen3.5-2B 认证(执行中)

套件适配点(实探已验证各机制可行):
1. spec 增加混合声明(N、align、prefix-caching-on);混合模型的
   harness 走 MPHarness(进程内 connector 无混合支持),server
   chunk=N + separate-object-groups;
2. 所有请求注入 ≥N token 的 pad 前缀(chunk=N 下短提示凑不满一个
   block,什么都缓存不了);断言公差从 CHUNK=16 换成每模型 chunk;
3. 每次 run 前 reset vLLM 本地前缀缓存(align 强制 prefix caching,
   不清则 LMCache 计数器失真);
4. 字节 replay oracle 降级:GDN 非位精确,靠探针/提取答案援救
   (GLM 先例);T0.7 存储守恒改为组感知(每 block 2 对象);
5. MME parity 同样需要 MP 化(现在的 runner 是进程内 connector)。
