# Qwen3.6-27B 认证:SUPPORTED(第 7 个模型,第一个 27B)

日期:2026-08-21(09:45–12:20)
分支:`multi_modal` @ `da537bf0`(未推送)
证书:`certificate_qwen3.6-27b.json`(schema v3)
前置:`9_qwen35_2b_certification_supported.md`(同架构类的 2B 版本)

## 结论

- 合成套件:**26 passed**(1461 s)
- MME 全量 parity(2374 题,MP 路径):**PASS**,两次跑数字**逐位相同**

架构:`Qwen3_5ForConditionalGeneration`——和已认证的 Qwen3.5-2B 是同一个
类,深度 4 倍:64 层 = 16 全注意力(4 KV heads × 256)+ 48 Gated-DeltaNet。
**没有 DeepStack**(`deepstack_visual_indexes` 为空),所以不需要 add-on 套件。
vLLM 0.23.0 原生支持,不需要升级。

引擎初始化实测:统一 block **784 token**,4 个 KV cache group(3× MambaSpec
各 16 层 + 1× FullAttentionSpec 16 层),page size 全部 3.2 MB,按滑窗大小
分桶成 **2 个 object group**。

## 两个"尺寸"缺陷(与缓存无关,但会让套件红)

注册这个模型踩了两条全套件级别的隐含假设,两条都是"小模型才成立"。

### 1. `ISOLATED_GPU_UTILIZATION = 0.35` 装不下权重

27B bf16 权重 52 GB = 一张 H200 的 0.37,隔离场景直接
`No available memory for the cache blocks`。改成 spec 字段
`isolated_gpu_utilization`(本模型 0.75),并写清为什么可以放大:隔离场景
各自起子进程,且 `test_isolated_paths` 在 `test_mm_acceptance` 之前被收集,
此时没有 session 引擎在占卡。

### 2. 混合 MP server 固定 60 GB,压力用例把自己的工作集挤掉了

T0.7 的 resident-key 审计报"expected ~556, found 416"。逐请求打点后看得很
清楚(`mp_...` 证据里的 KEYS 日志):

```
t02-61  new_keys=8    total_keys=500
t02-62  new_keys=-92  total_keys=408   <-- 46 个 block 被驱逐
t02-63  new_keys=8    total_keys=416
```

服务端同一时刻:`L1 memory usage 0.80 above watermark 0.80; triggering
eviction`。**两次跑都停在 416,完全确定性。**

根因是我自己算错了单位:`page_size_bytes = 3211264` 是**每层**每 block,
不是每 block(3211264 / 784 = 4096 B/token = 4 heads × 256 的 K+V)。所以
一个 block 要算上全部 64 层:**约 205 MB,262 KB/token**——是 Qwen3.5-2B
的 **5 倍**,不是我一开始写的"三分之一"。64 张图 × 4 block × 205 MB = 52 GB,
撞上 60 GB 的 0.8 水位线。改成 spec 字段 `mp_server_l1_gb`(本模型 200 GB),
因为这个量在已注册的混合模型之间差两个数量级,一个常数服务不了。

值得记一笔:resident-key 审计本来是抓"KV 被静默丢弃"的,这次抓到的是
**套件把自己的缓存饿死了**。公差放宽就会让它变绿,然后那个 60 GB 常数会
安静地截断以后每一个深层混合模型的工作集。

## flip 预算:量出来的,不是凑出来的

parity 第一次 FAIL,**只超一个 flip**:12/2374 = 0.505% vs 默认 0.5%。
按 GLM-4.6V-Flash 立下的规矩(先量引擎自身的地板,再谈放宽),量了四组
(全部用 gate 的**解析后答案**口径,证据:`flip_calibration_qwen3.6-27b.json`):

| 对比 | raw 文本 | **解析后** |
|---|---|---|
| baseline 重跑(同配置) | 0 | **0**(逐字节相同) |
| 无 LMCache,只改 `max_num_seqs` | 37 | **2**(0.084%) |
| 开缓存未命中 pass vs 纯 vLLM | 8 | **1** |
| 命中 pass vs 未命中 pass | 85 | **12**(0.505%) |

**中途纠错**:我一开始把控制组的 raw(1.56%)拿去和 gate 的解析后
(0.505%)比,得出"连引擎自己的噪声都没到"的结论——口径不一致,错了。
同口径下引擎地板只有 0.084%,命中路径是它的 **6 倍**。85 个 raw 差异里
73 个只是大小写/空格(`" Yes"` vs `" yes"`),gate 忽略它们是对的。

那 12 个 flip 是漂移还是损坏?对 ground truth 一查:

```
12 个 flip 里,未命中 pass 对 7 次,命中 pass 对 5 次
6/12 落在 landmark,其余 artwork / celebrity / scene
```

损坏会是**单边**的(命中 pass 几乎全错),7:5 且集中在"本来就模棱两可"的
识别题上,是漂移的样子;总分只动 +1.0/2179。

而且混合模型本就该比默认预算宽,原因是结构性的:**命中在混合模型上不是
同一个计算**。全注意力的命中装载出来的 KV 和算出来的逐位相同,所以 0.5%
是个紧的界;GDN 的命中是**恢复一页 recurrent state**(前缀的有损摘要),
根本没有"相同的算术"可指望。而且它随线性层深度走:18 层 GDN 的
Qwen3.5-2B flip 0.21%,48 层的这个 flip 0.505%。

所以 `mme_max_flip_fraction=0.01`,证据全部写进 spec 注释;10 分的分数
gate 保留作为真损坏的兜底(实测 1.0 分)。

## Parity 数字

| 项 | 实测 | 预算 |
|---|---|---|
| flips 命中 vs 未命中 | 12 / 2374 = 0.505% | 23.74(1%) |
| flips 未命中 vs 纯 vLLM | 1 / 2374 | — |
| 分数漂移 pass2 − pass1 | 1.0 / 2179 | 10.0 |
| 装载覆盖率 | 1.056 | 0.95 |
| **vLLM 自己缓存服务的 token** | **0** | — |
| baseline 解析率 | 0.9983 | 0.90 |

两次 parity 跑出的 12/1/1.00/0.75/1.056 完全一致——确定性分歧,不是竞态。

## 未完成

1. **覆盖率 1.056 > 1**:`achievable_hit_tokens` 的分母偏小。可能是 pass 1
   连 decode token 一起存了,于是 pass 2 能命中超出"prompt 最后一个整
   block"的部分。gate 是 ≥0.95,超出无害,但这个分母该收紧,不然它某天
   会掩盖真的覆盖不足。
2. 记录待补:`8_` 里的两个 MP 家族课题(retrieve 完成延迟根因、心跳连续
   失败判死)仍未动。
3. 下一步按 `2_model_priority_revision_aug2026.md`:vLLM 升级批
   (Qwen3.8-27B + Gemma 4,后者被 `5_` 的 vLLM 0.23 缺口挡着)。升级 venv
   是这批的前置。
