# Qwen3.5-2B(Mamba/GDN 混合)认证:SUPPORTED

日期:2026-08-21(09:00–09:45)
分支:`multi_modal` @ `2a522b33`(未推送)
证书:`certificate_qwen3.5-2b.json`(schema v3)
前置:`6_hybrid_suite_adaptation.md`(套件改造)、
`7_hit_provenance_oracle.md`(命中来源校验)、
`8_mp_retrieve_latency_and_hybrid_load_failure.md`(MP 延迟与装载失败)

## 结论

第 6 个通过认证的模型,也是**第一个 Mamba/GDN 混合模型**,第一个只在 MP
路径上认证的模型。

- 合成套件:**26 passed / 0 failed / 0 error / 0 skipped**(1724 s)
- MME 全量 parity(2374 题,MP 路径):**PASS**

## Parity 数字

| 项 | 实测 | 预算 |
|---|---|---|
| flips:pass1(开缓存未命中)vs 纯 vLLM | **0 / 2374** | — |
| flips:pass2(命中)vs pass1 | 5 / 2374 = 0.21% | 11.87(0.5%) |
| 分数漂移 pass2 − pass1 | 8.0 / 2179.4 = 0.37% | 10.0 |
| 装载覆盖率 | **1.000**(582624/582624) | 0.95 |
| **vLLM 自己缓存服务的 token** | **0** | — |
| baseline 答案解析率 | 1.000 | 0.90 |

MME 总分 2179.4(perception 1613.33 / cognition 566.07),pass2 2187.4。

最后两行是这次最想拿到的:全基准尺度上**每一个命中 token 都真的走了
connector 的 retrieve**(`external_cached=582624`,`local_cached=0`)。
`7_` 建的来源校验不只在合成套件里成立,在 2374 题上也成立,所以覆盖率
这个数字说的就是它字面的意思。

混合模型无法 bit-exact(GDN 内核没有 batch-invariant 模式),所以这里的
判据是 flip/分数预算,不是字节相等——证书里明确写着。

## 认证范围(证书 scope)

- 部署路径:**只有** `LMCacheMPConnector + MP cache server`(单 GPU,TP=1)
- 模态:image、video
- 缓存粒度:**544 token**(vLLM 统一 block,align 模式),不是 16
- 后端:MP cache server L1,separate object groups
- 调度:分块 prefill(混合模型天然覆盖:一个 scheduler step 前进一个统一
  block)、并发批

## 明确不覆盖(混合特有的三条 + 本次新增一条)

1. 进程内 `LMCacheConnectorV1` 路径——引擎初始化就失败;
2. 容量驱逐与抢占重算——两个隔离场景都跑进程内 connector;
3. bit-exact 生成;
4. **装载失败的恢复(connector 的降级模式)**:vLLM 用
   `_update_requests_with_invalid_blocks` 回卷,那里按单 KV group 解包,
   混合模型上直接抛异常。所以这条路上一次装载失败就是引擎致命,不是可
   恢复的。详见 `8_`。

## 过程里的关键判断

压力用例最初是红的,而且红得像"缓存键异常"。查下来是 vLLM 自己崩,根因
是上游对混合模型的既有缺口 + 一次超时的心跳(`8_`)。当时有两个选择:把
用例改宽松让它绿,或者把触发条件(误判服务端死亡)拿掉、把真限制写进
证书。选了后者——套件的心跳窗口从 10s 放到 60s(服务端 reap 300s 配套),
断言一条没动,证书新增一条明确排除。

这也是这次认证最值钱的部分:如果没有 `7_` 的来源校验,这个模型会以
"26 passed + 高命中率"的姿态拿到证书,而实际上命中断言半数是真空的、
retrieve 路径从没跑过、装载失败在混合上是致命的这件事也不会被发现。

## 产物

`records/2026/08/21/` 下:`certificate_qwen3.5-2b.json`、
`parity_qwen3.5-2b.json`、`suite_qwen3.5-2b.xml`,以及 `8_` 的四份延迟/
对照证据。

## 下一步

按 `2_model_priority_revision_aug2026.md` 的顺序:Qwen3.6-27B,然后是
vLLM 升级批(Qwen3.8-27B + Gemma 4,后者被 `5_` 记录的 vLLM 0.23 缺口
挡着)。`8_` 里留的两个 MP 家族课题(retrieve 完成延迟根因、心跳应改成
连续失败判死)不进认证流程,单独排。
