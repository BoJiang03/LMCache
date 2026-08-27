# records 本地化策略 + MME gate 补强 + T0.7 存储守恒

日期:2026-08-19
分支:`multi_modal`(commits `4cb81d2c`、`8cf37903`,重写后基线 `977fdf19`);`multi_modal_repro` @ `23f1bc64`
前置记录:`3_mm_fix_acceptance_suite_benchmark.md`

## 会话内容

1. 全量 MME 出分;2. 首次推送 fork 后确立"records 本地化"硬规则并完成清理;3. 讨论"MME 双 pass 是否足以作为验证机制",补强 gate;4. 针对"怕 LMCache 漏 KV"实现 T0.7 存储守恒层。

## 全量 MME 结果(Qwen2.5-VL-3B,2374 题)

**PASS,最强形式**:baseline / pass1(miss)/ pass2(hit)三组分数逐分一致——感知 1586.55、认知 613.21、总分 2199.76,14 类别全部相同;0 翻转(两个对比方向都是);pass2 命中率 1.000。绝对分与该模型公开水平相符。报告:scratchpad `mme_full.json`。

## records 本地化(硬规则,已入持久记忆)

用户规则:**records/ 只能存在于本地,任何 remote 不得含有该目录**。当时两分支已推 fork(带 4 个 records 提交),处理:

- 本地重写:`multi_modal` = dev 基线 + fix + suite + benchmark 三个纯代码提交(records 提交摘除);`multi_modal_repro` = 其上 + repro 提交。强推覆盖 fork。
- records 文件保留本地,改 untracked,写入 `.git/info/exclude`(`/home/bo/LMCache/.git/info/exclude`,worktree 共享)。
- 装 `pre-push` 钩子(`/home/bo/LMCache/.git/hooks/pre-push`):任何历史碰过 records/ 的 ref 一律拒推,dry-run 实测拦截成功。
- 全 fork 扫描确认 0 个 records 提交可达;旧提交在 GitHub 悬空对象里短暂存在过(知 SHA 可见,直至 GC),已向用户说明。
- **本策略下 /records 的记录文件不再入 git**(本文件即 untracked),只提交代码状态。

## MME gate 补强(`4cb81d2c`)

用户问"MME 双 pass 是否就是最有效的验证机制"。分析发现原 gate 只查 pass2-vs-pass1 + 命中率——**而串图污染发生在冷 pass,重放是确定性的,pass2-vs-pass1 恒为 0**,旧 #3301 会 PASS。修复:`flips_pass1_vs_baseline` 和 |pass1-baseline 分差| 纳入同阈值 gate(五条件)。定位共识:MME = 认证层(统计可见的质量损失),合成套件 = 抓虫层(单发、确定性、可定位;计数器不变量抓单次假命中不依赖数量堆积)。

## T0.7 存储守恒(`8cf37903`)

用户担心"LMCache 漏了 kvcache"(漏存/丢失)。实现双信号对账,接入 T0.2 压力测试:

- **store 意图**:`lmcache:num_stored_tokens`(`on_store_request` 时累加,即 masked 待存 token 数),同 lookup 的防偷差分(Prometheus 累计 + 未清 interval),`harness.cumulative_stored_tokens()`。
- **驻留真值**:直接内省 `engine.storage_manager.storage_backends["LocalCPUBackend"]` 的 `hot_cache`(持 `cpu_lock`),数 key 数 + `get_size()` 字节和,`harness.storage_snapshot()`。
- 断言:pass1 miss 的 token 必须被 store-request 且落为驻留 key(缺口 = 漏存);pass2 全命中重放近零新存(超 = key 不稳定重复存,lookup 侧不可见的浪费型 bug)、驻留 key 不许减少(减 = 容量未满时条目消失,对应 eviction 误触发/pin bug 类丢失)、字节增长跟随 key(不跟 = 泄漏)。
- 容差:chunk 对齐 ±n×CHUNK、decode 预算 Σmax_tokens、key 数 ±n~2n。
- **边界**:hot_cache 快照量的是条目逻辑字节;allocator 层泄漏(buffer 未归还但条目已删,即 "Double unpin" 1802 条告警那类)看不见——那需要读 memory_allocator 内部状态,留给 pin bug 立项时做。条目误删的后果已被"key 不许减少"兜住。

GPU 验证已过:压力(N=64)+ 隔离两用例 2 passed(2m33s,日志 scratchpad `t07_run.log`),`8cf37903` 已推 fork。

## 下一步

- P2:`_0180`/`_0201` MP connector 补 MM 处理(生产主路径,T3 铺开的前置)。
- 横向:Qwen2-VL-2B spec 已就绪可直接跑;MME parity 新 gate 下全量数据已验证仍 PASS(翻转均为 0)。
- 单独立项:pin count 负数 bug + allocator 层守恒对账。
