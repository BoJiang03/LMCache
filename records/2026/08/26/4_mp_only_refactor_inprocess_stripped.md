# 全面转 MP:in-process 从套件和 PR 里剔干净,deepstack 子套件随之失守

**日期**: 2026-08-26(当天第 4 篇,接 `3_`)
**代码状态**: `multi_modal@9c5a7d0f`,工作树干净(本篇 4 个提交,见 §二)
**归档分支**: `archive/e2e_mm-inprocess-and-mp` → `48314ad6`(**仅本地,未推**)

## 结论先写

1. **`tests/e2e_mm` 现在只有 MP 一条路。** `MPHarness` 折进 `MMHarness`,
   `DeploymentPath` / `selected_deployment_path` / `LMCACHE_MM_E2E_PATH` 全删,
   in-process 的缓存直读辅助(`_local_cpu_backend`、`storage_snapshot`、
   `resident_chunk_keys`、`clone_resident_kv`、`evict_resident_keys`、
   `resident_kv_tensor`)与 `LMCStatsMonitor` 读数(`cumulative_lookup_stats`、
   `cumulative_stored_tokens`)一起删。`harness.py` 净减 ~410 行。

2. **`test_deepstack.py` 删除 —— 这就是"某些模型不支持"的那一项,如实记在这里。**
   TD.1–TD.4 唯一敏感的 oracle 是"把已存 KV 读回来和续接后重存的 KV 比
   rel-Frobenius",MP 服务端**没有等价接口**:
   - `GET /cache/objects` 只支持 L2(`adapter.list_l2_keys`),测试服务端是 L1-only;
   - `POST /cache/checksums` 算的是 **GPU 块的 MD5**,而重算噪声本就不是逐位相同
     (实测 rel-Fro 0.02–0.04),MD5 永远不等,做不了"相同"判据;
   - `DELETE /cache/objects` 倒是按 key 支持 l1 —— 但"切"能做、"读回来量差多少"不能做。

   而输出型 oracle 在这个故障类上**实测全盲**(2026-08-21:完全关掉 deepstack 注入,
   合成探针一个输出字节都不变)。所以没有"弱一点的替代",硬留一个测不出东西的检查
   比删掉更坏。spec 的 `extra_suites={"deepstack"}` **保留**(它是模型属性),
   由 `certify.DEEPSTACK_NOT_COVERED` 出具排除项 —— 缺口写在每张证书上,不跟着文件消失。

3. **产品侧只剩 MP。** 按用户"in process 不在这次 PR 考虑内"的指示,
   `lmcache/integration/vllm/vllm_v1_adapter.py` 里那个抢占场景的 in-process 修复
   已 revert 回 dev。PR 的 `lmcache/` 改动现在是:MP 连接器(含两个版本钉死副本)、
   共享的 `utils.py` keying、服务端(l1_manager / storage_manager / l1_failures)。
   相关单测 **50 passed**(`test_mm_hash_utils.py` + `test_mp_connector_mm_keys.py`
   + `test_vllm_mp_adapter.py`)。

4. **本地 9 张证书全部作废。** schema 4 → **5**。≤4 的证书是双路径套件出的:
   `scope` 声明了套件已不再驱动的部署,`known_not_covered` 也没有 in-process 与
   deepstack 两条排除。**要重测,不是改标签。**(证书是 git-excluded 的本地产物,
   仓库里没有过期声明。)

5. **验证已出:0.23.0 上两个模型各 29/29 全绿。**
   qwen2-vl-2b 29 passed / 3 deselected(16:43),qwen3-vl-2b 29 passed /
   3 deselected(15:32),零失败零错误。日志归档
   `records/2026/08/26/mponly/`。qwen3-vl-2b 改造前是 **34**(deepstack 的
   4 个测试函数展开成 5 个用例),现在 29 —— 差的正是删掉的那 5 个,
   其余用例一个没丢。

## 一、用户决定(本轮)

1. 先把 in-process/MP 混合实现存归档分支,再剔干净、全面转 MP。
2. 跑不了的子套件**如实报告,不硬凑**。
3. in-process 不在本次 PR 考虑内,连那个 in-process 修复也去掉。

## 二、四个提交

| 提交 | 内容 | 规模 |
|---|---|---|
| `4e749b5a` | 剔除 in-process 部署路径,证书只声明 MP | 9 文件,+423 / −1110 |
| `01e6f317` | 证书 schema → 5;顺带删合并后失效的私有属性 | 3 文件 |
| `0badaae0` | 设计文档 `multimodal_cache_keying.md` 的验证段同步 | 1 文件 |
| `9c5a7d0f` | revert in-process 连接器的抢占 token 修复 | 1 文件 |

`4e749b5a` 逐文件:

- **harness.py**(−410 净):合并两个 harness 类;`configure_environment()`
  不再设任何 `LMCACHE_*` 引擎变量(MP 服务端从命令行取 chunk / L1,**实测不读**
  这些 env —— `lmcache/v1/multiprocess/config.py` 的 `getenv` 只碰 coordinator 系列);
  只保留 `VLLM_ENABLE_V1_MULTIPROCESSING=0`,理由改写为"计数器包装在本进程装,
  spawn 出去的 worker 看不到",不再是"stats 单例可读"。
- **conftest.py / isolated_cases.py**:删选路分支;`_deployment_harness` →
  `_scenario_harness`(现在每个场景都起自己的服务端);eviction 容量归一到
  `EVICTION_CAPACITY_GB = 0.0625`(服务端分配器 64 MB 单位,原 in-process 的 0.05 删掉)。
- **benchmark_parity.py**:恒起服务端、恒用 `MPTransportCounters`、
  `deployment_path` 恒为 `"mp"`;`achievable_hit_tokens` 的分母现在总是有,
  `None` 只留给老报告。
- **certify.py**:`certified_scope` 只出一条路径;新增 `IN_PROCESS_NOT_COVERED`
  (每张证书都带)与 `DEEPSTACK_NOT_COVERED`(声明 deepstack 的模型带);
  parity 报告的路径校验从"匹配所选路径"改成"必须是 mp,否则要求重跑"。
- **README.md / isolated_routing.py / specs.py**:同步措辞与理由。

## 三、T3 的身份变了(保留,但记一笔)

`mp_connector` 场景原本的意义是"另一条部署路径"。全面转 MP 后它不再是第二条路,
现在是**冷启动重放**:自己的服务端(L1 没见过任何东西)+ 自己的引擎(隔离 GPU 占比)。
保留的理由是这两点仍是别处没有的组合;但它与会话套件的重叠比以前大,
**是否值得每个模型多跑 ~4 分钟,留给后续决定**。README 的 T3 段已按新身份改写。

## 四、判据(已完成的)

| 检查 | 结果 |
|---|---|
| `ruff check` / `ruff format --check` tests/e2e_mm | 全过 |
| 12 个注册模型的 `certified_scope` / `known_not_covered` | 全部只出 MP 一条路径;只有 qwen3-vl-2b 多一条 deepstack 排除 |
| qwen2-vl-2b / qwen3-vl-2b 收集 | 各 29 collected / 3 deselected(与改造前同数) |
| MM keying 单测(revert 之后) | 50 passed |
| **qwen2-vl-2b 完整套件**(0.23.0,MP-only) | **29 passed / 3 deselected**,16:43 |
| **qwen3-vl-2b 完整套件**(0.23.0,MP-only) | **29 passed / 3 deselected**,15:32 |

## 五、教训

1. **"剔干净"必须连理由一起改写,否则留下的注释会变成假话。**
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` 的注释原本写"让 LMCStatsMonitor 共享" ——
   监视器已经删了,标志却仍然必须设(计数器包装在本进程)。同一行代码、
   完全不同的理由;只删代码不改注释,下一个人会以为它可以删。
2. **删掉一个测不出东西的检查,好过留着它当覆盖。** deepstack 的输出型 oracle
   是实测全盲的;把 TD.1–TD.4 改写成输出比对能让套件继续全绿,但那张绿是假的。
   删除 + 在证书上写排除项,是唯一诚实的处置。
3. **"是模型属性"的声明不要跟着实现删。** `extra_suites={"deepstack"}` 留着,
   缺口才有人认领;删了它,证书就只是少了一行,而少一行读起来跟"没有这个限制"
   一模一样 —— 这正是 certify 里 `CHUNKED_PREFILL` 那条注释记的旧坑。
4. **归档要在动手之前做。** 先建 `archive/e2e_mm-inprocess-and-mp` 再改,
   删掉的 320 行 deepstack 与 in-process 外科手术辅助随时可取回。

## 六、诚实边界

1. **两个绿是在 verify 树 `b4a06dec` 上量的,不是 `9c5a7d0f`。** 差异两项,
   都核对过:
   - verify 树多带三个 base 修复(`lmcache_driven_transfer` 的写预留回滚、
     native object group 的符号组闸门、fallback memcpy 的流序)。这三个
     **确实被执行到**,所以这两张绿是"multi_modal + 三个 base 修复"的绿,
     不是 multi_modal 单独的绿。三个修复本身要单独走上游。
   - verify 树里 `vllm_v1_adapter.py` 还带着已被 revert 的 in-process 修复。
     **不影响这两张绿**:该改动在 `LMCacheConnectorV1Impl`(第 1421 行),
     MP 路径的 mm keying 走的是 `lmcache_mp_metadata.py:88` 与两个版本钉死
     副本,`lmcache_mp_connector.py` 根本不引用 `vllm_v1_adapter`。
   - GPU5 / GPU7,`vllm-lazy`(0.23.0)+ pyguard sitecustomize。
2. **0.27.1 上没跑本次改造**。已知 `preemption` 在 0.27.1 + 任何 `is_kv_consumer`
   连接器上会活锁(`3_`:vLLM `defer_block_free`),与本次改造无关但会污染结果。
3. **其余 10 个模型未测**;12 张证书要按 MP 口径全部重出(小时级/模型)。
4. **MME parity 报告要重跑**:非混合模型的既有报告记的是 `in_process`,
   新的 certify 会直接拒收。
5. 本次没跑 `certify.py` 端到端(要几小时),只逐模型验证了它的 scope/排除项函数。
6. verify 工作树被我重建过(原来的未提交调试改动 —— 含一次性的
   `LMCACHE_MM_E2E_PREEMPT_BLOCKS` 钩子 —— 已存成
   scratchpad `verify_tree_wip_before_mponly.diff` 后丢弃)。

## 七、下一步

1. ~~等两个套件跑完,如实记结果~~ —— 已出,各 29/29(见 §结论 5)。
2. 0.27.1 上按 MP 口径重跑(preemption 的红预期属上游 `defer_block_free`)。
3. 12 个模型重认证 + MME parity 重跑;证书 schema 应再加 vLLM 版本字段(仍未加)。
4. deepstack 若要恢复:需要服务端先有"读回已存对象"或"服务端算 KV 距离"的 API,
   那是独立的一件事,不在本 PR。
5. 上游:`defer_block_free` 活锁的最小复现(去掉 LMCache)+ 给 #4463 补一条
   "我们独立复现 + MP 不受影响"。
