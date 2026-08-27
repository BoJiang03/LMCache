# 0.27.1 在 MP 口径下基本可用:三个红逐个归属,融合腐坏只坏 in-process

**日期**: 2026-08-26(当天第 1 篇,接 08-25 的 `7_`)
**代码状态**: `multi_modal@48314ad6`(两个新提交:`9177e0b1` 部署路径开关、
`48314ad6` 容量档位与 chunked prefill 选路),工作树干净。
证据归档 `records/2026/08/26/vllm_0271_mp/`(8 项)。

## 结论先写

1. **0.26+ 的 KV 融合布局腐坏只坏 in-process 路径,MP 不受影响。**
   上游 issue **#4463**(2026-08-08,OPEN)与修复 PR **#4467**(08-09,OPEN,
   REVIEW_REQUIRED,此后未动,作者无 CUDA 机器、E2E 全 pending)都写明
   "MP connector is unaffected"(其融合页几何是显式的 `kv_size=1, hs=CS`,
   直接拷 packed 行)。PR 给的根因比我们 08-25 挖到的更深一层:V2 用真 split-KV
   `KV_2LTD [2,L,T,NH*HS]`,而 `multi_layer_kv_transfer` 把融合缓冲当 packed
   `[1,L,T,NH*CS]`,slot stride 取了一半行宽 → slot 混叠、V 平面回不来;
   HND 分支又把 `head_size` 多乘了一次 2。**这解释了 08-25 记录 1 那个
   "强制 HND 换了检测格式、坏输出仍 16/16 逐字相同"** —— 病灶在传输寻址,
   不在格式选择。
   我们自己的数据吻合:同一棵带 memcpy 修复的树、同一个 0.27.1 venv,
   in-process 文本探针 hit 准确率 **0.0**(每命中装载 448 token,valid=true),
   而 MP 侧全绿(见 2)。

2. **0.27.1 上 MP 路径的实测**(verify 树 = multi_modal + 三个 base 修复
   cherry-pick + 守卫;全部带 pyguard):

   | 跑什么 | 路径 | 结果 |
   |---|---|---|
   | qwen2-vl-2b `mp_connector`(T3) | 非混合 → MP | PASSED (3:59) |
   | qwen3.5-2b 完整套件 | GDN 混合 → 全程 MP | **27/27** (15:32) |
   | qwen2-vl-2b 完整套件(强制 MP) | 非混合 → MP | 26 passed / 3 failed |
   | 同上,修完两处后重跑 `chunked_prefill`+`capacity_eviction` | MP | **2 passed** (5:12) |
   | qwen2-vl-2b 完整套件(强制 MP) | **0.23.0 对照** | **29/29** (18:51) |

   对照(0.27.1 无修复的旧树)是 qwen3.5-2b 4 failed / 23 passed —— 那 4 个
   内容分歧失败是 08-25 修掉的 base 坑,不是版本问题。

3. **三个红的归属(全部不是 0.27.1 的锅,除了第三个)**:

   | 失败 | 真实原因 | 归属 |
   |---|---|---|
   | `chunked_prefill` | 该场景**直接 new `MMHarness`**,根本没走选路 → 跑在 in-process 上,红的是 0.26+ 融合腐坏 | LMCache(#4463/#4467),与 MP 无关 |
   | `capacity_eviction` | 容量常量分 in-process `0.05 GB` / MP `0.0625 GB`(MP 分配器不给不足一个单位的容量),选档按"是不是混合模型";非混合上 MP 后拿 0.05 GB 的断言去量 64 MiB 的池 | **我的开关的 bug**,已修 |
   | `preemption` | 0.27.1 + MP **活锁**:引擎起来后 3 分钟刷 13,544 条 `<preempted> by preempted requests`,集合一个字不变(6 个请求每步全被抢占) | 待定,见 4 |

4. **preemption 活锁**(归属已两次修订,最终见 `3_`:**属 vLLM 0.27.1 的
   `defer_block_free`**,LMCache 只是触发它的 `is_kv_consumer` 连接器):

   | 实验 | 抢占数 | 结果 |
   |---|---|---|
   | 0.23.0 + LMCache MP,128 块 | 2 | 通过 |
   | 0.27.1 + LMCache MP,128 块 | 每步全抢占 ×13,544 | **活锁** |
   | 0.27.1 + LMCache MP,256 块 | **0** | 批次干净跑完(场景判 vacuous) |
   | **0.27.1 裸 vLLM,128 块,六请求一批** | **2** | **6/6 输出,2.8 秒** |
   | **0.23.0 裸 vLLM,128 块,六请求一批** | **2** | **6/6 输出,25.8 秒** |

   ~~裸 vLLM 在两个版本上都能抢占并恢复 → **不是 vLLM 回归**~~ ——
   这条推断也不成立:裸 vLLM 不带 `kv_transfer_config`,`defer_block_free`
   恒为假,对照组等于把待判开关一起关了(见 `3_`)。仍然成立的是:
   不是"阈值挪了一点",池加大到不触发抢占就一切正常,一旦真的抢占就回不来。
   定位线索:MP 服务端日志在 07:27:33 收完最后一笔 store 后**整段静默**,
   引擎却空转了三分多钟 —— 卡在引擎侧的调度门(连接器的异步 lookup 门控
   最可疑:被抢占的请求重排时 `check_lookup_result` 若永不返回,vLLM 会
   一直推迟它们),不是服务端不响应。**确切机制未验证(需连接器侧插桩)。**

   **更正**:本篇早先写的"裸 vLLM baseline 先跑完了,所以 vLLM 没问题"
   这条推断**不成立** —— `baseline_runner.py` 是一个一个请求顺序跑的,
   根本不产生批次压力、从不触发抢占。上表第 4/5 行才是真对照。

5. **上游 dev 没有修这个问题**:`origin/dev` 今天 `35d68819`,比我们测过的
   `c1ef01b9` 只多 3 个提交(MP 补漏注册 #4709、CLI query coordinator、
   CI nvcc 线程),`kv_format/detection.py` 与 `specs/nl_x_nb_bs_nh_cs.py`
   最后一次改动是 08-12 的重构 #4473。

6. **0.28 nightly 另有独立阻塞**:vllm#51718(08-22 起的 nightly)删了
   `get_kv_cache_layout`,LMCache 的 layout hint 静默丢失、回落到形状嗅探 +
   NHD 默认;修复是开着的 PR **#4729**(08-26)。另有 **#4731**(修
   `kv_cache_group_edits` 的 packed subpaged,closes #4701 —— 混合模型在
   vLLM≥0.26 只存 1/N 内核页)也还没合。

## 一、用户决定

1. **"只需要支持 MP,in-process 是锦上添花"** → 版本可用性一律以 MP 口径判。
2. **"不允许 in-process,这东西已经淘汰了"** → 套件要改成只有 MP 一条路
   (默认即 MP,删选路开关与 in-process 部署点,certify 声明只写 MP)。
   **尚未实施**,拦路项见 §四。

## 二、代码改动(两个提交,均测试侧)

| 提交 | 内容 |
|---|---|
| `9177e0b1` | `harness.DeploymentPath` + `selected_deployment_path()`,由 `LMCACHE_MM_E2E_PATH`(auto/mp/in_process)选路;conftest 会话夹具与 `isolated_cases._deployment_harness` 都走它;**certify 的声明跟着实际路径走**(MP-only 跑出的证书不再宣称 in-process;混合模型的 MP 声明也从"仅 T0/T1 核心"纠正为全套);README 记了这个变量 |
| `48314ad6` | 容量档位改按实际路径选;`run_chunked_prefill` 也走 `_deployment_harness`(否则 MP-only 跑出来的证书会把 in-process 测得的 T0.9 算进 MP 声明) |

## 三、教训

1. **"全塌"必须按部署路径拆开看**:同一个套件里混合模型走 MP、其余走
   in-process,不分路径统计,就会把 in-process 的病算到版本头上 ——
   08-25 记录 1 的"12 张证书在 0.27.1 上不成立"实际是 in-process 的故事。
2. **我犯的错:没逐个确认场景的 harness 构造点就断言"三个失败都在 MP 上"。**
   `server_log_tail` 在日志里只出现一次(属于 eviction),我把它当成三个都在
   MP 的证据。真相是 `run_chunked_prefill` 直接 new 了 `MMHarness`。
   路由改造后必须逐个场景查构造点,一处命中不能外推。
3. **常量选档要按它真正依赖的维度 key**:"MP 分配器不给不足一个单位的容量"
   依赖的是路径,却写成了"是不是混合模型" —— 在路径可切换的那一刻立刻变错。
4. **长 TMPDIR 会让 LMCache 引擎初始化死在 `sockaddr_un` 107 字节限制**
   (scratchpad 路径就超了),表现为"全程零缓存"。探针的有效性闸门
   (`valid: false`)当场挡住了一个假结论 —— 闸门值回票价。
5. `faulthandler.register(SIGUSR1, chain=True)` 转储后会**链到默认处理**
   直接终止进程;要诊断活锁必须 `chain=False`。

## 四、MP-only 改造的拦路项

`test_deepstack`(中段续接,只对声明 `deepstack` 的 Qwen3-VL 系模型跑)
依赖 in-process 的缓存外科手术:`resident_chunk_keys()` / `clone_resident_kv()`
/ `evict_resident_keys()` / `resident_kv_tensor()` 直接摸 LocalCPUBackend。
搬到 MP 需要重写这四个辅助:服务端有 `/cache/objects`(GET/DELETE)与
`/cache/checksums`,足够做"删指定 key + 内容校验"(校验和比 Frobenius 差
更严,但量不出"差多少")。**等指令。**

## 五、诚实边界

1. 0.27.1 的 MP 口径只测了 qwen2-vl-2b(29 用例)与 qwen3.5-2b(27 用例),
   其余 10 个已认证模型**未测**;12 张证书仍全部量在 0.23.0 上。
2. ~~preemption 的归属已由裸 vLLM 批量对照坐实(属 LMCache)~~ —— 见 `3_`:
   归属改为 vLLM 的 `defer_block_free`,机制已由调度器插桩与两组单开关控制
   坐实;仍只在 qwen2-vl-2b + 128 块这一个配置上测过。
3. `test_deepstack` **从未在 MP 上跑过**;qwen2-vl-2b 不声明 deepstack,
   所以 29/29 与它无关。
4. MP 服务端的 parity 基准(`benchmark_parity.py`)仍按"是不是混合"选路径;
   MP-only 之后非混合模型的既有 parity 报告(记的是 `in_process`)会被
   certify 的路径校验拒收,需要重跑(小时级)。
5. 所有 0.27.1 的跑都在 `vllm-mm` venv(无 cuda_ops → torch 回退,带 08-25
   的 memcpy 修复);**native 路径在 0.27.1 上未验证。**

## 六、下一步

1. preemption:插桩连接器调度侧(lookup 门控/块归还),定位活锁的确切环节,
   然后报上游(0.27.1 + LMCacheMPConnector,抢占后不可恢复)。
2. MP-only 改造(含 deepstack 四个辅助搬到服务端 API)—— 等指令。
3. 若 MP-only 成立:12 个模型在 0.27.1 上按 MP 口径重认证(证书 schema 应加
   vLLM 版本字段),顺带认证 DeepSeek-OCR 与 Mistral Small 3.1。
4. 上游报告清单(继承 08-25 记录 6 §五,新增):0.27.1+MP 的 preemption
   活锁(若判定属 LMCache);#4463 补一条"我们独立复现 + MP 不受影响的实测"。
