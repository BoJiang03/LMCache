# 支持测试宏观复盘:补齐"跑绿即支持"的保证链

日期:2026-08-19
分支:`multi_modal` @ `4f2bc199`(32d62180 认证补全 + 4f2bc199 oracle 修复;已推 fork;记录文件本身按政策不入 git)
前置记录:`4_records_policy_mme_gate_t07_storage.md`

## 任务

用户(8h 离线)要求宏观思考:支持测试还差什么、多了什么、如何简洁又完备、如何保证"跑完得到某个结果 → 可宣称支持该模型",并自主推进。

## 宏观结论

"支持"的定义拆成四项职责:key 携带图像身份 / 命中路径不腐蚀 KV / miss 全部恰好存一次 / 在真实调度(批量、分步 prefill、驱逐)下仍成立。完备性 = 枚举每项职责的失效模式,每个模式至少有一个探测器指着它(README 新增失效模式→探测器映射表,12 行)。简洁性 = 没有不对应失效模式的测试(复盘后没有可砍项;16 个边界相位是 chunk 对齐的完整周期,MME 三遍是刻意的夜间级认证层)。保证 = 证书而非绿灯:certify.py 一条命令产出机器可读证书,verdict=SUPPORTED 且证书内写明 commit、认证范围(路径/模态/后端)与 known-not-covered 边界——宣称永远不宽于证据。

## 本次发现并补掉的缺口(按严重度)

1. **探测器自证(negative control)**:套件从未证明"计数器真的会响"。`harness.identity_blindness()` 关掉 mm_hash 替换(#3301 的极端形态:key 对图像内容全盲),新测试断言 T0.1 式报警必须触发(B 图必须假命中到 A 的全深度)。绿灯从此自带证据。
2. **T0.9 chunked prefill 切图**(此前合成测试全是单步 prefill,store 侧 mid-image 截断分支零确定性覆盖):独立引擎 max_num_batched_tokens=128,prompt 336–384 tokens(448² 图 = 256 token span),4 个 pad 相位扫步边界过 span。实测 full hit 逐 token 精确、store==missed 精确到 0 偏差。
3. **T0.10 容量驱逐**(原 40GB 永不满,驱逐路径零覆盖):独立引擎 50MB 上限,32 张图 6× 溢出。实测常驻 53,673,984B ≤ 53,687,091B 上限、无假命中、被驱逐请求重算逐字一致。
4. **T0.8 并发 batch**(原全串行):同 batch 含重复图 + 混合流量(store 与重复项 lookup 竞态),逐条输出校验 + batch 后单发必须全命中。
5. **certify.py**:跑套件(junit 解析,**skip 一律不算绿**)+ 跑/注入 MME parity(benchmark_parity 抽出可复用 parity_gate(),报告内嵌 gate verdict)→ certificate_<model>.json。退出码 0=SUPPORTED / 2=PROVISIONAL(缺 parity)/ 1=NOT_SUPPORTED。

## 实现要点

- 隔离场景走子进程(`isolated_cases.py` CLI + `test_isolated_paths.py` 包装):引擎参数/LMCACHE_MAX_LOCAL_CPU_SIZE 需进程级隔离;gpu_util 0.35 可与主 session 引擎共存。
- MMHarness 增 `extra_engine_kwargs`、`max_local_cpu_gb`、`run_batch()`(聚合计数器,batch 内不可归因)、`check_text()`、公开 `probe_ok()`。
- 失明开关搭在既有 identifier recorder 包装上,adapter 两个调用点(lookup + store)都在包装安装之后绑定,双侧同时生效。

## 验证(GPU 3,最终状态 @ 4f2bc199)

- 正式证书(本目录):**qwen2.5-vl-3b = SUPPORTED**(26/26,0 skip,407s;+ 全量 MME parity gate)、**qwen2-vl-2b = SUPPORTED**(26/26,0 skip;全量 MME 最强形式通过:baseline/pass1/pass2 三组逐分一致 1966.06,双向 0/2374 翻转,hit ratio 1.000;报告归档本目录 `mme_full_qwen2-vl-2b.json`)。**第一个横向模型已完整认证,全程零新增测试代码——只用了 specs.py 里现成的一行 ModelSpec。**
- 过程验证:chunked_prefill / capacity_eviction 两模型单跑均 PASS;chunked 场景 store==missed 精确到 0 偏差,驱逐场景 6× 溢出下常驻 53,673,984B ≤ 上限 53,687,091B。

## 横向第一步:qwen2-vl-2b 认证暴露 oracle 缺陷(4f2bc199 修复)

用新流程直接跑 `certify.py qwen2-vl-2b`:25/26,唯一失败 = chunked_prefill 场景 pad 40/56/72 的 **miss 路径** probe(模型对纯红图答 'Blue')。诊断:失败点 hits==0(LMCache 未参与前向),所有缓存不变量(命中/隔离/守恒 stored==missed 精确)全过;纯 vLLM 无 LMCache 对照实验逐点复现('Blue'×3 + pad88 'Red')——**是 2B 模型在长 pad 前缀下的能力缺陷,场景把模型能力误归因给缓存**。

修复(4f2bc199):隔离场景改用与主套件相同的 oracle——同引擎配置的纯 vLLM baseline 子进程(baseline_runner/compute_baselines 增 extra_engine_kwargs 透传),probe 退回非确定性 rescue 角色。修复后 chunked_prefill 两模型全绿。方法论教训入 README:probe 单独做正确性 oracle 会把弱模型的错误算到缓存头上;oracle 必须 config-matched。

## 明确不在保证内(证书 known_not_covered)

MP connector(T3/P2 未做)、TP>1、remote/disk 后端、video/audio、抢占重算(仅被 MME 批量统计性覆盖)、allocator 层对账(pin bug 立项)。

## 下一步

- P2:`_0180`/`_0201` MP connector 补 MM(生产主路径;做完后证书 scope 加第二条 path)——等用户拍板。
- 横向继续铺模型:每个新模型 = specs.py 加一行 + `python certify.py <key> --run-parity` 一条命令(qwen2-vl-2b 已验证该流程端到端成立)。
- 单独立项:pin count 负数 bug + allocator 层守恒对账。
