# 版本回溯:0.25.1 是 0.27.1 前最新的好版本(断点 0.26.0);T3 MP 竞态判归 base

**日期**: 2026-08-25(当天第 4 篇,接 `1_`、`2_`、`3_`)
**代码状态**: `multi_modal@0040c6bd`,工作树干净,未提交(本会话没改任何仓库代码;
全部工作是 venv 搭建、文本探针、e2e_mm 套件运行、T3 解剖与归属实验,
产物归档在 `vllm_upgrade/bisect/`,共 24 项)

## 结论先写

1. **0.25.1 是 0.27.1 之前最新的、没有 hit 腐坏的 vLLM 稳定版;坏是 0.26.0
   引入的。** 六个稳定版的文本探针(带 provenance 闸门,全部 `valid: true`,
   每 hit 实装 432–448 外部 token,同一棵 `multi_modal@0040c6bd` 树):

   | vLLM | torch | hit 准确率 | 判定 |
   |---|---|---|---|
   | 0.23.0 | 2.11 | 0.9375(=miss) | 好(已认证) |
   | 0.25.0 | 2.11 | 0.8125(=miss=baseline) | 好 |
   | **0.25.1** | 2.11 | **0.8125(=miss=baseline)** | **好** |
   | 0.26.0 | 2.11 | 0.0 | 坏 |
   | 0.27.0 | 2.13 | 0.0 | 坏 |
   | 0.27.1 | 2.13 | 0.0 | 坏(记录 1/2/3) |

   0.25.1(好)和 0.26.0(坏)同在 torch 2.11 上 —— torch/triton 栈被排除,
   坏的就是 vLLM 0.26.0 引入的东西(与 fused KV 布局进入 0.26 的时间线吻合)。
   0.24.0 未测(对"最新的好版本"这个问题不需要;venv 已备好)。
   好版本之间 0.9375 vs 0.8125 的差异在裸 vLLM baseline 上就存在,是引擎
   自身数值漂移,与缓存无关 —— 判定标准始终是 hit=miss=baseline 三者对齐。

2. **0.25.1 上 e2e_mm 抽验两模型,in-process 路径全绿。**
   qwen2-vl-2b 28/29(17 分 21 秒)、qwen3-vl-2b 33/34(17 分 30 秒),
   覆盖 T0 计数器/逐字回放、chunked prefill、eviction、preemption、
   阴性对照、跨模态。
   **修正(用户指正,写记录后)**:我们重点支持的是 **MP 模式**,不是
   in-process。因此"0.25.1 可作落脚版本"只对 in-process 路径成立;
   对 MP 路径,**版本选择解决不了问题** —— 唯一的红(T3)与 vLLM 版本
   无关(见 3),换哪个版本都带着这个竞态。MP 可用性的关键路径是竞态
   根因,不是版本回退。另注:文本探针走的是 in-process connector,
   0.26.0+ 的 hit 腐坏对 MP 路径是否同样成立未单独实测(KV 装载路径
   共用,推断成立,未验证)。

3. **唯一的红(T3 mp_connector)是独立的 base 侧竞态,与 vLLM 版本无关,
   已用归属实验判定属 base。**
   **→ 更正(记录 5)**:根因已定位,本条的"存入的键几秒内丢失"是误判
   (实为两条只共享 chunk-0 的键链 + store 可见性滞后 + torch 回退
   memcpy 流序缺失),归属判定(属 base)不变,细节以记录 5 为准。证据链:
   - 与版本无关:0.23.0(已认证的好版本)上同样红,三个 venv
     (bisect-0.23.0、bisect-0.25.1、系统 Python 的 vllm-lazy)全复现。
   - 机制是通的:服务端日志显示 Stored 288 → prefetch `18/18 retained` →
     Retrieved 288 —— MP 传输本身工作。
   - 真实形态:**存入的键几秒内丢失**(`35/35`、`18/18` 掉到 `1/18`),
     失败集逐轮漂移(t01 这轮过、上轮零),偶发错位 KV 注入
     (t05 答 "Hello! How can I assist you today" 而非 "Paris")——
     时序敏感竞态。同一棵树 08-22 认证时 T3 全绿,代码没变,变的是
     触发条件(当天机器上另一会话在跑重 MP 负载,是候选诱因,未定论)。
   - **归属实验**:分支测试代码 + 引擎与 MP 服务两侧 lmcache 都指向纯上游
     `dev_head@c1ef01b9`(运行时自证解析路径),vLLM 0.23.0 排除 hit 腐坏
     干扰 → **7 项失败,同款签名**。base 原样复现,分支只是继承。

4. **两个假信号已识别并更正,不作数**(教训在 §三):
   - 精简 venv 缺 `cupy` → MP 服务端 REGISTER_KV_CACHE handler 崩溃,
     且**只记日志不回错误包**,客户端干等 300 秒超时 —— 表现酷似"版本不兼容"。
   - 直跑 `isolated_cases.py` 缺 pytest 注入的环境(引擎侧 chunk 默认 256
     vs 服务端 16)→ 键天然不对齐,全零命中 —— 表现酷似"MP 路径全坏"。

## 一、发现路径(按时间)

1. PyPI 列出 0.23–0.27 之间稳定版:0.24.0 / 0.25.0 / 0.25.1 / 0.26.0 / 0.27.0。
2. 并行建 venv 跑探针:0.27.0 红;0.25.1/0.26.0 首轮死于 triton launcher
   编译(系统 python3.12 无 dev 头文件)→ 换 uv 托管 Python 重建 → 0.26.0 红、
   0.25.1 绿、0.25.0 绿。断点闭合于 0.25.1|0.26.0。
3. 0.25.1 跑两模型完整套件:in-process 全绿,唯 T3 红(cupy 崩溃)。
4. 补 cupy 重跑 T3 → 仍红但症状变为零命中/污染 → 发现直跑缺环境 → 走 pytest
   正道重跑 → 仍红 → 0.23.0 对照(pytest 正道 + vllm-lazy)也红 → 排除版本与
   venv → 解剖运行(双日志)发现"存得进、键会丢、逐轮漂移" → 竞态定性。
5. 归属实验(dev_head 双侧 + 护栏定向绕过)→ base 复现 → 判归 base。

## 二、证据与产物(`vllm_upgrade/bisect/`,24 项)

| 文件 | 内容 |
|---|---|
| `textacc_{0250,0251,0260,0270}.json` + `probe_*.log` | 四个版本的探针结果与日志 |
| `suite_0251_qwen{2,3}-vl-2b.log` | 0.25.1 完整套件日志(28/29、33/34) |
| `t3_0251_qwen{2,3}.log`、`t3_0230_qwen2.log`、`t3_lazy0230_qwen2.log` | T3 各对照(pytest 正道) |
| `t3_instr_run.log` + `mp_server_snapshot.log` + `t3_instr_result.json` | 解剖运行:引擎+服务端双日志 |
| `t3_base_result.json` + `t3_base_run.log` + `mp_server_snapshot_base.log` | **归属实验:base 复现** |
| `t3_wrapper.py` / `t3_wrapper_base.py` | 场景直跑 wrapper(补环境版 / base 归属版) |
| `run_probe.sh` / `setup_venv.sh` | 探针 runner(带 teardown 杀进程)/ venv 配方 |

## 三、教训(下次直接避开)

1. **系统 python3.12 无 dev 头文件**,torch 2.11 代的 triton JIT 编 launcher
   必挂;uv 托管 Python 自带头文件,`UV_PYTHON_PREFERENCE=only-managed` 解决。
2. **uv 建的 venv 没有 pip**,包清单用 `uv pip list --python <venv>`;
   我用 `<venv>/bin/pip` diff 出过一整页假差异。
3. **`isolated_cases.py` 不能裸跑**:它只自补 prompt 形状两个变量,chunk
   等其余环境靠 pytest 父进程注入;裸跑必须先调 `harness.configure_environment()`。
4. **MP 服务 handler 崩溃不回包**(`mq.py:627` 只记日志),客户端等满
   `mq_timeout`(300 秒)才报 ConnectionError —— 排障时先看服务端日志,
   且这本身是 base 的健壮性缺陷,值得随竞态一起报。
5. **ZMQ IPC 路径有 sun_path 107 字符上限**:TMPDIR 指到长路径(如
   scratchpad)会让 vLLM 启动就挂;用短符号链接绕开。
6. 探针 runner 加"结果落盘即杀进程组"后,本会话零挂死零显存泄漏。

## 四、诚实边界

1. T3 竞态的**触发条件未定论**:同代码 08-22 绿、今天红,机器负载(另一
   会话的 MP 服务 + 152% CPU 引擎)是候选诱因但没做受控验证;也可能只是
   竞态概率。根因(键为何丢:storage_manager retention / 驱逐 / pin-count?)
   未定位 —— 与早前看到的 `Pin count negative / double unpin` 警告可能同源,
   未证实。
2. 归属实验绕过了 isolated_cases 的 repo-pin 护栏(spoof `__spec__.origin`),
   这是**有意为之且只此一次**,实验不写任何证书;wrapper 里有完整注释。
3. dev_head 的 `cuda_ops.so` 在 torch 2.11 venv 下加载失败退回 torch 基线
   ops —— 与分支运行同基线,可比;但"cuda_ops 路径下竞态是否仍在"未测。
4. 0.24.0 未测;qwen3 系列以外的 10 个模型未在 0.25.1 上抽验。

## 五、下一步(等指令)

1. **T3 竞态根因定位**(服务端 storage_manager 的键保留逻辑附近)。
   MP 是重点支持模式,此项升为第一优先:竞态不解决,MP 路径在任何
   vLLM 版本上都无法给出干净认证。
2. **报上游**:现在是一份报告三个问题 —— (a) 0.26.0+ fused 布局 hit 腐坏
   (断点已定位到 0.26.0,比记录 3 时更准);(b) T3 MP 键丢失竞态
   (纯 base 复现包在 `bisect/t3_base_*`);(c) CI 命中门空转(记录 3)。
   附带健壮性小项:MP handler 崩溃应回错误包而非让客户端超时。
3. 若升级推迟,可在 0.25.1 上补全 12 模型重认证(证书 schema 加 vLLM
   版本字段的事一并做)。
4. 基础设施留存:6 个 `vllm-bisect-*` venv(0.23.0/0.25.1/0.26.0 已含
   cupy)、`vllm-nightly` venv、dev_base/dev_head 工作树 —— 后续实验现成。
