# 翻转方向门(schema 8)落地,三跑重认证在飞

**日期**: 2026-08-26 21:45(当天第 15 篇,接 `14_`)
**代码状态**: `multi_modal@cb4f789e`,工作树干净
(`cb4f789e` = MME 答案翻转的方向门 + 证书 schema 8;父 `fc5755ca`)

## 一、这一篇解决的问题

`10_` 判定 19 个答案翻转是 vLLM 0.27.1 的 ±1 bf16 量子噪声、不是缺陷,
`a419a4c5` 据此把 qwen2-vl-2b 的 flip 预算放到 0.01。问题是:**放宽计数
就等于放宽污染的掩护**。0.005 预算在 2374 题上是 11.87 个翻转的掩护,
0.01 是 23.74 个;一个只产生 ~20 个翻转的真实污染就能混过去。

而 specs.py 里 12 个模型有 7 个还挂着默认 0.005,那是在 vLLM 0.23 上定的
(当时全注意力模型逐字节相同、0 翻转,见 gemma-3-4b 的 1715.68 对
1715.68)。0.27.1 的 batch 形状效应会把这 7 个逐个顶红,届时每个都要
放宽 —— 掩护一路变大,而计数本身分不出噪声和缺陷。

## 二、判据:方向,不是计数

两种让 verdict 移动的东西,只在方向上不同:

- **引擎噪声是双向的**。batch 形状让首 token logits 差 ±1 个 bf16 量子,
  卡在 yes/no 边界一个量子内的题两边都翻。实测:qwen2-vl-2b 翻 19/2374,
  总分 1968.78 → 1966.53,**只动 2.25 分(0.11%)**,分类目一涨一落
  (code_reasoning +7.5、color +5.0 对 numerical_calculation −7.5、
  text_translation −7.5)。
- **KV 污染是单向的**,只会变差。`13_` 的垃圾就是靠"单向 + 头部聚集"
  认出来的。

这条推理此前**只活在 specs.py 的人类注释里**(glm 那条:"the hit pass is
right 5 times against the miss pass's 7 (a corrupting cache would be
one-sided)"),没进门。19 个翻转分 10/9 和分 19/0,旧门一视同仁地放过。

## 三、改动(`cb4f789e`)

| 位置 | 内容 |
|---|---|
| `benchmark_parity.py` | `Benchmark.ground_truth()` 新抽象方法(MME → `item["answer"]`,MMAU → `item["answer_letter"]`);两个 `scores()` 也改走它,答案键不会再各读各的 |
| 同上 | `FlipCounts` 增 `regressions` / `improvements` / `lateral`,三者划分 `answer_flips` |
| 同上 | `flip_asymmetry_p()`:精确单侧二项尾 `P(X >= regressions)`,公平硬币;单侧是故意的,改对变多不是污染特征 |
| 同上 | `MAX_FLIP_ASYMMETRY_P = 0.01` 进 gate,**计数预算之内也能判红** |
| `certify.py` | schema 7 → 8;`parity_command()` 从 `run_parity` 拆出 |
| `specs.py` / `README.md` | 文档 |
| `test_parity_gate.py` | 11 → 17 项 |

**标定**:19 个翻转要 ≥15 个偏一边才红(p=0.0096 < 0.01),14 个过
(p=0.032)。10/9、12/7 轻松过。计数预算内单向 11 个 → p=0.5^11 判红。
glm 实测的 7/5 → p=0.387,过。测试把 15 红 / 14 绿 的边界钉死。

**lateral**(错→另一个错)不进二项分母,只受计数约束。MME 出不来
(yes/no 一翻必跨答案键),MMAU 四选一会有。

## 四、schema 8 的代价(必须知道)

`load_parity_report()` 现在**拒收没有方向计数的 parity 报告**,不按计数
重判 —— 否则 schema 8 证书会宣称一项从没跑过的检查。后果:

- 两张 schema 7 证书(qwen2-vl-2b、gemma-4-e4b)**无法靠重判升级**,
  必须重跑 parity。这是本次改动的真实代价,已在跑(§六)。
- 手动跑 parity 用 `certify.parity_command(key, limit, out)` 取 argv,
  别手拼。拆出来的命令跟 `mm_temp_ctx.md` 里手写的 gemma 参数一致,
  算反向验证。

**标定的实测基础偏薄**,要留意:直接数出来的方向分布只有 glm 的 7/5
(还是人工注释里的)。qwen 那 19 个的"接近均分"是从 2.25 分总分位移和
分类目一涨一落**推**的,不是数的 —— 跑的时候还没有计数器。§六 的三跑
会给出第一批真正数出来的分布,届时回看这个 0.01 是否合适。

## 五、用户决策:preemption 活锁不归我们

vLLM 0.27.1 在 `max_concurrent_batches > 1` 且连接器是 KV consumer 时置
`defer_block_free`(`scheduler.py:131/150-156`),抢占释放的块进
`deferred_frees` 拿不回来,调度器不知道,于是抢空整批 → 下一步全放 →
再撞,实测 430+ 抢占零进度。

澄清了一个误解:**不配 KV 连接器碰不到**(置位那段整个嵌在
`if kv_transfer_config is not None:` 里),所以现象是"开 LMCache 就卡死",
容易被误判成我们的 bug。我们用 `kv_role="kv_both"` 必然落进 consumer 集合,
躲不开。连接器没有 opt-out 口子(对比 `requires_kv_delivery` 是连接器自己
声明的属性)。

**用户 8/26 定:不归我们管,不 investigate,PR 里提一嘴即可。**
已记进 `mm_temp_ctx.md` §九。

## 六、在飞跑(21:45 采样)

三跑并行,错开 4 分钟起(避 vLLM 显存 profiling 一致性断言,见
`mm_temp_ctx.md` §六.2 那次 false red):

| GPU | 模型 | 起时 | 目的 |
|---|---|---|---|
| 2 | gemma-4-e4b | 21:41 | schema 8 重认证 |
| 3 | qwen2-vl-2b | ~21:45 | schema 8 重认证 |
| 7 | qwen2.5-vl-3b | ~21:49 | 探针:默认 0.005 预算的模型在 0.27.1 上翻多少、往哪边翻;顺带出第三张证书 |

- 启动器 `$SP/run/launch.sh`(spec 派生 argv)+ `$SP/run/supervise.sh`
  (错峰,`setsid` detach)。报告落 `$SP/parity_<key>.json`,
  日志 `$SP/run/{gemma,qwen2,qwen25}.log`。
- 三跑完再**串行**出证书 —— 套件有抢占、容量驱逐等时序敏感场景,
  并行容易假红。

**并行的坑**:MP server 端口是 `26000 + (pid % 1000)` /
`27000 + (pid % 1000)`(`benchmark_parity.py:1318`)。两跑 PID 模 1000
同余就撞,而且**不一定报错** —— healthcheck 打 `localhost:{http_port}`,
先起的会回 200,后起的可能拿到别人的 server 句柄,两跑共用一个缓存,
静默毁两边。概率约千分之一。§九 那个"server 起立但 ZMQ 300s 不可达"
的未解释事故,这是个说得通的候选原因,但没有证据当时是否有并发跑,
仅列假设。

## 七、过程中发现的三处错误

1. **`mm_temp_ctx.md` §四 的旧码树路径是错的**。写的
   `/home/bo/LMCache-worktrees/verify` 不存在,`git worktree list` 实际是
   **`/home/bo/LMCache-worktrees/multi_modal_verify`**(commit `2485fdbc`,
   对得上)。已就地更正并加注。
2. **树不干净会毁证书**。`certify.py` 用 `git status --porcelain` 判 dirty,
   未跟踪文件也算。`mm_temp_ctx.md` 未被 exclude,会让证书标
   `stable: false`、不指向任何可复现 commit。已加进
   `/home/bo/LMCache/.git/info/exclude`(跟 `certificate_*.json`、
   `parity_*.json` 同一处理),并在跑之前先提交了 `cb4f789e`。
3. **ruff 抓到拆函数的残留引用**。`parity_command()` 拆出去后
   `run_parity` 里还留着 `cwd=script.parent`,`script` 已不在作用域 —— 
   F821。已修。lint 值这个钱。

## 八、保留项

1. **静箱旧码复现(原 §四,仍未做)**。21:24-21:43 共租
   (vllm-lazy,pgid 1176678,GPU 0/1/5/6)在跑,把窗口关了;21:43 它
   整组退出,机器现在安静(只剩 GPU 4 上一个 k8s root 常驻服务,
   1.4% 单核,不吃 host)。但我自己的三跑正好填进这个窗口。
   **触发条件**:三跑收工后若 `nvidia-smi` 上只剩闲置服务、load 低,
   用 `multi_modal_verify` 树跑 qwen 全量,看 pass2 头部垃圾是否复现。
   判据见 `mm_temp_ctx.md` §四。
   注意共租近 30 分钟内翻了两次(21:18 静 → 21:24 忙 → 21:43 静),
   窗口不稳;而垃圾聚集在 pass2 头部,对应开跑后 ~45-60 分钟那一段,
   没法调度到确定的安静时刻。
2. 10 个模型在 0.27.1 上没测过(本轮后剩 9 个)。7 张 schema 2-3 证书
   已过期。
3. 音频路径(MMAU)在 0.27.1 上一次没走通过;qwen3-omni-30b 是唯一带
   audio 的 spec。
4. vLLM ±1 量子问题要不要报上游 —— 用户未定。
5. `async_scheduling=False` 绕开的 preemption 活锁,PR 提一嘴(§五)。
