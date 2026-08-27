# 抢占活锁翻案:病灶是 vLLM 0.27 的 `defer_block_free` 栅栏,LMCache 只是触发条件

**日期**: 2026-08-26(当天第 3 篇,接 `2_`;`2_` 与 `1_` §结论 4 的归属已按本篇更正)
**代码状态**: `multi_modal@48314ad6`,工作树干净(本篇仍不含仓库代码改动;
插桩脚本与三组 trace 归档在 `vllm_0271_mp/`,该目录现 20 项)

## 结论先写

1. **`2_` 把活锁判给 LMCache 是错的。真正的病灶在 vLLM 0.27.1 的调度器**:
   抢占释放的块被延迟归还(`defer_block_free`),导致同一步里的抢占"不解渴",
   调度器把整批请求全抢占;下一步块回池后整批又被放行、刚好填满块池;
   再下一步谁都拿不到新块 —— 两步一循环,零推进。

2. **因果锁死在单个开关上**(同 scenario、同 128 块、同 0.27.1 + MP 连接器):

   | 配置 | 抢占数 | 结果 |
   |---|---|---|
   | 默认(`defer_block_free=True`) | 每 2 步全抢占,单请求 npreempt 430+ | 活锁 |
   | `async_scheduling=False`(该标志随之关闭) | **2** | **PASSED,failures=[]** |
   | **async 照开,只强制 `defer_block_free=False`** | **2** | **PASSED,failures=[]** |

   第三行是关键:它排除了"async scheduling 本身"的嫌疑 —— 并发批次照旧,
   只把延迟释放关掉,活锁就消失。

3. **LMCache 的角色只是"让开关变真"**:
   ```python
   # vllm/v1/core/sched/scheduler.py:150-156 (0.27.1)
   multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
   if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
       self.defer_block_free = True
   ```
   实测该实例:`defer_block_free=True, requires_kv_delivery=True,
   max_concurrent_batches=2, async_scheduling=True, watermark_blocks=0,
   is_kv_consumer=True`。**任何** `is_kv_consumer` 的连接器 + 默认开启的
   async scheduling 都会踩到;裸 vLLM 没有 `kv_transfer_config`,标志恒 False,
   所以 `2_` 里的裸 vLLM 对照能恢复 —— 那组对照测的是"没有这个开关"的 vLLM,
   不是"vLLM 有没有毛病"。

## 一、机制(逐步骤 trace 实证)

插桩:包 `Scheduler.schedule` / `Scheduler._preempt_request` /
`Scheduler.__init__`,每步记录 running/waiting/skipped、块池 free、
`deferred_frees` 的条目数与块数、栅栏号、`sched_step_seq`/`processed_step_seq`、
本步调度 token 数,以及连接器 `get_num_new_matched_tokens` 的返回值。

关键片段(`sched_trace_default_livelock.log`,qwen2-vl-2b,128 块,6 请求):

```
step=39 run=6 wait=0 free=1  deferred=0entries/0blocks    sched_tok=6
step=40 run=1 wait=5 free=0  deferred=5entries/105blocks  sched_tok=1     <- 一步抢掉 5 个
step=41 run=6 wait=0 free=0  deferred=0entries/0blocks    sched_tok=1681  <- 全放回来,重算 5×336
step=42 run=1 wait=5 free=0  deferred=5entries/105blocks  sched_tok=1
...
step=56 run=0 wait=6 free=0  deferred=6entries/127blocks  sched_tok=0     <- 6 个全进等待
step=57 run=6 wait=0 free=0  deferred=0entries/0blocks    sched_tok=2032  <- 全放行,池刚好填满
step=58 run=0 wait=6 free=0  deferred=6entries/127blocks  sched_tok=0
...  (到 step=400 一直是这两态交替,npreempt 每请求 430+)
```

三处要点:

* **抢占在本步不解渴**。`_preempt_request` → `_free_request_blocks`,当
  `defer_block_free` 为真且请求的 `last_sched_seq > processed_step_seq` 时,
  块进 `deferred_frees` 而不是回池。所以调度器为了给一个请求腾一个块,
  会连着抢占 5 个、6 个,池的 free 始终是 0。裸 vLLM 那边抢一个就够了
  ——`2_` 记的"裸 vLLM 只抢 2 次"正是这个区别的表现,不是"vLLM 更健壮"。
* **整批再放行 = 池刚好填满**。128 块的池,6×336 token 的提示词吃掉 127 块,
  `watermark_blocks=0`(0.27.1 在本配置下算出 0),于是解码第一步谁都拿不到
  第 128 块 → 再全抢占。
* **栅栏本身还有一个更硬的死锁形态**:排空条件是
  `update_from_output` 里 `scheduler_output.total_num_scheduled_tokens > 0`
  才 `processed_step_seq += 1`。若某步因无可用块而调度 0 token,栅栏就不推进,
  而块又只能靠栅栏推进才回池。本次跑观察到的是"两步交替"(奇数步仍能调度到
  token,栅栏得以推进),但 `proc_seq` 始终紧咬栅栏号差 1,离硬死锁只差一步。

## 二、判别实验怎么做的

`preempt_traced.py`(归档):在 `scenario_wrapper.py` 的基础上加插桩,
`TRACE_STEP_CAP` 到点后转储全线程栈并 `os._exit`(外层 `setsid` + 组 kill
回收 MP 服务端);两个控制开关:

* `TRACE_ASYNC_OFF=1` —— 给 `vllm.LLM.__init__` 注入 `async_scheduling=False`;
* `TRACE_DEFER_OFF=1` —— 包 `Scheduler.__init__`,init 后把
  `self.defer_block_free` 置 False(**只用于判别,不是可推荐的规避手段**:
  这个标志本来是为"并发批次下连接器可能把仍在被写的块拿去装载"而加的)。

两个控制组都跑完了完整 scenario(含抢占后的 replay 校验),
`failures: []`,`preemptions: 2`。

## 三、可推荐的规避

**`async_scheduling=False`**(而不是强关 `defer_block_free`):
`max_concurrent_batches` 回到 1,`defer_block_free` 自然为假,栅栏与抢占的
互锁不成立,连接器该有的保护也没被绕过。代价是丢掉 async scheduling 的吞吐。

**未定**:套件的 `preemption` 场景是否固定 `async_scheduling=False`。
固定了就测不到"用户默认配置下会活锁"这件事;不固定,场景在 0.27.1 上永远红。
倾向:场景里保留默认配置并把这条记为**已知的上游缺陷**,等上游修。

## 四、证据(`vllm_0271_mp/`,新增 6 项)

| 文件 | 内容 |
|---|---|
| `preempt_traced.py` | 插桩跑法(含两个控制开关) |
| `sched_trace_default_livelock.log` | 默认配置:两态交替、`deferred=6entries/127blocks` |
| `sched_trace_asyncoff_pass.log` / `scen_result_asyncoff.json` | async 关:2 次抢占、通过 |
| `sched_trace_deferoff_pass.log` / `scen_result_deferoff.json` | 只关延迟释放:2 次抢占、通过 |

## 五、教训

1. **对照组要对照"那个开关",不是对照"那个组件"。** `2_` 用裸 vLLM 做阴性
   对照,而裸 vLLM 恰恰不满足 `is_kv_consumer`,等于把被测开关一起去掉了 ——
   于是"vLLM 没坏"这个结论其实是"没开这个功能的 vLLM 没坏"。这与 08-25 的
   "命中门空转"、`2_` 的"baseline 顺序跑"是同一族错误:**对照组必须保留
   触发条件,只动待判变量**。这次是第三次栽在同一处。
2. **归属判断要读被测版本的代码,不能只看行为差**。行为上"裸的行、挂连接器的
   不行"极易读成"连接器的锅";而 0.23→0.27 的 scheduler diff 里
   `defer_block_free` 是新增的、且只对连接器生效 —— 十分钟的 diff 能省掉
   一整篇错误结论。
3. **插桩的粒度决定能不能一次定案**:这次一次跑就够,是因为把
   "块去哪了"(`deferred_frees` 条目/块数)和"栅栏走到哪"
   (`sched_seq`/`proc_seq`)一起打了。只打抢占日志(`2_` 的做法)只能看到
   "又抢了一遍",看不到"块被谁扣着"。

## 六、诚实边界

1. 只在 **qwen2-vl-2b + 128 块**这一个配置上做的判别;其他模型/池大小未测,
   但机制与模型无关(纯调度器与块池)。
2. 没有向上游确认这是不是已知问题(**未查 vLLM issue/PR**),也没做最小
   复现脱离 LMCache —— 理论上用任何 `is_kv_consumer` 的假连接器就能复现,
   报上游前应当先做出这个最小复现。
3. `watermark_blocks=0` 是否也是 0.27 的回归(0.23 上是 1% 向下取整)没查;
   它只是放大器,不是病因。
4. 三组跑都在 `vllm-mm` venv(无 cuda_ops → torch 回退,带 08-25 的 memcpy
   修复)。

## 七、用户决定与下一步

**用户决定(本轮)**:
1. **只支持 MP**;把当前 in-process/MP 混合实现**先存一个分支**留档,
   然后把 in-process 代码**剔除干净**,全面转 MP。
2. 若因此某些模型/子套件跑不了,**测完如实报告**,不硬凑。

**下一步**:
1. 建归档分支(混合实现) → 主线删除 in-process 部署路径:
   `harness.selected_deployment_path`/`DeploymentPath` 去掉,
   `MMHarness` 与 `MPHarness` 合一,conftest / `isolated_cases` /
   `benchmark_parity` / `certify` 一律 MP。
2. **`test_deepstack` 搬不过去**(已确认):它的唯一有效 oracle 是 KV 内容的
   rel-Frobenius(其模块 docstring 自证:关掉 deepstack 注入不改变任何输出
   字节,所以输出级 oracle 全盲),而它靠 in-process 直读 LocalCPUBackend 张量;
   MP 服务端没有任何接口能把已存对象读回来(`GET /cache/objects` 只支持 L2 且
   本地服务器是 L1-only;`POST /cache/checksums` 校验的是 **GPU** 块、且只给
   MD5,而重算噪声本就不是逐位相同)。可行的重写方向是"GPU 侧取 KV 做距离",
   属独立一件事。按用户指示:MP-only 之后如实报告该子套件不可用。
3. 上游报告:vLLM 0.27.1 `defer_block_free` 与抢占互锁(先做脱离 LMCache 的
   最小复现);`#4463` 补我们的独立复现与"MP 不受影响"实测。
