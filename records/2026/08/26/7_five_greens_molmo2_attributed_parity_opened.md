# 抽样五套件转全绿:molmo2 定性、抢占钉住、诊断落盘,parity 起步

**日期**: 2026-08-26(当天第 7 篇,接 `6_`)
**代码状态**: `multi_modal@4904b9f4`,工作树干净(本篇含六个新提交,均在测试侧)
**测量树**: `multi_modal_verify@2485fdbc` + 三个提交 cherry-pick(未提交)
**日志**: `records/2026/08/26/run0271b/`(五个绿套件 + 两个被抢卡的残跑 + molmo2 单场景 + gemma parity smoke + 三个启动脚本)

## 结论先写

1. **五个抽样套件 5/5 全绿,含抢占场景**,用例数与事先按谓词算的预估逐个吻合:

   | 模型 | 结果 | 用时 | 预估 |
   |---|---|---|---|
   | qwen2-vl-2b | 29 passed / 3 deselected | 16:19 | 29 ✓ |
   | qwen3.5-2b | 27 / 3 | 15:34 | 27 ✓ |
   | gemma-4-e4b | 27 / 3 | 22:04 | 27 ✓ |
   | molmo2-4b | 26 / 5 | 16:20 | 26 ✓ |
   | qwen3-omni-30b | 31 / 1 | 19:12 | 31 ✓ |

   `6_` 里那个红(`capacity_eviction[molmo2-4b]`)和那个挂死(`preemption`)都不在了。
   抢占按 `isolated_scenarios()` 只路由给 qwen2-vl-2b、molmo2-4b、qwen3-omni-30b
   三个模型,这三个都跑过且都绿。

2. **三个提交**(全在 `tests/e2e_mm/`,PR 范围内):

   | 提交 | 内容 |
   |---|---|
   | `4debe5d0` | 抢占场景钉 `async_scheduling=False` |
   | `687bede3` | 子进程输出落盘、超时 2400→900s、服务端日志与回收、驱动断言不再派生 |
   | `93d31dfa` | molmo2-4b `eviction_capacity_gb=0.5`,并改掉 `EVICTION_CAPACITY_GB` 注释里的错误说法 |
   | `7c72caf7` | `benchmark_parity.py` 父进程侧进度:`load_items` 前后、baseline spawn 前后 |
   | `4720a75d` | 证书 schema 5→6,`runtime` 块记 vllm/torch/transformers/lmcache 版本与解析路径 |
   | `4904b9f4` | baseline 子进程侧进度,行前缀 `[parity:baseline]` 与父进程区分 |

3. **用户对"支持程度回到 0.23"的判断作了校准**(本轮的实质讨论):0.23 的
   `SUPPORTED` 是有定义的 —— `verdict_meaning = "synthetic suite + MME parity
   green on the paths below"`,12/12、双路径、含抢占。用户接受"MP 单路径"和
   "抽样 5 个"两条收窄(前者是既定范围,后者本来就是抽查),但明确"只有合成套件"
   是问题、"合成套件里还有红和挂死"是要解决的问题。后两条本篇解决,第一条开始动。

4. **parity 跑法在 0.27.1 上是通的**:gemma-4-e4b 的 24 题 smoke 门**过**
   —— MP 路径、`flips_pass1_vs_baseline=0`、`flips_pass2_vs_pass1=0`、
   两个分数差都是 0.0、`pass2_hit_coverage=1.0`、parse ratio 0.9583。
   但全量一个都还没跑完(见 §四.4、§五)。

## 一、molmo2 定性:是档位,而且推翻了注释里的说法

给 spec 加 `eviction_capacity_gb=0.5` 重跑,`failures: []`:

| 量 | 64 MB(默认) | 512 MB |
|---|---|---|
| 常驻 key | 0 | 181 |
| 常驻字节 | 0 | 427032576(cap 的 0.795) |
| 意图流量 | 0(派生) | 3.70 GB(6.9x 溢出) |
| allocator | 0 active allocations,64 MB 全空 | 正常 |

但**它是"单位是一个 cache object"这个说法的反例**。molmo2 的 KV 是
144 KB/token(全套件最宽),一个 object 是 16 token = **2.36 MB**,64 MB 里能放
28 个 —— 按对象大小根本不该被拒。而一个 molmo2 图请求是 ~787 token = **~115 MB**。
两次实测都符合"存是按请求整笔预留、放不下就整笔拒",而不是逐对象填。

**这个机制是从两次测量推出来的,没有去读存储路径。** 能确定的只是它蕴含的要求:
cap 必须装得下一整个请求的 KV。`EVICTION_CAPACITY_GB` 的注释原来写"单位是一个
cache object",现在把这个反例和要求都记进去了。

## 二、抢占:为什么钉 async 而不是保留默认配置

`3_` 留的未定项(保留默认配置永久红 vs 场景层面规避)本轮定了:**场景钉
`async_scheduling=False`**。理由是这个场景的被测对象是 LMCache 的重算路径,
不是 vLLM 的调度器;保留默认只能让它永久红,还测不到我们自己的代码。
`3_` 已经证过 async 关掉后 2 次抢占、`failures=[]`。同一个值经
`extra_engine_kwargs` 也进 baseline 引擎,所以两边引擎配置仍然对齐。
默认配置下的活锁作为已知上游缺陷单列,最小复现仍未做。

本轮的新证据:钉住之后 `preemption` 在三个路由到它的模型上全部随套件跑过。

## 三、超时改了主意:900s,不是分钟级

原计划是"2400s 换短超时"。做的时候改了:真正治挂死的是把活锁从根上关掉,
超时只是兜底,按分钟级压反而容易在 27B 上假红。于是去翻了九份归档 junit 的
逐用例时长:**最慢的隔离场景是 249s**(glm-4.6v-flash 的 `mp_connector`),
两个 27B 是 220s / 228s。900s 是最坏情况的 3.6 倍,挂死 15 分钟变红而不是 40。
数字有依据,不是拍的。

## 四、过程里的事故与教训

### 1. 并行度不能按"现在空几张卡"定

10:12 起三个套件(GPU 1/5/7),10:14 另一个 session 抢下 GPU 0/1/2/4/5/6,
每张一个 87324 MiB 的引擎。我在 1 和 5 上的两跑当场死,判据清楚:
`Free memory on device cuda:0 (53.13/139.8 GiB) on startup is less than desired
GPU memory utilization (0.6, 83.88 GiB)` —— 各 25 个验收用例 error,全是 baseline
引擎起不来。GPU 7 上那跑没人碰,活到跑完。

教训:**在共享机上并行,要按"这张卡最近是否被别人反复占"选,不是按"此刻空着"选**。
后面改成串行 + 优先复用一直归我的卡,五个套件全部跑完。

### 2. 守候循环的模式串自匹配

我起的等待循环是
`while pgrep -f "isolated_cases.py capacity_eviction molmo2-4b"; do sleep 60; done`。
这个循环自己的命令行里**就含有那个模式串**,于是 `pgrep` 匹配到自己,被等的进程
早退出了它还在转。教训:守候的模式串要用字符类打断字面量
(`pytes[t]`),否则永不退出。

### 3. 0.5 GB 第一次重跑失败不是容量

第一次加 0.5 GB 重跑报的是
`Cannot reach the LMCache MP server at 'tcp://localhost:27660' within 300.0s`。
手起一个同参数(`--l1-size-gb 0.5`)的服务端,25 秒健康 —— 档位没问题。
场景的端口是 `25000 + os.getpid() % 5000` 推的,**跨 session 结构性可撞**。
换卡重跑就过了。这一次的教训是别把环境性失败读成被测量的失败。

### 4. 判"卡死还是在算",先看它有没有子进程 —— 我把一跑杀早了

qwen2-vl-2b 的全量 parity 起来 14 分钟日志 0 字节。本机**没有 py-spy,
ptrace 也被禁**(`strace` 直接 `Operation not permitted`),所以做了对照跑:
拿 gemma 起了个 `--limit 24` 的 smoke,**90 秒内打了 6.8 KB、模型已加载**
(15.47 GiB / 8.1 秒),顺手把门也跑过了:MP 路径、两个方向 flip 都是 0、
分数差 0.0、`pass2_hit_coverage=1.0`、`pass2_lookup_hit_ratio=0.9422`、
parse ratio 0.9583。**parity 跑法在 0.27.1 上通,这条成立。**

但我对那一跑的处置是错的。当时看到"CPU 停在 13:02、进入 `hrtimer_nanosleep`、
GPU 1 已被别人占满",判成疑似 wedge 就 SIGTERM 掉了。后来清进程时才看清:
那一跑留下了一个 `ppid=1`、`--role baseline`、`CUDA_VISIBLE_DEVICES=1`、
`TMPDIR=/tmp/mm27/p_qwen2vl2b` 的孤儿 —— **它当时是跑完前置处理、把 baseline
半程交给子进程、正在等子进程**,不是卡死。

日志 0 字节的原因**当天我写错了一次**,读了源码才对:`benchmark_parity.py:964`
的 `subprocess.run([...], timeout=7200)` **没有传 `stdout=`/`stderr=`**,子进程
是**继承**父进程的 fd,输出本来就直接落进同一个日志文件,不存在"被父进程
capture"。真实原因是两层块缓冲:父进程在 `main()` 里到 `:958` 才打第一行
(`[parity] N MME questions loaded`),那一行约 40 字节,在没有
`PYTHONUNBUFFERED` 的管道里躺在 8 KB 缓冲区里出不来;baseline 子进程继承同一份
环境,刚起来还没攒够一个缓冲区。所以 `PYTHONUNBUFFERED=1` **对这种情况是有用的**
(当天我判它没用,也错了),它现在已经在 `parity.sh` 里。

**决定性的检查是"它有没有子进程"(`ps --ppid`),我没做**,却去试了两个更重的
手段(py-spy、strace)并在都不可用后凭间接征象下了结论。这与 `6_` §三.3
"'ps 里没了'不等于'被杀了'"是同一族错误:用"我看不见它在干什么"代替
"去看它到底在干什么"。代价是白扔了 13 分钟的前置处理。

孤儿还有一层危害:它写的 `parity_qwen2-vl-2b.baseline.json` 与我重开那跑的
输出路径**同名**,两个写者一个文件。已 kill。

### 5. 新落盘诊断第一次就派上用场

被抢卡那两跑的报错直接给出子进程 stdout/stderr 的落盘路径,那行 OOM 一眼可见。
换成改之前的 `[-2000:]` 截尾,看到的会是一堆 LMCache 关停日志 —— 这正是 `6_`
§三.2 记的那个坑。

## 五、诚实边界

1. **parity 一个都没跑完**,所以"支持程度"这个词还没有可比的度量。`6_` §四 的
   那条边界依然成立。
2. 第一跑 qwen2-vl-2b parity 的零输出**已定性**(§四.4):它在等 baseline
   子进程,不是卡死;是我杀早了。缓冲那一层**已观测确认**:11:21 qwen2-vl-2b
   打出 `[parity] 2374 MME questions loaded`,日志正好 35 字节 —— 加了
   `PYTHONUNBUFFERED=1` 之后那一行是即时落盘的,所以当天"`PYTHONUNBUFFERED` 没用"
   的判断确实是错的。同一分钟 baseline 子进程 4136903 起来。**A 相实测 13:05**,
   与我在 14 分钟处杀掉那一跑的时刻严丝合缝:它当时刚跨进 C 相。
   仍该补的只有一条:`benchmark_parity.py` 在 spawn baseline 前后各打一行进度
   —— 现在整个前置处理段(`configure_environment` + `load_items`,实测 >10 分钟)
   一个字都不打,日志 0 字节既可能是"在加载数据集"也可能是"真挂了",光看日志
   分不开。落盘那条不用做,子进程本来就写同一个文件。
3. **molmo2 那个"按请求整笔预留"的机制是推断**,没读 MP 存储路径。
4. **抢占只在三个路由到它的模型上验过**;另两个抽样模型该场景不适用,不是没测。
5. **gemma parity smoke 只有 24 题**,门虽然过了但样本太小:24 题的 flip 预算是
   0.12 个,题目只覆盖到 `code_reasoning` 一个类别。它证的是跑法通,
   不是这个模型在 0.27.1 上没有静默损坏 —— 0.23 上那次抓到损坏用的是全量 2374 题。
6. **时长数字不能当性能结论**:同机竞争激烈,同一批跑里 GPU 归属换过三轮。
7. 证书 schema 已经是 5(`01e6f317` 那次就为单路径 scope 升过),缺的是
   vLLM 版本字段。
8. 三个提交只在 `multi_modal` 上,verify 树是 cherry-pick 未提交状态。

## 六、下一步

1. **qwen2-vl-2b 全量 MME parity**(GPU 7,在飞)。0.23 参照全在案:
   baseline / pass1 / pass2 三套都是 1966.06、0 flip,任何偏移都无从辩解。
2. **gemma-4-e4b 全量 parity**(GPU 3,在飞)。smoke 已证跑法通,加上对面把卡
   全放了,就不再串行等。它是唯一被 parity 抓过静默损坏的模型(0.23 上 54.3%
   flip、分数掉 920),6 KV 组 / 2 object group。
3. ~~`benchmark_parity.py` 的一条可观测性。~~ **已做,`7c72caf7` + `4904b9f4`**:`load_items`
   前一行(带 benchmark 与 limit)、原计数行补耗时、baseline spawn 前后各一行
   (带退出码与耗时),spawn 那行并写明子进程继承本流。落盘那条**撤销**,
   子进程继承 fd,本来就写同一个文件(理由见 §四.4 的更正)。
   `4904b9f4` 补上子进程那半:`run_baseline` 自己那次 `load_items` 之前/之后各一行,
   并且子进程的行打 `[parity:baseline]` 而不是 `[parity]` —— 它写的是父进程继承来的
   stdout,同一个前缀会把两个进程的输出混在一起,分不清是谁停了。
   注意:这些进度行**改在 `multi_modal`,没动 `multi_modal_verify`** ——
   在飞的两跑从 verify 树起,而 baseline 子进程是 `sys.executable __file__`
   重新从磁盘读的,改测量树会当场换掉子进程执行的代码。下一轮 parity 才吃到。
4. ~~schema 加 vLLM 版本字段。~~ **已做,`4720a75d`,schema 5→6**:新增 `runtime`
   块,记 vllm / torch / transformers / lmcache 的版本**和 import 解析到的路径**。
   路径不是冗余 —— `lmcache` 是 editable 装的,解析到哪个 worktree 取决于
   `PYTHONPATH`(实测无 pyguard 时解析到 `multi_modal/lmcache`,不是测量树),
   一张"写着 commit A 却导入了 B 树"的证书比不写更坏。两个查询都不 import 模块。
   `certify.py` **端到端出证还没做**,要卡。
5. **parity 的 26 分钟前置浪费**(新发现,还没动手)。`MMEBenchmark.load_items`
   把 1097 张图逐张 PIL→PNG→base64 编出 data URI(`:231-232`),父子进程各做一遍
   = 每跑 ~26 分钟纯 CPU。直接把 items 传给子进程不划算(那是 1-3 GB 的 base64),
   该做的是把编好的 URI 按 qid 落一层磁盘缓存,父子和后续所有跑都共用。
   这是独立的一个改动,不在有跑在飞的时候动。
6. 上游:`defer_block_free` 活锁的最小复现(脱离 LMCache);`#4463` 补
   "MP 不受影响"实测。
7. 1/2 出结果就 move on 到新模型支持(用户本轮明确的目标)。

## 七、这两跑怎么盯(监控口径)

盯法脚本在 `records/2026/08/26/run0271b/`:`pstat.sh` 一次性快照,
`pwatch.sh` 常驻事件流(只在状态**跃变**时出一行,静默是正常的)。

相位机是从 `benchmark_parity.py` 的 `main()` 读出来的,不是猜的:

| 相 | 代码位置 | 外部可见特征 |
|---|---|---|
| A | `configure_environment()` + `benchmark.load_items()` | 日志 **0 字节**、**无 GPU**、**无子进程**;实测 qwen2-vl-2b **13:05**、gemma-4-e4b **12:31** |
| B | `:958` `print("[parity] N MME questions loaded")` | 日志出现第一批字节 |
| C1 | `:964` spawn `--role baseline`,而**子进程在 `:803` 又把数据集重载一遍**才在 `:804` 建 `LLM` | 子进程在,**但不占 GPU**;日志不长 |
| C2 | 子进程引擎起来,开始跑 baseline | **GPU 记在子进程头上** |
| D | 子进程退出,父进程起 MP server + 自己的引擎 | GPU 转到父进程,`.mp_server.log` 出现 |
| E | 报告写出 | `<out>.json` 存在 |

**C1 这一相是今天量出来的,之前不知道**:`run_baseline()`(`:785-821`)自己调
`benchmark.load_items(limit)`,父进程加载过的 2374 题在进程边界上传不过去。
所以每一跑在碰到 GPU 之前有 **~13 + ~13 = 26 分钟**纯 CPU 的静默前缀,而且这
26 分钟里日志只有一行。这是"日志不动"最大的一个良性来源。

判活的四个信号,按 discriminating 能力排序:

1. **有没有 `--role baseline` 子进程**(`ps --ppid`)。这是唯一能把"在等子进程"
   和"卡死"分开的信号,也正是 §四.4 我漏掉的那一个。
2. **GPU 占用记在谁头上**(`nvidia-smi --query-compute-apps=pid,used_memory`)。
   A / C1 两者都是 0;C2 在子进程;D 在父进程。**注意 C1 不占 GPU 是正常的。**
3. **CPU tick 增量**(`/proc/<pid>/stat` 的 `utime+stime` 差)。这一条今天改过:
   **`ps -o %cpu` 不能用作判活** —— 它是"CPU 时间 / 存活时长"的**整段平均**,
   父进程只是阻塞在 `subprocess.run` 里就从 100 一路读到 71、55;反过来真卡死的
   进程也会在很长时间里继续读出高值。要看瞬时,只能采 tick 差。实测两个 baseline
   子进程 4 秒各烧 501 / 500 ticks = 满一个核,在算。**而且 C 相该采的是子进程,
   不是父进程** —— 父进程此时理应闲着。
4. **日志字节数 + mtime**。注意:**头 26 分钟只有一行是正常的**(见 C1),
   不是征象。

还有一条便宜的旁证:`ls -l /proc/<child>/fd/1` 直接指向我的日志文件 ——
这就在 fd 层面证了子进程是继承而非被 capture,不必再从行为上推。

`pwatch.sh` 的失败面覆盖(不是只等好消息):进程消失但没写报告 → 出事件并带
日志尾 12 行;新增日志里出现 `Traceback` / `CUDA out of memory` /
`Cannot reach the LMCache MP server` / `baseline subprocess failed` /
`AssertionError` / `Killed` → 出事件;日志 30 分钟不长 **且** 无子进程 **且**
cpu<5 → 出一次 "suspect, not proof"。两跑都到终态就退出。

pgrep 模式写成 `benchmark_parit[y].py`,免得像 `6_`/`7_` 那次一样匹配到自己。

两条盯法本身的教训:

* **失败模式写窄一点,否则告警变噪音。** 我把被抢卡那次的报错原文
  `Free memory on device ... is less than desired` 只截了前半句进 grep,结果
  qwen2-vl-2b 的 baseline 引擎正常起来时就报了一次假警 —— vLLM **每次**健康启动
  都打这行(`gpu_worker.py:789`)。已收窄到 `on startup is less than desired`。
  假的失败信号比漏报更伤:它会让人开始忽略这条流。
* **盯法要能重启而不重放。** 前两版的相位只在内存里,改一次 pattern 就得重启,
  重启又把 A->B->C1->C2 当新跃变重播一遍。现在相位落 `.pwatch_state`,重启接着走。

顺带一条卡的旁证:qwen2-vl-2b 的 baseline 引擎报
`Free memory on device (139.29/139.8 GiB)` —— GPU 7 整张都是我的,占了 86732 MiB。
这一跑不会重演 `r2_*_gpu_evicted.log` 那两次被抢卡。
