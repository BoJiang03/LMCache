# 全停:孤儿 MP server 泄漏、records 上 dev 分支、钩子放行口径

**日期**: 2026-08-27 12:23(当天第 8 篇,接 `7_`)
**代码状态**: `multi_modal@6b70d199`(= c37a3d38 + records 提交),工作树干净
**推送**: `fork/multi_modal` = 6b70d199,本地已对齐,不再分叉

## 一、用户这一轮的三条指令

1. 「不是我的代码的问题就不归我管,策略是**避开**」→ `7_` 已记,不改 `lmcache/`。
2. 「**实验全部停了**」→ 我停了自己的三个臂和循环。
3. 「重点检查我这个用户启动的 vllm、lmcache 等占资源的服务,**现在全部停了**」→ 全库清空。
4. 「records 也可以走,push 到 fork 的 multi_modal **dev 分支**;**PR 有专用分支,repro 也有专用分支**」
   → fork 上的分支分工确认:`multi_modal`(开发,可带 records)、
   `multi_modal_pr`(PR,79d90fd8)、`multi_modal_repro`(复现,4a21af88)。

## 二、清场结果:8 张卡全空

按进程组杀,全部属于 bo:

| 进程组 | 内容 | GPU | 代价 |
|---|---|---|---|
| 2512411 | `chain4.sh 3`:qwen3.8-27b parity,跑了 43 分钟 | 3 | 报告未出;同队列的 glm certify 之前已完成,证书已落盘 |
| 2615646 | `chain4.sh 2`:internvl3.5-2b certify,38 分钟(pytest 中) | 2 | 白跑;该模型的 schema 8 / stable 证书早已出,不受影响 |
| 2661675 | 12:16 另一会话新起的 phi4-mm parity(GPU 1) | 1 | 才 5 分钟,损失小 |
| 2596585 / 2600824 | **我杀实验时留下的两个孤儿 MP cache server** | 5 / 2 | **占着 119.5 GB + 14.8 GB 不放** |

**教训(重要,写进操作口径)**:`benchmark_parity.py` 会拉起
`lmcache.v1.multiprocess.http_server` 子进程。只杀主 python 进程时,这个 MP server
会变成 `ppid=1` 的孤儿**继续占满显存**,而且**不响应 SIGTERM,要 SIGKILL**。
以后停这类跑必须按进程组:`kill -TERM -<pgid>`,然后核对 `nvidia-smi` 归零,
再对残留的 `http_server` 补 `kill -KILL`。

清场后:GPU 0-7 全部 ≤ 2.5 GB;GPU 0 上残留的 2.4 GB 与几个小进程属于 **root 与
rui 的 lmcache server**,不是我们的,没碰。bo 名下再无 vllm/lmcache/pytest/parity/
certify 进程,RSS > 1 GB 的一个都没有。只剩两个别的会话开的 `tail -F` 日志跟随。

## 三、推送与钩子口径

- **代码**:`fork/multi_modal` 991a88c3 → c37a3d38(43 个提交,fast-forward)。
  其中 c37a3d38(`test(e2e_mm): make phi4-mm's suite runnable`)是**另一个会话**在本
  worktree 里的产出,这次推送把它一并保存了。本会话没改任何源码。
- **records**:再叠一个提交 → **6b70d199**,只加 464 个文件、全在 `records/` 下,
  其余目录零改动。署名 `Bo Jiang <bo.jiang@temple.edu>`,无 Co-Authored-By。
- **pre-push 钩子改了口径**。原钩子(8/19 装,`/home/bo/LMCache/.git/hooks/pre-push`)
  禁止任何带 `records/` 的推送;现在改为:**只有推到 fork 的
  `refs/heads/multi_modal` 才放行**,其余一律拦(已实测:把同一提交推
  `multi_modal_pr` 会 BLOCKED)。旧钩子备份在
  `$SP/pre-push.bak`。钩子是 `.git` 共享的,对所有 worktree 生效。
- **当时为什么用 plumbing 而不是普通 commit**:推送时另一会话有 certify 在飞,
  在本 worktree 提交会把它们的证书盖成 `stable:false`。所以用
  `GIT_INDEX_FILE` + `read-tree` + `update-index --add` + `commit-tree` 造提交直接
  推,不碰共享的 index / HEAD / 工作树。本篇 /records 时所有跑都已停,已用
  `git reset 6b70d199` 把本地 ref 对齐,分叉消除。
- **`.git/info/exclude` 的 `records/` 那行保留不删**。理由:若删掉,在检出
  `multi_modal_pr` 的 worktree 里,盘上这 464 个 records 文件会变成 untracked,
  一次 `git add -A` 就可能扫进 PR 提交(exclude 里那段注释记的 2026-08-21 事故
  正是这样发生的)。所以 dev 分支上新增 record 用 `git add -f <file>` 显式提交,
  exclude 继续当「别被 -A 扫进去」的护栏,pre-push 钩子当第二道。

## 三.5、写入范围核对(用户 12:23 追问)

- **上游 `origin`(LMCache/LMCache)零写入**:它根本没有 `multi_modal` 分支,
  `git branch -r --contains` 对 c37a3d38 与 6b70d199 都是 0 个 origin 分支。
- **fork 上只有 `multi_modal` 动过**:`multi_modal_pr` 仍是 79d90fd8、
  `multi_modal_repro` 仍是 4a21af88,与推送前逐字节一致。
- **没有建任何 PR**:`gh pr list` 列出的 4499/4444/4442/4432/4418 都是既有的,
  本会话一次 `gh` 写操作都没做。
- 远端之外的本地副作用共四项:pre-push 钩子口径(备份
  `$SP/pre-push.bak`)、本地 `multi_modal` ref 由 `git reset` 前进到 6b70d199、
  删掉 `/raid/data/hub` 下 79 G 的 Llama 4 部分权重(我自己下的,按指示)、
  按指示杀掉 bo 名下的跑。

## 四、当前证书面(未变)

| key | verdict | schema | stable |
|---|---|---|---|
| deepseek-ocr / gemma-4-e4b / kimi-vl-a3b / mistral-small-3.1-24b / molmo2-4b / qwen2-vl-2b / qwen2.5-vl-3b / qwen3-omni-30b / qwen3-vl-2b / qwen3.6-27b / **glm-4.6v-flash** | SUPPORTED | 8 | ✅ |
| gemma-3-4b / internvl3.5-2b / phi4-mm | NOT_SUPPORTED | 8 | ✅(诚实证书) |
| qwen3.5-2b | SUPPORTED | **3** | 陈旧(活锁,类已由 qwen3.6-27b 覆盖) |
| qwen3.8-27b | SUPPORTED | **3** | 陈旧(parity 被本轮清场杀掉,未刷新) |

分类学:16 个类填了 14 个,空的两个(**均匀滑窗单组**、**模态 LoRA 进 key**)都只有
phi4-mm 一个候选,卡在 `7_` 那个上游抢占缺陷上;DeepStack 那一格是「绿但无活检测器」。

## 五、下一轮的入口(按用户口径,全在 tests/e2e_mm/)

1. `benchmark_parity.py:1129` 的 `gpu_memory_utilization=0.6` 硬编码改为走 spec 已有
   字段并抬高,使块池不会耗尽 → 抢占不发生。
2. 加「本次运行抢占数必须为 0」的前置校验,复用套件已有的
   `vllm_preemption_total()`(T0.11 已在用);抢占 > 0 判**报告无效**(环境因素),
   不判模型红。
3. certify 加一条 exclusion 点明上游成因(仿 `DEEPSTACK_NOT_COVERED` 的写法);
   `IN_PROCESS_NOT_COVERED` 的措辞改为 out-of-scope(in-process 已正式放弃)。
4. 然后重跑 phi4-mm / gemma-3-4b / internvl3.5-2b 的 parity + certify;phi4-mm 一张
   证书同时填掉两个空类。qwen3.8-27b 顺带刷到 schema 8。

放弃项:Llama 4(用户定,权重已删)、Step-3(最小 fp8 328 GB,需 TP≥3,出范围)。
