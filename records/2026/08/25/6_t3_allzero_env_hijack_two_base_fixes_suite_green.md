# T3 全零的真根因:venv editable 子模块劫持 + 两个 base 缺陷;修复落地,MP 套件转绿

**日期**: 2026-08-25(当天第 6 篇,接 `5_`;更正记录 5 的两处误判)
**代码状态**: `multi_modal@17a6ef47`(新增 1 个测试侧 commit),另有两条独立修复分支
`fix_memcpy_stream_order@9436769a`、`fix_mp_store_native_gate@7f32a2fe`(均 off dev,未推送)。
产物在 `vllm_upgrade/t3_allzero/`,共 5+ 项。

## 结论先写

1. **今天认证环境(vllm-lazy)T3 全零命中的根因是环境污染,不是滞后也不是负载。**
   因果链四环:
   - 今天 12:19,另一条工作流向共享 venv `vllm-lazy` 安装了指向
     `/home/bo/LMCache-worktrees/lazy_offloading` 的 lmcache **editable install**;
   - editable 的 meta-path finder 排在 PathFinder 之后:PYTHONPATH 保得住
     `lmcache` 主包(解析到 multi_modal),保不住 multi_modal worktree 里
     **不存在的编译子模块** —— `import lmcache.cuda_ops` 静默落到外树旧 .so
     (运行时 `__file__` 实证:包在 multi_modal,cuda_ops 在 lazy_offloading);
   - 旧 .so(早于 8/14 #4502)导出 `execute_object_group_transfer` 却没有
     `KernelGroupSpec` 等计划类型 → `bind_native` 逐符号绑定后能力错位:
     入口在、类型是会抛 `NotImplementedError` 的 stub → 服务端 store 选了
     native 计划路径,在 reserve_write 之后崩(服务端日志:
     "Cannot store keys due to exception ... KernelGroupSpec requires the
     cuda_ops native extension");
   - base 第二缺陷放大成全零:**store 失败不回滚预留**,键永久写锁
     (插桩:FW 全程 0 次;30 秒后 chunk-0 仍 NOT_READABLE;其余键
     NOT_EXIST;store 与 lookup 键 id 35/35 完全吻合 —— 键没错,锁死了)。
     前缀折叠把 chunk-0 一锁放大成整段 0 命中,负控/纯文本 repeat 全灭。
   - **08-22 同环境全绿、今天全红的分水岭就是 12:19 的安装**,与机器负载
     无关(venv 今天零包升降级,只动了 editable finder;12:19 前该 finder
     不指向带旧 .so 的树)。

2. **修复三件套已落地,各归其位(用户指正 PR 边界后拆分):**
   | 分支 | commit | 内容 |
   |---|---|---|
   | `fix_memcpy_stream_order`(off dev) | 9436769a | torch 回退 `lmcache_memcpy_async` 拷贝前 drain 当前 CUDA 流(记录 5 的腐坏 bug)+ 回归测试(无修复必红、有修复 4 后端全绿) |
   | `fix_mp_store_native_gate`(off dev) | 94dafb21 | `bind_native` 符号组门:native 模块缺任一计划类型则撤回 `execute_object_group_transfer` 并告警(本篇缺陷一) |
   | 同上 | 7f32a2fe | store 失败经 `rollback_write` stream 回调强删预留键(本篇缺陷二);23 项单测全过 |
   | `multi_modal`(本 PR) | 17a6ef47 | 仅测试侧:MPHarness store 落盘屏障(先等引擎侧提交计数达标,再等服务端 write_locked=0 且计数静默;超时非致命) |

3. **运行侧防线(共享 venv 零改动)**:pyguard sitecustomize(PYTHONPATH 第一项)
   剥掉 `__editable___lmcache*` finder → cuda_ops 干净缺失 → torch 回退路径
   (带 memcpy 修复后正确)。模板归档 `t3_allzero/pyguard_sitecustomize.py`,
   已写永久记忆 `vllm-lazy-venv-editable-hijack`。

4. **验证(vllm-lazy,vLLM 0.23.0,verify 树 = multi_modal + 三修复 cherry-pick + 守卫)**:
   - T3 mp_connector qwen2-vl-2b:**PASSED**(2 分 23 秒;写锁正常释放后
     屏障亚秒级通过;对照:锁死时屏障每请求超时 30s,12 分 22 秒后全零红)。
   - 完整套件(GPU 6/7 并行,双引擎同跑本身就是负载压力):
     **qwen2-vl-2b 29/29 PASSED**(13 分 12 秒)、
     **qwen3-vl-2b 34/34 PASSED**(13 分 14 秒),零失败零错误,
     T3 mp_connector 含在内。与 08-22 证书的用例数一致(29/34)。
     日志:`t3_allzero/suite_verify_qwen{2,3}.log`。

## 一、发现路径(接记录 5)

1. 用户指令"把 support test 过了,有问题就修问题" → 先落屏障+memcpy 修复。
2. 屏障后的 T3(lazy)仍红且形态突变:全场 0 命中(含纯文本 repeat 与负控),
   store 全落库(4608 tokens、195 键 resident)→ 排除时序,指向键/协议。
3. 存档比对:全零形态在我改动之前(13:20 的 lazy 运行)就存在;bisect venv
   同期是部分命中(滞后形态)→ 环境分水岭,不是代码。
4. 插桩重跑(archived sitecustomize):RR 全 KEY_NOT_EXIST→存后 30 秒仍
   NOT_READABLE、FW=0 次、键 id 完全吻合 → "finish_write 从不发生"。
5. 服务端日志抓到 store 崩溃栈(KernelGroupSpec stub)→ 逐层排查解析:
   `lmcache.cuda_ops.__file__` 指向外树 → editable finder(12:19 mtime)→
   四环闭合。dispatcher/native completion 单测全过,排除了回调机制本身。

## 二、证据与产物(`vllm_upgrade/t3_allzero/`)

| 文件 | 内容 |
|---|---|
| `mp_server_instrumented.log` | 服务端:RW ok 键 id、RR 全 NOT_EXIST/NOT_READABLE、FW=0、store 崩溃栈 |
| `engine_instrumented.log` | 引擎侧:patch 行自证 `torch_ops`/adapter 解析到 multi_modal |
| `t3_lazy_allzero_with_barrier.log` | 全零红的完整 pytest 输出(12:22,屏障超时叠加) |
| `t3_verify_green.log` | 修复+守卫后的 T3 绿(2:23) |
| `pyguard_sitecustomize.py` | finder 隔离守卫(可复用模板) |

关键代码:`bind_native`(`lmcache/v1/platform/base/device_ops.py`)、
store finally 块(`lmcache_driven_transfer.py` ~1185)、stub 定义
(`lmcache/v1/platform/ops_types.py:80`)、editable finder
(`vllm-lazy/.../__editable___lmcache_0_5_4_dev147_finder.py` → lazy_offloading)。

## 三、教训

1. **"同环境昨天绿今天红"先查环境本身**:site-packages 的 editable/pth 变更
   + 关键模块 `__file__` 运行时自证,五分钟能排除/坐实,比任何代码理论都快。
2. **PYTHONPATH 不是隔离**:PEP 660 editable finder 对"主包有、子模块无"的
   worktree 会静默混装两棵树。凡 worktree 缺编译产物,必须显式验证每个
   native 模块的 `__file__`。
3. **hasattr 式能力检测必须按符号组**:逐符号 bind + 单符号 hasattr 门 =
   新旧扩展混配时选中跑不动的路径。能力要么整组宣告,要么不宣告。
4. **reserve→finish 两段协议必须有失败臂**:中间任何异常若不回滚,锁+内存
   双泄漏,且表现为"数据丢失"级的下游症状,极难从表象溯因。
5. 屏障(等 write_locked=0)意外成了锁泄漏的检测器:每请求恰好超时 30s
   是"锁永不释放"的显著信号。
6. 插桩打印键 id(记录 5 教训 1)这次直接把"键错"假设一击毙命(35/35 吻合),
   把排查从键空间拉回生命周期 —— 该教训已连续两次兑现。

## 四、诚实边界

1. 12:19 的 editable 安装归属未查(哪个会话/谁装的);只确认了 mtime 与指向。
   卸载与否是用户的决定(另一条工作流可能依赖它)。
2. verify 树的 T3 绿走的是 torch 回退路径(cuda_ops 干净缺失)。**native
   计划路径(新 .so + 完整符号组)未验证** —— 需要重建扩展,共享环境不允许;
   能力门保证的只是"跑不动就不选",不是"native 路径正确"。
3. 修复分支基于今日 dev(09bc14c0);上游若已动过这些文件需 rebase。
4. 记录 5 的滞后与 memcpy 结论在 bisect venv 证据下仍然成立;被更正的只有
   "负载解释 08-22/今天差异"与"lazy=cuda_ops 正常"两点。
5. 双模型套件绿是抽验(qwen2/qwen3 系),不是 12 模型重认证;verify 树是
   一次性验证分支(multi_modal + 三修复 cherry-pick),不推送,修复合入
   dev 后由 mm 分支 rebase 取得。

## 五、下一步(等指令)

1. **用户已决定(2026-08-25):`fix_mp_store_native_gate` 的两个问题不发
   PR,用户自行修复。** 分支留在本地作参考(已验证的修复 + 23 项单测,
   随取随删);上游报告清单里也不再单列这两项。
2. **用户已决定(2026-08-25):memcpy 流序修复也不归 mm PR 管** ——
   至多在 PR 文案里一句带过(cuda_ops 正常的环境不触发,mm PR 不依赖它);
   正式归宿是上游报告。分支 `fix_memcpy_stream_order` 留本地作参考
   (修复 + red/green 回归测试)。待推送仅剩 `multi_modal`(mm PR,
   含屏障 commit 17a6ef47),推 fork 需明确指令。
3. 上游报告清单(其余项):0.26.0+ fused 布局 hit 腐坏(记录 4)、torch
   回退 memcpy 流序(记录 5)、store 可见性滞后(记录 5)、CI 命中门空转
   (记录 3)、MP handler 崩溃不回包(记录 4)。
4. 是否让环境所有者处理 vllm-lazy 的 editable(或为 multi_modal 重建
   cuda_ops)—— 用户决定。
5. 其余 10 个已认证模型是否用 verify 配方补一轮 T3 抽验。
