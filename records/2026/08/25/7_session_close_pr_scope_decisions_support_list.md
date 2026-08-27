# 会话收尾:PR 边界三决定、支持模型清单交付、分支与环境台账

**日期**: 2026-08-25(当天第 7 篇,接 `6_`;本篇是会话日志,技术实质在 `5_`/`6_`)
**代码状态**: `multi_modal@17a6ef47`,工作树干净(本篇不新增提交,17a6ef47
即本会话唯一的 mm 分支提交:MPHarness store 落盘屏障)

## 一、本会话主线(时间序)

1. 用户指令:"我不在乎你到底干啥,我要求是把 support test 过了,有问题就修问题。"
2. 修复落地(过程与证据在记录 6):memcpy 流序修复 + 回归测试、
   bind_native 能力门、store 失败写锁回滚、MPHarness 屏障、pyguard 守卫。
3. 用户中途指正 **PR 边界**("你怎么修到了非此 pr 的地方了")→ base 修复
   全部拆出 mm 分支,落到两条独立 dev-based 分支。
4. T3 全零真根因定位(venv editable 子模块劫持 + 两个 base 缺陷,记录 6)。
5. 验证:T3 绿(2:23);完整套件 qwen2-vl-2b **29/29**、qwen3-vl-2b
   **34/34**(GPU 6/7 并行,各 ~13 分钟)。**support test 目标达成。**
6. 交付支持模型清单(用户要求 md 可复制,英文,反复精简:去 Status 列、
   去括号注解、去 Blocker 列 —— 教训:**给用户的复制件从最简开始给**,
   分组本身就是标记,别加冗余列)。

## 二、用户决定(约束后续工作)

1. **`fix_mp_store_native_gate` 不发 PR** —— 能力门 + 锁泄漏两个问题用户
   自行修复。分支留本地作参考(2 commit + 23 项单测,随取随删),
   上游报告清单不再单列这两项。
2. **memcpy 流序修复不归 mm PR 管** —— 至多 PR 文案一句带过(已给英文
   一句话模板,在会话里),正式归宿是上游报告;分支
   `fix_memcpy_stream_order` 留本地作参考。
3. mm PR 保持纯测试侧(唯一新提交 17a6ef47);待推送仅 `multi_modal`,
   推 fork 需明确指令(沿用长期约束)。

## 三、台账(本会话结束时)

| 项 | 状态 |
|---|---|
| `multi_modal` | @17a6ef47,领先 fork 15 提交,未推送 |
| `fix_memcpy_stream_order`(off dev) | @9436769a,本地参考,不发 PR |
| `fix_mp_store_native_gate`(off dev) | @7f32a2fe(2 commits),本地参考,用户自修 |
| `multi_modal_verify` worktree | 一次性验证树(mm+三修复 cherry-pick),可删 |
| `fix_*` 两个 worktree | 留存;删除时记得先 cd 回主树 |
| vllm-lazy venv | 仍带 editable 劫持(不动共享环境);跑测试必须加 pyguard sitecustomize(模板在 `vllm_upgrade/t3_allzero/`) |
| 永久记忆 | 新增 `vllm-lazy-venv-editable-hijack` |
| 证据归档 | `vllm_upgrade/t3_allzero/` 7 项(插桩双日志、红/绿对照、守卫模板、双套件日志) |

## 四、支持模型清单(已交付版本,便于日后复用)

Branch: `BoJiang03/LMCache:multi_modal`。
Supported(认证序):qwen2-vl-2b、qwen2.5-vl-3b、internvl3.5-2b、
qwen3-vl-2b、glm-4.6v-flash、molmo2-4b、gemma-3-4b、gemma-4-e4b、
qwen3.5-2b、qwen3.6-27b、qwen3.8-27b、qwen3-omni-30b(12/12,
证书 `records/2026/08/22/all12/`)。
Queued:DeepSeek-OCR、Mistral Small 3.1 24B、MiniCPM-V 4.6。
Whisper 排除。最终交付格式:两组两表,仅 #/Model/HF ID 三列,无状态列
无阻塞列(用户明确要求最简)。

## 五、下一步(等指令)

继承记录 6 §五(1、2 已由本篇 §二的用户决定改写):
1. 推送 `multi_modal` 到 fork(需指令)。
2. 上游报告:fused 布局腐坏(记录 4)、memcpy 流序(记录 5)、可见性滞后
   (记录 5)、CI 命中门空转(记录 3)、handler 崩溃不回包(记录 4)。
3. vllm-lazy 环境处置(editable / 重建 cuda_ops)—— 用户决定。
4. 其余 10 模型的 T3 抽验(verify 配方现成)。
