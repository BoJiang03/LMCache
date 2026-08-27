# 0.27.1 支持度:按维度抽样 6 跑并行开测(结果待出),preemption 按已知上游缺陷摘出

**日期**: 2026-08-26(当天第 5 篇,接 `4_`)
**代码状态**: `multi_modal@9c5a7d0f`,工作树干净(**本篇无仓库代码改动**;
改的只有 verify 工作树的分支位置与 scratchpad 里的启动脚本)
**被测树**: `multi_modal_verify@2485fdbc` = `9c5a7d0f` + 三个 base 修复 cherry-pick
**日志**: scratchpad `run0271/`,跑完后归档到 `records/2026/08/26/run0271/`

## 结论先写

1. **抽样按"改变 LMCache 侧行为的维度"选,不按模型数选。** 12 个注册模型压到 5 个:

   | GPU | 模型 | 代表的维度 | 0.27.1 上测过? |
   |---|---|---|---|
   | 0 | qwen2-vl-2b | 非混合,单 KV 组 —— 对照组 | 改造前测过(26+2) |
   | 1 | qwen3.5-2b | RECURRENT_STATE(GDN)混合,chunk 544 | 改造前 27/27 |
   | 3 | **gemma-4-e4b** | SLIDING_WINDOW 混合,**6 个 KV 组 / chunk 32 / 2 个 object group** | **从未** |
   | 5 | **molmo2-4b** | 唯一"非混合 + mm-prefix-LM",且 no-system / media-first / 前缀不稳 | **从未** |
   | 6 | **qwen3-omni-30b** | 唯一 audio + cross-modal | **从未** |
   | 7 | qwen2-vl-2b | 只跑 `preemption`,25 分钟硬超时 | 单独确认失败 vs 挂死 |

   跳过的 6 个及理由(**理由是模型属性,不是"懒得跑"**):
   - qwen2.5-vl-3b / internvl3.5-2b / glm-4.6v-flash / qwen3-vl-2b:LMCache 侧同形
     (非混合、单 KV 组、image+video、system 角色可用、媒体前缀稳定),由 GPU0 代表。
     qwen3-vl-2b 原本多一条 deepstack,`4_` 已删,现在与 qwen2-vl-2b 无差别。
   - qwen3.6-27b / qwen3.8-27b:与 qwen3.5-2b 同为 RECURRENT_STATE、chunk 784 vs 544,
     只是大一个数量级(L1 要 200 GB、每块 ~205 MB),由 GPU1 代表。
   - gemma-3-4b:与 gemma-4-e4b 同族,**留作 gemma-4 出红时的对照**
     (它是把 chunk 16 与多组 sliding window 分离开的那个受控比较,见其 spec 注释)。

2. **gemma-4-e4b 是信息量最高的一跑。** vLLM ≥0.26 的 packed subpaged 分组问题
   (上游 **#4731**,closes #4701,**未合**)正打混合模型的多 KV 组
   ——"混合模型在 vLLM≥0.26 只存 1/N 内核页"—— 而它是套件里唯一的多组
   sliding-window(5 组 SlidingWindowSpec + 1 组 FullAttentionSpec,chunk 32,
   2 个 object group)。qwen3.5-2b 的 GDN 是另一条几何,替不了它。

3. **`preemption` 从 5 个套件里 `-k` 摘掉,理由不是省时间而是"已知且与模型无关"。**
   `3_` 已把病灶钉在 vLLM 0.27.1 的 `defer_block_free`,触发条件是
   `max_concurrent_batches > 1 且 kv_transfer_config.is_kv_consumer` ——
   **任何** `is_kv_consumer` 连接器 + 默认开着的 async scheduling 都会踩,
   与模型无关。重复量 5 遍换不来新信息,只会换来 5 次活锁挂机(实测一次活锁
   3 分钟刷 13,544 条日志、要手动 SIGUSR1)。GPU7 单独跑一遍,只为回答一个
   改造后尚未确认的问题:**它到底是判失败,还是挂死不返回。**

4. **结果未出**(见 §四)。截至写作:各 1–3 个用例,`molmo2-4b` 第一个用例已 F。

## 一、跑法(可复跑)

```
verify 树   /home/bo/LMCache-worktrees/multi_modal_verify @ 2485fdbc
venv        /home/bo/venvs/vllm-mm/bin/python   (vLLM 0.27.1)
PYTHONPATH  <scratchpad>/pyguard : <verify 树>      # editable 劫持守卫,必带
HF_HUB_CACHE=/raid/data/hub                        # 5 个模型全部已缓存
TMPDIR      /tmp/mm27/<tag>                        # 必须短:sockaddr_un 107 字节
pytest      . -q -k "not preemption"               # preempt 那趟是 -k "preemption"
```
启动脚本 `run0271/launch.sh`(第 5 个参数是 TMPDIR tag,见 §三教训 1)。

用例数(已扣掉 preemption):qwen2-vl-2b 28、qwen3.5-2b 27、gemma-4-e4b 27、
molmo2-4b 25、qwen3-omni-30b 30。

**时长的大头不是用例数**:08-21 的 XML 显示 34 个用例 741 秒里,593 秒是
4 个独立场景各自**单独起一个引擎**;剩下 20 多个 acceptance 用例共用会话引擎、
每个不到 1 秒。所以每个模型的时长 ≈ 基线 + 会话引擎 + (场景数 × 2~3.5 分)。
场景数(扣 preemption 后):qwen2-vl-2b 3、qwen3.5-2b 2、gemma-4-e4b 2、
molmo2-4b 2、qwen3-omni-30b 3。

## 二、verify 树重新对齐到 PR HEAD

原来是 `4e749b5a` + 3 个 base 修复(`b4a06dec`),`4_` 的两张 0.23.0 绿量在那上面,
不含 `0badaae0`(文档)、`01e6f317`(证书 schema)、`9c5a7d0f`(revert in-process 修复)。
本轮 `git rebase --onto 9c5a7d0f 4e749b5a` → `2485fdbc`,无冲突
(revert 只碰 `vllm_v1_adapter.py`,三个修复碰的是 platform / transfer)。
现在测的就是 PR HEAD + 三个 base 修复。

## 三、教训

1. **TMPDIR 用模型名做 key 会撞车,而启动脚本第一件事是 `rm -rf`。**
   GPU7 的 preempt 对照跟 GPU0 的完整套件都是 `qwen2-vl-2b`,于是
   `T=/tmp/mm27/$1; rm -rf "$T"` 把对方启动到一半的目录删了。
   **key 必须按"这一趟"而不是"这个模型"取**,脚本加了第 5 个参数做 tag ——
   和 `1_` §三.3 那条"常量选档要按它真正依赖的维度 key"是同一个错的两次犯。
   两趟都重启了,没有把污染的数据当结果。
2. **杀并行跑之后要按"实际 GPU"而不是"进程 env 里的 `CUDA_VISIBLE_DEVICES`"清残留。**
   同一台机器上别人的老进程 env 里也写着 `CUDA_VISIBLE_DEVICES=1`,但它们的
   映射是各自启动时的;拿 env 当物理卡号会误杀。要用
   `nvidia-smi -i <n> --query-compute-apps`。
3. **基线子进程退出会让显存瞬间掉到接近 0**,别把这一幕当成"跑挂了"。
   看到 GPU1/5/6 从 40~107 GB 掉到 2 GB 时我先怀疑自己误杀,查了父进程才确认
   那是 `compute_baselines` 子进程正常退出、会话引擎还没起。
4. **"抽样节约时间"要把跳过的理由写成模型属性。** 写"跟 qwen2-vl-2b 同形"
   才可复核、才能在新模型注册时自动落到正确的一侧;写"这几个不重要"就是
   把覆盖面偷偷缩小 —— 这是 `isolated_routing.py` 模块注释里记的同一条纪律。

## 四、诚实边界

1. ~~**本篇写作时六跑全部在飞,一个结论都没有。**~~ **结果已出,见 `6_`:
   五个套件 4 全绿 + molmo2-4b 1 红(`capacity_eviction`,存了 0 字节),
   抢占那跑挂死。日志归档 `records/2026/08/26/run0271/`。**
2. **跳过的 6 个模型在 0.27.1 上仍然未测**,包括两个 27B。"同形"是我按 spec
   字段做的判断,不是实测等价。
3. **`preemption` 在 0.27.1 上对 5 个抽样模型未测**,按 `3_` 的机制外推为
   "全部会踩"。外推的依据是 vLLM 侧的开关条件与模型无关,但**没有逐模型验证**。
4. 六跑并行共享主机(160 核,load 27),彼此的时长互相影响;**本篇的时长数字
   不能当性能结论**。
5. 测的是 verify 树(PR HEAD + 三个 base 修复),不是 `multi_modal` 单独。
   三个修复要单独走上游。
6. 仍未做:证书 schema 加 vLLM 版本字段;MME parity 重跑;`certify.py` 端到端。

## 五、下一步

1. 六跑收结果,红的逐个归属;gemma-4-e4b 若红,立刻拉 gemma-3-4b 做受控对照
   (分离 chunk 32 与多组 sliding window 这两个变量)。
2. GPU7 的 preempt 对照回答"失败还是挂死";若挂死,套件层面要考虑给该场景
   加超时,否则 0.27.1 上任何完整跑都会卡住。
3. 结果稳定后再决定要不要补 qwen3.6-27b / gemma-3-4b / glm-4.6v-flash。
4. 上游:`defer_block_free` 活锁的最小复现(去掉 LMCache);#4463 补我们的
   独立复现;盯 #4731(混合模型 packed subpaged)是否影响 gemma-4-e4b。
