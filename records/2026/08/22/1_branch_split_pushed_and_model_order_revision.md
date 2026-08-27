# 分支三分、历史重写、全部推 fork,以及模型顺序改按覆盖维度排

日期:2026-08-22(00:00–00:10,接 `../21/15_` 之后)
分支状态:三条都已推 fork,本地与远端 IN-SYNC;`origin` 一个分支没碰,
PR 未创建(用户明确:PR 他自己手动做)

```
09bc14c0  (dev @ 08-18)
 ├── multi_modal_pr    79d90fd8   3 commits, 9 files, +384 −112   ← 要发的 PR
 └── multi_modal       991a88c3  33 commits                        ← 工作分支
      └── multi_modal_repro 4a21af88  34 commits                   ← 证据分支
```

本条记三样:**两次历史手术的方法与校验**、**提交信息卫生这条新规矩**、
**模型顺序的重排理由**。

## 一、artifact 历史重写(用户批准后)

要清的是 `git add -A` 扫进来的 28 个 run artifact(见 `../21/15_` 第二节),
blob 在 `416fdaa2..e33973a8` 五个提交里,约 1.7 MB。

**方法:按路径过滤重放代码 diff,不用 cherry-pick。** 原因是
`47f9e183` **删** `certificate_gemma-4-e4b.json`、三个提交**改**
`suite_gemma-4-e4b.xml` —— 在一段这些文件从未存在的历史上 cherry-pick,
每一个都会撞 modify/delete 冲突。先枚举确认"被碰过的代码文件集合是封闭的"
(五个文件),再对每个提交取 `git diff $c~1 $c -- <那五个文件> | git apply --index`。

`e33973a8` 整个被丢掉 —— 它**只**删 artifact,零代码改动,在新历史里是空的。

两道校验(写进脚本,失败即 exit):

1. **重写后的树与 `e33973a8` 逐字节相同** —— 证明只去掉了 artifact,没碰代码;
2. **新范围里 0 个 artifact object 可达**(`git rev-list --objects | grep -c`)。

旧尖端留在本地 `backup/pre-artifact-rewrite`(`e33973a8`),那 1.7 MB blob
现在只存在于这一个 ref 下。**未推,也不会推。**

## 二、PR / 证据拆分

用户的话:"那 20 提交其实可以放 repro branch,我们不会 merge 进 dev,
但是 pr info 里面会用到。"

清点:33 个提交里**只有三个**碰生产代码或文档:

| 提交 | 内容 | 是否混 |
|---|---|---|
| `d1c44a5c` | 生产 + 单测 + 设计文档 | 干净 |
| `3dced28f` | MP 连接器修复 **+ 套件** | **混的** |
| `755272e1` | 纯设计文档 | 干净 |

`3dced28f` 按同一套路径过滤拆半,生产半边进 PR 分支,套件半边留在
工作分支的原位。两道校验:

1. **PR 分支在 `tests/e2e_mm/` 之外与 `multi_modal` 逐文件相同** ——
   任何差异都意味着"有生产改动只活在测试提交里";
2. **PR 分支上 0 个套件文件**。

**校验抓到一处真东西:根目录 `pytest.ini`。** 它被套件提交 `c3a172f8`
改过,加的唯一一行是注册 `mm_e2e` marker。判断:这是套件配置,dev 不需要,
归到套件那边 —— 但**显式列进 `SUITE_PATHS` 并写理由**,不是笼统排除。
第一版校验就是靠"不许有例外"把它逼出来的,如果一开始就写宽,它会静默
跟着 PR 走。

`multi_modal_repro` 用 `git rebase --onto multi_modal 977fdf19` 把它唯一的
独有提交(repro/ 脚本)重放到套件之上 —— 两条分支在 `977fdf19` 分岔,
所以这是重放一个提交,不是重放整段。注意它在**另一个 worktree**
(`/home/bo/LMCache-worktrees/multi_modal_repro`)里 checkout 着,
必须在那边执行,否则 `fatal: already checked out`。

## 三、新规矩:提交信息只能引用自己包含的东西

拆分暴露了一个之前看不见的问题:**两条生产提交的信息都引用了不在自己
里面的内容。**

- `3dced28f` 的标题 `Cover T3 MP path, video modality, and preemption
  recompute` —— 讲的是套件覆盖,而套件半边已经不在这个提交里了;
- `d1c44a5c` 的 Verified 段写"新增的 tests/e2e_mm 验收套件(22 个测试,
  含 800 图碰撞压力测试)通过" —— 那些文件根本不在这个分支上。

第二条是我第一轮**漏掉的** —— 只盯了明显混装的那个,没有回头核第一条。
重写后:

- `b1cc4e31`(→ `5d8e6d51` 之后)标题改成
  `fix(vllm): key every MP connector operation on MM-adjusted token ids`,
  正文只讲本提交做的事,并补上一条设计理由:**替换放在 tracker 而不是
  每个调用点,是为了让"每个操作都走它"成为可检查的断言** ——
  一个请求只有一个 key token 来源;
- `5d8e6d51` 的 Verified 段改成"单测在这里,800 图压力测试和 #3301
  复现脚本在 `multi_modal_repro`,不进 dev"。

规矩:**提交信息只能引用自己包含的东西;证据在别的分支就点名那条分支
并说明它不进 dev。** 拆分是这条规矩的照妖镜 —— 混装时它看起来完全正常。

改完复查:**树与改消息之前逐字节相同**(0 行漂移)。

## 四、推送:一个把我的判断纠正了的事实

三次推送,只有一次真的需要 force:

| 分支 | 结果 |
|---|---|
| `multi_modal_pr` | `[new branch]` |
| `multi_modal` | `a3c6a2c3..991a88c3` —— **fast-forward,没用上 force** |
| `multi_modal_repro` | `+23f1bc64...4a21af88` (forced update) |

我事先说"两次 force",错了。`a3c6a2c3` 是第 13 个提交,artifact 重写从
第 29 个(`416fdaa2` 之后)才开始,所以 `a3c6a2c3` **仍然是新历史的祖先**。

**教训:"改过历史"不等于"需要 force"。要看远端 ref 是否仍是新 tip 的祖先
(`git merge-base --is-ancestor`),而不是看有没有做过重写。**

两次都带了 `--force-with-lease=<branch>:<推之前的远端值>`,锚在具体
sha 上,所以远端若被别人动过会拒绝而不是覆盖。

推之前核了两件事,三条分支都是 0:**没有 `records/` 路径被跟踪**、
**没有 run artifact 被跟踪**。

## 五、模型顺序:改按覆盖维度排,不按热度排

用户问"接下来支持什么模型,什么顺序"。`../21/2_` 那份排序是按
"热度 × 易支持度",现在瓶颈变了 —— 不是模型热度,是**覆盖维度**。
新顺序:

0. ~~**先修存储层,再加模型**。7 张 SUPPORTED 证书的 chunk 全在 544–784,
   没有一个到过那个压力。若 `KEY_NOT_READABLE` 与模型无关,这 7 张的
   适用范围要改写成"小 chunk 下不成立"。**在可能坏的存储层上继续刷证书,
   等于批量生产不可靠结论。**~~

   **【本条作废,见 `2_`】** 当天稍后去读证书本身,发现前提是假的:
   8 张证书里有 **5 张的 chunk 是 16**(qwen2-vl-2b / qwen2.5-vl-3b /
   qwen3-vl-2b / internvl3.5-2b / glm-4.6v-flash),**比 Gemma 4 的 32 更小**,
   且都是 2374 题全量 MME 全绿。所以"小 chunk 是触发条件"不成立,
   **这些证书不需要重写适用范围**。真正的区分维度是"多 object group +
   高对象数",Gemma 4 是唯一同时满足的。教训:我是从
   `hybrid_block_tokens` 那一列(544/784/32)反推的,而非 hybrid 的模型
   这一列是 0,chunk 走 `LMCACHE_TEST_CHUNK_SIZE`——**我把"字段为空"读成了
   "不在样本里"。**
1. **Gemma 3(4B/12B)**。Gemma 4 挂在存储层,所以它本该验的两件事
   **一件都没验成**:滑窗多组 KV,以及图像 token 双向 mask
   (vLLM #40106 —— 若真被静默忽略按因果跑,KV 内容本身就是错的,与缓存无关)。
   Gemma 3 还是上面那个 bug 的**独立证人**(非 Gemma-4 模型上复现小 chunk 压力)。
   前置未知:Gemma 3 的 5:1 滑窗在 vLLM 里是否也分裂成多组 KV。
2. **Qwen3-Omni** —— 第一个音频模型。8 张证书全写着
   `audio modality (no audio model registered yet)`。音频 placeholder 与
   图像同构(同一个 `mm_hash` 通道),**生产代码大概率零改动**;
   工作量在套件:MME 是图像基准,音频要换(AIR-Bench / MMAU),探针重做。
3. **搭车批**:DeepSeek-OCR、Mistral Small 3.1、MiniCPM-V 4.6、Molmo 2。
   一次跑一批,价值是样本量 —— **四个里有一个意外挂了比四个都过更有信息量**。
4. Llama 4 / Kimi-VL / Step-3(Llama 4 要看的是 iRoPE 与长上下文下的 chunk)。
5. **Phi-4-multimodal —— 它其实是一次接口改动**,不是一个模型。模态 LoRA
   必须进 key,做它等于做 `../20/5_` 那次 `extra_keys` 重构。等前面的样本量
   把接口需求定清楚再动。
6. 单独立项:Kimi K3(KDA recurrent state,GDN 三个已认证后增量收益低于成本)、
   ERNIE-4.5-VL。Whisper 明确不做(cross-attention KV)。

## 六、一个结构性回答:为什么三个修复能覆盖八个模型

用户问了两次("就这三个修复就能支持这么多模型?"),值得记下答案的形状:

**因为 LMCache 对"多模态"本来几乎无感。** key 是 token id 序列的 hash,
搬 KV 按 vLLM 给的 block/layer 布局逐层拷贝 —— 它不知道某段 token 是文字
还是图片。模型特定的活全在 vLLM 侧。所以多模态给 LMCache 带来的**新问题
只有一个**:placeholder token 不携带内容身份。三个修复解决的就是这一件事,
而它**天然与模型无关**(任何模型的图片都从
`apply_mm_hashes_to_token_ids` 这一个口经过)。

比例:**~190 行生产代码 / ~6000 行套件**。

模型之间真正的差异不是"多模态怎么做",是 **KV 布局** —— 那些不需要改
多模态代码,需要的是正确配置(`ModelSpec` 十八个字段)。而**最有说服力的
是反例:Gemma 4 没被覆盖住**,它挂的地方与 mm_hash 无关。

## 七、shells

我这个 job 留了 **8 个僵尸等待循环**(`until grep ...; sleep`,15–17 小时前),
全部是 Bash 工具 2 分钟超时截断后父调用返回、shell 还在轮询早已跑完的日志。
已清空。GPU 零占用(GPU 4/6/7 上的是别人的 nightly)。

## 待办(顺序即优先级)

1. ~~`KEY_NOT_READABLE` 转 miss;复核"连接器声称装载了多少 token"这个契约。~~
   **已完成,见 `2_`**:契约的破口在 MP 工作端适配器 `get_finished`,
   失败 retrieve 的 block id 被丢掉了。`e18e55f2`。
2. ~~强制 chunk 32 回归一个已认证模型,验证 bug 与模型无关。~~
   **实验作废**:已认证的全注意力模型本来就在 chunk 16(比 32 更小)且全绿,
   这个对照跑不出信息。替代对照是 Gemma 3(chunk 16 + 多组滑窗),见 `2_`。
3. 套件加"高对象数持续并发"场景。
4. 之后按第五节的模型顺序。
5. 确认 fork 上三条分支无误后删 `backup/pre-artifact-rewrite`。
6. 存量:`../21/8_` 两条 MP 家族课题、`achievable_hit_tokens` 分母、
   P5 bypass guardrail、MP race flake、`extra_keys` 重构。
