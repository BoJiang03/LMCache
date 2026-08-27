# 开始 audio 支持:模型存储外迁、探针定型、benchmark 选定

日期:2026-08-22 06:02 PDT

接 [`5_`](5_recurrent_state_eviction_opened_preemption_walled_off.md) 第九节的下一步:Qwen3-Omni。这一篇是 audio 这条线的**起点记录** —— 还没有写任何套件代码,记的是三件必须先定下来的事,以及用户中途给的两个方向修正。

`git` 工作区干净,**没有创建提交**:`5_` 的代码全在 `782d612c` 里,audio 这边目前只有 `$CLAUDE_JOB_DIR/tmp` 下的实验脚本(已复制到本目录 `audio_probes/`)。

---

## 一、用户给的两个方向修正

1. **「模型都要外放到一个 ssd 还是 disk 上,不是 home 下面」** —— 我原来打算在 `/home` 里腾地方(甚至考虑删掉别人可能在用的缓存)。方向错了,见第二节。
2. **「我觉得你得和 image 一样,搞 benchmarks」** —— 我原本的计划把合成探针放在第一位,benchmark parity 放在最后。这个顺序是错的:**合成探针只能证明"没串味",证明不了"质量没掉"。** MME 那一层必须有音频对应物,而且要和 image 一样是三道闸门(flip / score / parse-ratio)。见第四节。

---

## 二、存储:模型移到 `/raid/data/hub`(共享 HF 缓存)

原状况:`/home` 92% 满、只剩 77 GB,而 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 是 **70.5 GB**。

我先后提过两个错方案,都撤回了:

- ❌ 删掉缓存里 23 GB 的 `gemma-4-12B-it` 腾地方 —— 那是共享缓存里别人也可能用的条目。
- ❌ 让用户跑 `sudo mkdir /raid/bo` —— **根本不需要 sudo。**

正解是用户提示的:`/raid` 下**已经有**一个共享模型目录 **`/raid/data/hub`**,`drwxrwsr-x root:users` + setgid,而我在 `users` 组里 —— 直接可写,而且它本来就是 HF hub 布局(`models--*` / `datasets--*` / `CACHEDIR.TAG`),已有 2.6 TB,还剩 6.8 TB。

顺手发现真正的浪费:**我 `/home` 缓存里有 61 GB 是和共享缓存逐字节重复的**。

| 重复条目 | /home 占用 |
|---|---|
| gemma-4-12B-it | 22.8 GB |
| Qwen3-8B | 15.6 GB |
| gemma-4-E4B-it | 15.3 GB |
| Qwen2.5-VL-3B-Instruct | 7.2 GB |
| Qwen3-0.6B + gemma-3-270m | 2.0 GB |

所以"删缓存"这件事根本不需要冒风险:重复的那份删掉,原件还在 `/raid`。

迁移脚本 `audio_probes/migrate_hf_cache.sh`,一条不可让步的顺序:**先复制 → 再校验 → 才删除**,从不反过来。结果验证了这个顺序是对的 —— `models--Qwen--Qwen3-0.6B` 那条 rsync 失败了(`-a` 想保留 group,而 `/raid` 里那个目录是 root 所有,chgrp 被拒),脚本按设计**保留了源文件**而不是删掉。`/home` 从 92% → 81%,还在降。

**约定(需要记住):以后所有跑动都带 `HF_HUB_CACHE=/raid/data/hub`。** 这不写进 harness —— 那是机器路径,不该硬编码进套件;它属于环境。

Qwen3-Omni-30B-A3B 已下好,66 GB,RAID 上 2 分钟。

---

## 三、探针:唯一可用的是 `sound_kind`,而"稳定地答错"这条退路被实测否掉

image 套件的地基是「模型能可靠报出合成物的内容」("什么颜色?" → "Red"),假命中才会暴露成"说出了另一张图的颜色"。audio 需要同一个东西,而**纯合成音对模型来说是弱得多的刺激**。

第一轮(4 个 item):

| item | answer | correct | stable |
|---|---|---|---|
| 1 beep | "1" | ✅ | ✅ |
| 2 beeps | "2" | ✅ | ✅ |
| 3 beeps | "4" | ❌ | ✅ |
| 4 beeps | "3" | ❌ | ✅ |
| 220 Hz | "low" | ✅ | ✅ |
| 1760 Hz | "low" | ❌ | ✅ |

**6/6 完全确定性** —— 这是好消息,说明 baseline 逐字节比对这一层(套件的主 oracle)可用。

然后我推了一步:检测器其实**不需要答对,只需要"稳定 + 两两不同"** —— A 和 B 答案不同,假命中就会让 A 的 prompt 吐出 B 的答案。按这个标准 beep 计数(1/2/4/3 四个互异)还能用。

**第二轮把这条退路否掉了。** 扫 4 个刺激族,同时量 correct / stable / **distinct**:

| family | correct | stable | distinct | answers |
|---|---|---|---|---|
| **sound_kind**(tone/noise/silence) | ✅ | ✅ | ✅ | tone, noise, silence |
| beeps(间隔 0.6 s,n=1..5) | ❌ | ✅ | ❌ | 1, **3, 3**, **4, 4** |
| pitch_wide(110/660/3520 Hz) | ❌ | ✅ | ❌ | **low, low**, high |
| pitch_direction(升/降) | ❌ | ✅ | ❌ | **up, up** |

**关键:模型答错的方式是"塌向同一个答案",不是"随机错"。** 拉长 beep 间隔本想改善计数,结果从"四个互异"退化成两对撞车;pitch 和方向直接塌成一个词。而撞车正是让检测器**瞎掉**的那种失败 —— A 和 B 答案相同,假命中就完全不可见。

所以:`sound_kind` 是唯一三项全中的族,退路方案作废,不用再考虑。这也是一次「先测再建」省下的返工:如果按第一轮的推理去搭套件,检测器会在 pitch/direction 上静默失效。

**顺带量到 audio token 密度:约 25 token/秒**(1.6 s → 86 tokens,4.75 s → 156 tokens,扣掉 ~40 模板开销后线性)。这是后面设计 chunk 对齐、prompt padding、以及 T0.4 boundary phase 要用的数。

---

## 四、benchmark:MMAU test-mini

调研结果:

- ✅ **`TwinkStart/MMAU`**,**2.84 GB**,已下好并验过 schema:**1000 行**,audio 以 WAV bytes **直接嵌在 parquet**(不用另外抓文件),字段有 `question` + 4 个 `choices` + ground-truth `answer`,还有 `task`(sound/music/speech)、`category`、`difficulty` 可分层。
- ❌ `apple/mmau` —— 同名但是 Apple 的 agent benchmark,不是音频。
- ❌ `qyang1021/AIR-Bench-Dataset` —— 49 GB / 25780 个散文件,第一版不值得。

**为什么它是对的选择**:四选一有 ground truth,parse 和判分跟 MME 的 yes/no 同构,所以 `parity_gate` 那三道闸门(flip 预算 / score delta / parse ratio)**整套可以复用,不用新发明判分逻辑**。

代码形状上,`benchmark_parity.py` 只有三处是 MME 专属的:`load_items`、`conversations`、`parse_yes_no` + `mme_scores`。其余(`parity_gate` / `run_batch` / `engine_kwargs` / `run_baseline` / `achievable_hit_tokens`)都与 benchmark 无关。**所以这是一次小的抽象,不是 fork。**

**必须重新标定、不能继承的两个数**:
- **flip 预算**:MME 的 0.5% 是短答案模型标出来的。四选一 + audio encoder 是另一个数值 regime,得按 Qwen3.6 那套先测 baseline 自比(应为 0)、再测 no-LMCache 只换 batch shape 的对照,拿到"引擎自身抖动地板"再定预算。
- **parse ratio 下限**:要看这个模型怎么格式化选项答案才知道。

---

## 五、audio 特有、必须新测的四件事(风险从高到低)

1. **keying** —— LMCache 的多模态 identity 替换目前对 image/video 的 `mm_hash` 生效,**audio 走不走同一条路?** 如果音频 hash 没进 cache key,两段不同音频会互相命中 → 静默错答案。这是整条线里唯一可能是真 bug 的地方,必须靠 negative control(`identity_blindness`)证明探针真抓得到,而不是探针自己瞎了。
2. **audio token 密度与 chunk 对齐** —— 已量到 ~25 token/s,用来让 audio span 精确落在 chunk 边界内/外(对应 T0.4 的 boundary phases 和 truncated-span 分支)。
3. **encoder cache** —— 引擎启动报 16384 token 的 encoder budget,audio 也占它。这跟 KV cache 是两套东西,LMCache 只管后者;要确认 resume 时不会出现类似 Qwen3-VL DeepStack 的问题。
4. **跨模态混合** —— omni 能同时吃 audio+image。T2.1(顺序)/ T2.2(部分共享)可以扩成"只换 audio 不换 image""交换 audio/image 顺序"。这是纯图像模型测不到的,也是 omni 真正新增的覆盖面。

---

## 六、当前进行中 / 下一步

进行中:MMAU 40 题 smoke 跑在 **Qwen3-Omni-30B**(plain vLLM,无 LMCache),两遍相同输入,输出 parse ratio、accuracy(总体 + 分 task)、determinism、prompt token 分布 —— 这四个数是 parity 闸门标定的前置。

下一步顺序:

1. smoke 结果定 prompt 形状与 parse 函数;不合格就改 prompt 而不是放宽 parse。
2. geometry probe:Qwen3-Omni 的 KV group 数、chunk、是否 hybrid(30B MoE,可能又是一个多组模型)。
3. `benchmark_parity` 抽出 benchmark 抽象,加 MMAU 分支。
4. catalog 加 audio item + `audio_url` 消息形状,先跑最小集(fresh miss / repeat hit / cross-audio isolation)+ negative control —— 即第五节第 1 条。
5. 扩到跨模态 T2.x。
6. 三道闸门标定 + 全量 1000 题 parity。

`Qwen2.5-Omni-3B`(12 GB)保留作快速迭代载体;认证目标是 30B。

---

## 七、方法论

延续 `5_` 第七节那条链(证明代码被执行 → 按正确性质 gate → 证明你的验证跑的是真东西),这一轮加一条:**在为一个新模态搭套件之前,先证明这个模态的 oracle 存在。**

具体地说,我差点按"第一轮 + 一个看起来成立的推理"就去写代码;第二轮花一次实验,把那个推理否掉了。**「稳定但答错」和「稳定且互异」是两个不同的条件,而模型恰好在前者成立时违反后者。** 这个区别只有量了 distinct 才看得见 —— 而我第一轮根本没量它。
