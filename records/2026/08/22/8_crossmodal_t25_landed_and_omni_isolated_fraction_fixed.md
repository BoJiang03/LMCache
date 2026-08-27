# 跨模态 T2.5 落地 + qwen3-omni-30b 的 isolated 比例修正

接着 record `7_` 第八节的顺序推进:`6_` 第六节第 5 步——跨模态 T2.x。
提交为 **`ae5c7c73`**,工作区干净,本地领先 `fork/multi_modal` **8 个提交**,
**未推送**(等明确指令)。

---

## 一、这一轮到底想测什么(为什么不是"音频再来一遍,顺手挂张图")

之前 image 和 audio 是各自独立验过的。放在同一个 prompt 里之后,有两类失效
是任何单模态测试都看不见的:

1. **一个模态的 identity 丢了,但被另一个模态盖住。** 套件里其他所有音频
   断言,音频的差异总是和文本 salt 或图像的差异同时存在——所以"隔离成功"
   这件事,永远分不清是谁做到的。
2. **item 被当成集合而不是序列来 key。** 内容完全一样、只有顺序不同的两个
   prompt,如果键是 item 的 multiset,就会在这里碰撞,而且**只在这里**碰撞。

所以 T2.5 是一个 prompt(image 0 红 + clip 0 tone)配两个对照:

| case | 内容 | 与 IA 的差别 | 谁在承担隔离 |
|---|---|---|---|
| `t25-IA` | image 0 + clip 0 | — | 基准 |
| `t25-IB` | image 0 + clip **2** | 图像**不变**,只换音频 | **只有 audio hash** |
| `t25-AI` | clip 0 + image 0 | 内容不变,只换**顺序** | **只有位置** |

三个 case 共用一个 salt,所以系统前缀完全一致,命中数的每一点差异都来自媒体。

断言(全部通过):

- `IB` 必须命中过图像跨度(`>= image_span_margin + chunk`)——正向的一半:
  共享的图像确实被复用了,不是停在文本前缀;
- `IB` 必须止步于音频(`<= IA_full - image_span_margin`)——反向的一半;
- `AI` 必须比 `IB` **更早**断开(`<= IB - image_span_margin`),因为它在第一个
  媒体项就分叉了。

---

## 二、组合探针先量了才用(per-model 探针规则的直接应用)

`probe-oracle-is-per-model` 那条记忆的原文就是:探针属于 (model, stimulus) 这个
对,不能从别的模型搬过来。这次的探针是**新的**(两个词的组合答案),所以在写
测试之前先在认证目标上量,而且是**通过 catalog 自己的 builder** 量的
(`crossmodal_probe.py`,不是手写 prompt)。

5 个 (image, audio) 组合 × 2 个顺序 = 10 条:

- **correct 10/10**、**stable 10/10**;
- 每个顺序内部 **5/5 互不相同**;
- 两个顺序的答案**逐字节相同**(`red, tone` 正反都一样,5/5 组合都如此)。

最后这条不是好消息也不是坏消息,是一条**必须写进测试的边界**:顺序交换那一半,
语义探针**看不见**碰撞,只有命中计数器能看见。换 clip 那一半正好相反——假命中
会让 IB 用 IA 的声音种类回答,探针和计数器都能抓到。这条限制写在测试 docstring
和 README 里,不然后人会以为两半的强度一样。

顺带一个原本担心的点被否掉了:把 clip 放在图像**前面**,模型仍然正确地把
"先说颜色"绑到图像上。所以顺序 case 不需要换问法。

---

## 三、`media_order`:一个小字段,和一个不肯静默失败的守卫

`MMRequest.media_order` 决定媒体项的出现顺序,默认空 = 历史顺序
`("image", "video", "audio")`。两个守卫:

- 顺序里出现未知模态 → `ValueError`;
- 顺序**漏掉**了这个 request 确实挂了 item 的模态 → `ValueError`。

第二个才是重点。如果只按 `media_order` 拼接而不检查,写错一个字的顺序会**静默
丢掉一个媒体项**,prompt 依然合法、测试依然会跑、答案可能还对——这正是
`assertion-satisfiability-check` 那条记忆里说的"不会报错的维度错误"。

重构的惰性也证明了,不是声称的:导出 `HEAD` 版本的 catalog.py,把两个版本的
**66 个既有 case** 的 `messages()` 逐一 JSON 比对,**0 处差异**。

---

## 四、`requires_modality` 只读最近一个 marker

`pytest_collection_modifyitems` 用的是 `get_closest_marker("requires_modality")`。
跨模态测试需要**同时**满足两个模态,而 `get_closest_marker` 只会返回一个——
另一个会被静默忽略,测试就会跑在缺模态的模型上。改成 `iter_markers` 收集全部,
用 `modalities <= spec.modalities` 判断。这是既有代码的一个真实缺陷,只是在
只有单模态 gate 的时候没有暴露面。

---

## 五、这一轮真正的意外:isolated 场景在这个模型上从来没跑过

之前 record `7_` 里那句"非 hybrid,四个 isolated 场景都适用"——**只是从 config
推出来的,没有验过**。因为那次只跑了 `test_mm_acceptance.py`。这次跑整个目录,
`test_isolated_paths.py` 4 个场景里 **3 个直接崩在引擎启动**:

```
ValueError: No available memory for the cache blocks.
  at vllm/v1/core/kv_cache_utils.py:720 _check_enough_kv_cache_memory
```

原因是算术:`ISOLATED_GPU_UTILIZATION` 默认 0.35,H200 140 GiB 的 0.35 是
**49 GiB**,而这个模型权重 **59.4 GiB**。连一个 KV block 都放不下。修法就是
spec 上的 `isolated_gpu_utilization=0.75`(和两个 27B 一样),四个场景
**4 passed in 752s**。

**为什么这个坑一直藏着,比坑本身值得记:** preemption 场景在 0.35 下是**通过**
的。因为它用 `num_gpu_blocks_override=128` 直接指定块数,**绕过了**那个内存检查。
也就是说,唯一一个不依赖 vLLM 自己 profile KV 内存的场景,恰好是唯一一个会
"假装没事"的场景。一个过小的比例就是这样把自己藏起来的。

顺手把所有 spec 都查了一遍:除了两个 27B(本来就是 0.75),其余全是 ≤4B 的模型,
0.35 够用。**没有第二个 spec 有同样的问题。**

---

## 六、结果

| 项 | 结果 |
|---|---|
| acceptance 套件(含 T2.5) | **27 passed**(此前 26 + 新增 1) |
| isolated 场景 | **4 passed**(修正比例后) |
| 组合探针 | 10/10 correct、10/10 stable、每序 5/5 distinct |
| 既有 case prompt 回归 | 66 个 case,0 处字节差异 |
| ruff check / format | 全绿 |
| 证书 | **SUPPORTED** — `certificate_qwen3-omni-30b.json` |

证书内容:整目录 **31 passed / 6 deselected / 0 skipped**(953 s,`pressure_n=64`);
parity 复用已记录的**全量 MMAU** 报告(1000 题,`gate.pass = True`,flip 0、
score delta 0.00,`source` 字段明确写了 `recorded:...`)。`load_parity_report` 会
校验模型 id 和 deployment path 都对得上,所以不是拿别的跑分冒充。
`known_not_covered` 为空(非 hybrid,四个场景全跑)。

**第一份证书作废重跑了。** 第一次跑的时候我在它运行期间提交了代码,于是
`certify.py` 在启动时抓到的 `commit` 是上一个提交 `8519c60c`,而它实际测的树是
`ae5c7c73`。测的内容没错(启动之后我没再改任何文件),但证书里那一行是不成立的
——这正是 `7_` 里说 Gemma 4 那份证书"有两句不成立"的同一类问题,不能自己犯。
所以在提交完成之后原样重跑一遍,让 `commit` 字段和被测的树一致。

---

## 七、状态与下一步

- 未推送,8 个本地提交(`ae5c7c73` 为最新);`records/` 仍被 exclude。
- 产物在 `records/2026/08/22/crossmodal/`:探针脚本 + JSON、整目录 pytest 日志、
  修正前的 isolated 崩溃日志(留着,因为它是"过小比例如何藏起来"的证据)。
- `qwen3-omni-30b` 已出证书(**SUPPORTED**),音频这条线到此闭环。
- 按 `7_` 第八节的遗留顺序:五个 hybrid 证书重开(Gemma 4 那份 JSON 有两句
  不成立)、`block_pool.cache_full_blocks` 崩溃上报、`4_` 第二节那个读锁续期的
  设计问题、in-process `pass2_hit_coverage: 0.0` 的误导性显示。
- 模型顺序:hitchhiker 批(DeepSeek-OCR / Mistral Small 3.1 / MiniCPM-V 4.6 /
  Molmo 2)→ Llama 4 / Kimi-VL / Step-3 → Phi-4-multimodal(实质是 `extra_keys`
  重构)。Whisper 明确排除。

---

## 八、方法论(这一轮新增的三条)

第 1、3 条已进长期记忆(前者补进
`verify-through-pytest-not-a-hand-runner`,后者新建
`override-hides-misconfiguration.md`),因为它们跨任务复用。


1. **只跑一个测试文件,就只证明了那个文件。** "四个场景都适用"是从 config 推的,
   跑了整个目录才发现三个根本起不来。推论和运行之间的差距,只有运行能填。
2. **不要在证书跑的过程中提交。** `certify.py` 在启动那一刻抓 `commit`,之后
   工作区变成什么样它都不知道。这次结果没受影响,但证书里那一行会说谎。
3. **绕过检查的那条路径,是唯一会假装没事的路径。** preemption 用
   `num_gpu_blocks_override` 跳过了 KV 内存检查,于是它在错误配置下依然绿。
   下次给场景加"跳过某个校验"的开关时,要意识到那个场景同时也失去了报告这类
   配置错误的能力。
