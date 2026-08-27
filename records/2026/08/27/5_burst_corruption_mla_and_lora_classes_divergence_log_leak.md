# 成段坏输出的第二例、MLA/模态 LoRA 两个空类落地、证书稳定性的一个漏洞

日期:2026-08-27
分支:`multi_modal` @ `756684b6`(两笔新 spec 提交已备好,等安静窗口)
前置记录:`4_internvl_control_decisive_mme_cache_model_taxonomy.md`

## 一、缺陷的形状变了:不是翻转,是成段坏输出

记录 4 把 InternVL 的 pass2 退化定性为"命中路径的真缺陷",但仍按
"±1 bf16 量子的边界翻转"理解它。**这个理解是错的。**

把答案逐条拉出来看:

```
idx=1469  base/p1 'No.Here is a description of the image'   p2 'GreekGreekGreekGreekGreek...'
idx=1520  base/p1 'Yes, this is a picture of Must'          p2 '.nz.nz.nz.nz.nz.nz'
idx=1530  base/p1 'Yes, this is a picture of Sv'            p2 'proofazoazoazoazoazo'
```

baseline 和 pass1 逐字节相同且正常,pass2 从第一个 token 起就进重复循环。
这是**前缀 KV 被写坏**,不是边界附近的数值抖动。

### 而且 gemma-3-4b 是同一个东西

gemma-3-4b 这轮 parity 挂了,门挂在 parse ratio 上:

```
answer_flips_pass2_vs_pass1 = 2   (退 2 进 0,p = 0.25)
parse_flips_pass2_vs_pass1  = 125
parse ratio 0.9937 -> 0.9410      delta 0.0527  (门限 0.02)
```

答案几乎不翻,125 题变成解析不出来的东西。它的坏输出:
`'allure-yesandderoleas'`、`'ResearchJack le وفي et excel and the'`、
`'-servings\nspecout'` —— 同一题 pass1 是 `'Yes.\n\nBased on the image,'`。

### 关键新事实:成段连坏

把"pass1 能解析、pass2 不能"的题按运行顺序切成连续区间:

```
gemma-3-4b     125 个 / 9 段
  1752-1783  len=32  scene
  1814-1831  len=18  scene
  1976-2019  len=44  scene
  2146-2159  len=14  posters

internvl3.5-2b  79 个 / 30 段
  1696-1705 (10) / 1714-1721 (8) / 1836-1841 (6) / 2302-2306 (5) ...
```

**连续区间**,不是零散点。两个架构毫不相干的模型(gemma-3-4b 是滑窗混合
两组 + 图像 span 双向注意力;internvl3.5-2b 是全注意力单组 + 瓦片)坏法
一模一样。这把排查从"某个模型/某根轴的 bug"重定为**一次状态性事件**。

### 已经排掉的解释

- **容量驱逐**:gemma-3-4b 只存了 736336 token,上限 280 GB,远没满;
  离上限最近的是 molmo2-4b(约 315 / 340 GB),它是**过的**。

  | 模型 | cap | pass1_stored_tokens | gate |
  |---|---|---|---|
  | gemma-3-4b | 280 GB | 736,336 | False |
  | internvl3.5-2b | 280 GB | 1,632,832 | False |
  | molmo2-4b | 340 GB | 2,185,760 | True |

- **命中 token 超出理论上限**(记录 4 的次要线索):**作废**。9 份 parity
  报告 coverage 全部 > 1(1.0004 ~ 1.0076),通过的模型也一样,是两个
  计数点之间的系统性偏移,不是 internvl 的信号。

- **图像几何**:internvl 上 aspect p=0.0006 / longest edge p=0.014 显著,
  但坏区集中在 landmark / scene / posters,而这几类恰好排在运行后段,
  几何与运行位置在 MME 的固定顺序下混淆,不能分离。deepseek-ocr 的同类
  检验 p=0.26/0.42/0.95,不显著。

### 判定性实验(在跑)

internvl 原样重跑一遍(GPU6,`OUTTAG=internvl3.5-2b.run2`):坏的下标
**一样** → 确定性,可二分;**不一样** → 竞态。第一次尝试的 baseline 子
进程在 111 秒时被 SIGKILL(`-9`),没抓到 OOM 记录,已重启。

## 二、两个空类的候选都跑通了

### Kimi-VL A3B —— 套件第一个 MLA

活引擎实测(0.27.1):

```
use_mla: true
KV group: MLAAttentionSpec, block 16, 27 层, 18432 B/page -> 30.4 KB/token
prompt tokens: 442 (640x480) / 1052 (1540x1540)
全 MME 的 KV: 33 ~ 78 GB
```

语言塔是 DeepSeek-V2 形状(`kv_lora_rank 512`,`qk_nope_head_dim 128`,
`v_head_dim 128`),这是 KV 只有 GQA-8 模型三分之一宽的原因。vLLM 只给它
注册了 image,没有 video 路径,证书不能声称 video。config / preprocessor /
tokenizer 三处都有 `auto_map`,需要 `trust_remote_code`。16B bf16 约 33 GB
权重,超过隔离场景 0.35 的默认额度,`isolated_gpu_utilization=0.75`。

### Phi-4-multimodal —— 模态 LoRA,外加一个新 KV 类

```
KV group: SlidingWindowSpec 覆盖全部 32 层 -> 128 KB/token(全套最宽)
hybrid_family 仍是 NONE:gemma-3/4 是"全注意力 + 滑窗"两组,
                        它是单组、且这一组是窗口的
```

这个结构现有 14 个模型都没有。图像规模用处理器自己的裁块预算收:

| dynamic_hd | 640x480 | 1540x1540 | 全 MME 的 KV |
|---|---|---|---|
| 36(默认) | 1072 | 4440 | 超上下文 |
| 4 | 1072 | 1336 | ~342 GB |
| 2 | 484 | 552 | ~200 GB |

取 2:128 KB/token 下 342 GB 主机缓存在这台共享机上要不起,而 484 token
正落在其它 spec 给照片定的那一档(Mistral 446,Molmo 2 770)。

不需要自带模板:仓库自己的模板在 **string content** 下渲染出
`<|system|>...<|end|><|user|><|image_1|>...<|end|><|assistant|>`,这正是
vLLM chat 路径传的东西;它只在 **list content** 下抛 TypeError,所以手写
`[{"type":"image"}, ...]` 的探针会看到一个 vLLM 永远碰不到的失败。

### 模态 LoRA 的洞是真的

vLLM 的 `phi4mm.py` 用 `AutoWeightsLoader(self, skip_substrs=["lora"])`
跳过仓库里的 `vision-lora/` 和 `speech-lora/`,适配器要靠 `--enable-lora`
逐请求挂。而 LMCache 这边:`token_database.py` 的 `extra_keys` 通道存在
但**从来没有调用方传过值**(`_hash_tokens` 的三个调用点都不传),
`request_configs` 只收用户自己塞进 `kv_transfer_params` 的 `lmcache.*` 键。
**LoRA 身份不进 cache key。** 同一段文本前缀在 vision-LoRA 和 speech-LoRA
下 KV 不同、key 相同,会互相串。这是 `extra_keys` 重构要解决的问题。

## 三、一个把所有证书都会毁掉的漏洞

`c64eae85` 新增的 divergence 日志会在 `tests/e2e_mm/` 落
`divergences_<key>.jsonl`,而 `.git/info/exclude` 只挡了
`certificate_*.json` / `parity_*.json` / `suite_*.xml`。于是**每次 certify
一跑就把树弄脏**,`dirty_at_finish=true` → 从此每一张证书都会被盖
`stable: false`。deepseek-ocr 第一张证书就是这么废掉的。

已补进 exclude,重发后 `stable=True`。教训:新增落盘产物时,exclude 的
模式要跟着加 —— 证书的可信度依赖树的洁净,而这个依赖是隐式的。

## 四、当前证书表

| 模型 | 判定 | schema | stable |
|---|---|---|---|
| deepseek-ocr | SUPPORTED | 8 | True |
| gemma-4-e4b | SUPPORTED | 8 | True |
| mistral-small-3.1-24b | SUPPORTED | 8 | True |
| qwen2-vl-2b | SUPPORTED | 8 | True |
| qwen2.5-vl-3b | SUPPORTED | 8 | True |
| qwen3-vl-2b | SUPPORTED | 8 | **False**(重发中) |
| gemma-3-4b | **NOT_SUPPORTED** | 8 | False |
| internvl3.5-2b | **NOT_SUPPORTED** | 8 | False |
| glm-4.6v-flash / qwen3.5-2b / qwen3.6-27b / qwen3.8-27b | SUPPORTED | 2-3(旧) | — |

qwen3.6-27b 的新 parity **过了**(2 翻转,p=0.75),certify 在跑,
recurrent-state 那一类有着落了。但要留一笔:它的
`cache_granularity_tokens` 是 784,全 MME 只有约 220 题长到能命中,
`pass2_lookup_hit_ratio` 只有 0.077,门用的是 coverage(1.059)。
**这一类的命中路径被检验的量很薄。**

qwen3.5-2b 又死锁了(看门狗 1212 秒无输出杀掉),是记录 3 那个老问题。

## 五、MME 图片缓存的实测收益

gemma-3-4b 是缓存落地后第一个从头跑的模型:

```
[parity] 2374 MME questions loaded in 12.5s
[parity:baseline] 2374 items loaded in 12.7s
```

对比并发时的 12-13 分钟一遍、每次 parity 两遍。缓存按数据集
(`mme-2374`)而不是按模型建,所有 MME 模型共享。**唯一的例外是
qwen3-omni-30b**,它是全表唯一 `parity_benchmark="mmau"` 的模型,那份
数据集的加载没进缓存。

## 六、两个操作教训

1. **看门狗会连自己一起杀。** `chain3.sh` 的 `watch_stall` 用
   `kill -TERM -PGID` 杀停滞的 parity,而 chain 脚本自己就在那个进程组里,
   于是 GPU7 的整条队列(glm-4.6v-flash、qwen3.8-27b)跟着没了。要杀就杀
   子进程自己的组,或者用 `setsid` 把 parity 放进独立组。
2. **没落地的 spec 也能先跑 parity。** Kimi-VL 和 Phi-4 的 parity 用
   `launch_raw.sh` 直接按 `certify.parity_command()` 将来会生成的同一份
   argv 跑,报告落盘后可以被 `certify --parity-report` 复用 ——
   `load_parity_report` 只校验 model id、deployment path 和方向计数器,
   不校验 commit。这样"等提交窗口"就不用占着机时空等。

## 七、当前状态

| GPU | 在跑 |
|---|---|
| 2 | qwen3-vl-2b certify(重发) |
| 3 | qwen3.6-27b certify |
| 5 | phi4-mm MME parity(首次) |
| 6 | internvl3.5-2b 重跑(判定确定性 vs 竞态) |
| 7 | kimi-vl-a3b MME parity(首次) |

两笔 spec 提交(`register Kimi-VL A3B` / `register Phi-4-multimodal`)已经
写好并验过(能 import、parity argv 正确、ruff check + format 干净),
挂在一个后台等待器上,等系统里 certify 计数归零的那一刻提交。等提交落地
后要排回去的:`certonly:molmo2-4b`、`cert:qwen3-omni-30b`,以及
`certonly:kimi-vl-a3b`、`certonly:phi4-mm`。
