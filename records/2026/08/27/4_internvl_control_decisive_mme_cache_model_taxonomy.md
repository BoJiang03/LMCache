# InternVL 对照实验定性、MME 图片缓存落地、模型分类学

日期:2026-08-27
分支:`multi_modal` @ `756684b6`
前置记录:`3_internvl_regression_suite_forgives_divergence_qwen35_livelock.md`

## 一、InternVL 的退化不是 vLLM 数值问题(决定性对照)

记录 3 留下的关键问题:InternVL3.5-2B 在 0.27.1 上 pass2 单向退化
(p = 5.3e-05),这是 vLLM 的 batch-shape 数值噪声,还是 LMCache 读路径的
真缺陷?两条路的处置完全相反 —— 前者要重设计门,后者是发布阻塞。

对照实验:**纯 vLLM,不挂 LMCache**,同一模型同一份 MME,只改
`max_num_seqs` 256 → 32(8 倍的 batch shape 变化,远大于三遍之间的差异)。

| | 纯 vLLM 256→32 | LMCache pass1→pass2 |
|---|---|---|
| 逐字节相同的答案 | 2230 / 2374 | — |
| 答案翻转 | **2** | **44** |
| 回归 / 改善 | 1 / 1(p = 0.75) | 35 / 9(p = 5.3e-05) |
| 解析翻转 | 1 | 86 |
| parse ratio | 0.9368 不变 | 0.9368 → 0.9065 |

pass1 vs baseline 是 0 翻转、逐字节相同 —— **miss 路径完好,只有 hit
路径退化**。8 倍 batch shape 只换来 2 个翻转,44 个单向翻转没法用引擎
噪声解释。结论:**LMCache 命中路径的真缺陷**,NOT_SUPPORTED 成立,不能靠
放宽翻转预算解决。

一个附带的可疑数:pass2 的 `external_cached_tokens` = 2630288,比理论上限
`achievable_hit_tokens` = 2628080 多 2208(coverage 1.0008)。两个数来自
不同计数点(前者是 vLLM 的 `PrefillStats`,后者由 `lookup_request_tokens`
按 `g*((t-1)//g)` 算),可能只是记账漂移,但方向是"命中了不该命中的
token",顺着查的时候要看。

对照脚本会顺带证明自己没跑偏:`ctl_..._256.json` 的 parse_ratio 0.9368
与 parity 报告的 `baseline_answer_parse_ratio` / `pass1_answer_parse_ratio`
完全相同。

## 二、MME 图片编码缓存(827 秒 → 38 秒)

### 先量再修

三块 GPU 同时 0% 的现象查下来是数据集加载。逐段计时:

| 段 | 耗时 |
|---|---|
| `import datasets` | 1.0s |
| `load_dataset("lmms-lab/MME")` | 3.7s |
| 非图片列全量 2374 行 | 0.1s |
| **1187 张图 PNG 重编码** | **空载 4.4 分钟,三任务并发时 12-13 分钟** |

PNG 编码占 95%。而且**每次 parity 付两遍** —— items 是 1.3 GB 的 base64,
不跨进程边界,父进程和 `--role baseline` 子进程各编一次。molmo2-4b 的
baseline 子进程实测总共 1086 秒,其中真正生成只有 60 秒。

### 修法

只缓存图片的 PNG 编码,按 question_id 的 sha256 落盘(MME 的 id 带路径
分隔符,`count/000000450303.jpg`,不能直接当文件名);题目、答案、类别每次
仍从数据集读。顺带把"每行都物化会解码图片"改掉 —— 元数据那遍
`remove_columns("image")`,图片按下标只取每个 qid 的第一行。

写入用私有名 + `replace()` 原子改名(并发的 parity 共用同一目录,读者只会
看到完整文件或没有文件);任何 `OSError` 一律吞掉(只读 home、磁盘满不能
让一个只为提速存在的缓存搞挂正跑的认证)。

### 验证

改的是**被认证代码路径上的取数逻辑**,缓存出来的 items 和现加载的有一丁点
差别,所有 parity 结果就不可比。所以先验一次,全量 2374 题按三种方式各加载
一遍,对每个 item 的每个字段做 sha256:

```
committed      827.3s  2374 items 1177 images  c5d7602e7f723fa8...
cache-cold     761.5s  2374 items 1177 images  c5d7602e7f723fa8...
cache-warm      38.2s  2374 items 1177 images  c5d7602e7f723fa8...
cache files: 1187 png, 0 stray tmp
VERIFIED identical
```

**827 → 38 秒,21 倍。** 缓存约 3 GB(早先按每 200 行抽 12 个样估的
1.3 GB 偏低一倍多,实际每张 2.6 MB),落在 `~/.cache/lmcache_mm_e2e/mme-2374`,
`LMCACHE_MM_E2E_IMAGE_CACHE` 可改根目录,删目录即强制重编码。

1187 个唯一 qid 只产出 1177 个唯一 URI —— 有 10 对 qid 的 PNG 逐字节相同。
无害,记一笔。

### 提交纪律

提交脚本先查 `pgrep "bin/python.*certify\.py"`,再查树是否已脏,然后
应用+提交在一条命令里走完,把脏树窗口压到一秒内 —— certify 记录
`commit_at_start`/`commit_at_finish` 和两端的 dirty 标志,提交期间跑的
certify 会被盖 `stable: False`(记录 3 里 qwen3-vl-2b 和 internvl3.5-2b
就是这么被盖的)。提交前后都确认 certify 计数为 0。落地
`e4937183` + `756684b6`,且 `diff` 确认仓库文件与被验证的副本逐字节相同。

## 三、模型分类学:每类要有一个活代表

用户提出的框架:模型必然可分类,每类过一个代表就可以认为整类大概率支持。
分类按 spec 里的**实测字段**做(不从架构名推)。

| 轴 | 类 | 成员 | 最新 0.27.1/sch8 |
|---|---|---|---|
| KV 结构 | 全注意力 GQA 单组 | qwen2-vl-2b, qwen2.5-vl-3b, qwen3-vl-2b, glm-4.6v, mistral, molmo2, omni, internvl3.5 | ✅ qwen2-vl-2b |
| | MHA + MoE | deepseek-ocr(唯一) | ❌ PROVISIONAL |
| | recurrent-state 混合(2 组/粗块) | qwen3.5-2b, 3.6-27b, 3.8-27b | ❌ 只有 0.23 |
| | 滑窗混合(2 组/细块 16-32) | gemma-3-4b, gemma-4-e4b | ✅ gemma-4-e4b |
| | **MLA** | — | ❌ 无成员 |
| MM 注入 | 普通 placeholder span | 多数 | ✅ |
| | DeepStack 侧缓冲 | qwen3-vl-2b(唯一) | ⚠️ 见下 |
| | 图像 span 双向注意力 | gemma-3-4b, molmo2-4b | ❌ |
| | 非前缀媒体布局 | molmo2-4b(唯一) | ❌ |
| | **模态 LoRA 进 key** | — | ❌ 无成员(Phi-4) |
| 视觉 token | 动态分辨率单块 | qwen 系, glm, mistral | ✅ |
| | **瓦片切分** | internvl3.5-2b, deepseek-ocr | ❌ **零通过** |
| | 固定 soft token | gemma-4-e4b | ✅ |
| 模态 | video | qwen 系, gemma-4, glm, internvl | ✅ |
| | audio | qwen3-omni-30b(唯一) | ❌ |

### 三个结论

1. **"一个代表过了整类就算过"已有实测反例。** internvl3.5-2b 和
   qwen2-vl-2b 同属"全注意力 GQA 单组",代表过了,它 NOT_SUPPORTED。
2. **但反例可能说明分类少了一根轴 —— 瓦片。** 瓦片类两个成员
   (internvl3.5-2b NOT_SUPPORTED、deepseek-ocr PROVISIONAL)零通过,而
   internvl 正是"全注意力 GQA"类里唯一的瓦片模型,也是那类里唯一的失败。
   这把 internvl 的排查从"某个模型的 bug"重定为"瓦片这一类的 bug"。
   注意:记录 2 里被证伪的是"解析翻转 ⇔ 瓦片"这条相关,方向单向性这条
   没有被证伪(2 个瓦片模型单向,4 个非瓦片双向)。
3. **qwen3-vl-2b 的 DeepStack 是假绿。** 证书 SUPPORTED,但 DeepStack
   子套件"声明但不可运行"(读回存储 KV 需要已删除的 in-process 后端),
   每张证书都写成 exclusion。这个类**没有活的检测器**。

### 一处纠正

记录 `../22/10_` 把 DeepSeek-OCR 称为"MLA + MoE",并当作套件第一个 MLA
覆盖。**错的。** spec 里 2026-08-27 在活引擎上实测:一个 KV 组
(FullAttentionSpec,block 16,12 层,80 KiB/page),`use_mla` 在
ModelConfig 和 KV spec 上都是 False,`kv_lora_rank`/`qk_nope_head_dim`/
`v_head_dim` 全未设。它带来的是第一个 **MoE** 和第一个 **MHA**(10 Q 头 /
10 KV 头,其余全 GQA)。**MLA 仍然零覆盖**,Kimi-VL 是唯一候选,价值没有
因 DeepSeek-OCR 降低。

## 四、"每类一个活代表"的最小集合

| 类 | 只能由谁覆盖 | 现状 |
|---|---|---|
| recurrent-state | qwen3.5-2b / 3.6-27b / 3.8-27b 任一 | 3 个都在队列 |
| MHA+MoE **和** 瓦片 | deepseek-ocr | PROVISIONAL,控制实验跑着 |
| 双向注意力 **和** 非前缀媒体 | molmo2-4b(后者唯一成员) | parity 已绿,certify 排队 |
| audio | qwen3-omni-30b(唯一) | 刚排上 |
| DeepStack | qwen3-vl-2b(唯一) | 已 SUPPORTED,要重发(`stable:False`) |

最小集合 4 个模型 + 1 次证书重发。已排队的 glm-4.6v-flash、gemma-3-4b、
qwen3.6-27b、qwen3.8-27b 严格说冗余,但机时已经占上了,让它们跑完。

两个空类不是"跑一遍"能解决的:MLA(要下 Kimi-VL 权重)、模态 LoRA
(Phi-4,实质是 `extra_keys` 重构)。

## 五、本轮的实验结果

- **molmo2-4b parity 通过**:7 翻转(-2/+5,p = 0.9375),pass1 vs baseline
  0 翻转,hit_ratio 0.992 => PASS。但当时的 chain3 队列只排了
  `cert:qwen3.6-27b`,没排 molmo 的 certify,报告在盘上没人用 —— 加了
  `certonly` 步骤直接复用报告认证,不重跑 parity。
- **DeepSeek-OCR 控制实验**第一遍 seqs=256 完成,第二遍跑着。

## 六、两个操作教训

1. **正在运行的 bash 脚本不能原地改。** bash 按字节偏移增量读脚本:
   `for` 循环整体已解析进内存,循环结束后才回到记录的偏移继续读,
   插入内容会让那个偏移落到错位的位置,可能执行出一条半截的命令 ——
   在这里就是可能拿错参数跑一次 `certify_short.sh`。已按原字节还原
   (md5 复核),新步骤放进 `chain4.sh`。
2. **`kill` 那个 pid 不一定是脚本。** `nohup ./chain4.sh ... &` 拿到的
   `$!` 是包装 shell,杀掉它以后真正的 `bash ./chain4.sh` 还在,于是同一
   GPU 上排了两条队。要按 `ps -eo args | grep chain4\.sh` 复核。

## 七、下一步(用户定的四阶段)

1. 让每个类至少有一个模型过最新 support 测试。
2. 再测主线上没试过的四个:Llama 4 / Kimi-VL / Step-3 / Phi-4-multimodal。
3. 重构代码,更结构化,面向以后支持新的多模态模型。
4. 最后全部再跑一遍,确保全过。

当前队列:

| GPU | 队列 |
|---|---|
| 2 | deepseek 控制实验(seqs=32)→ cert:gemma-3-4b → certonly:qwen3-vl-2b |
| 3 | cert:qwen3.6-27b → certonly:molmo2-4b → cert:qwen3-omni-30b |
| 7 | cert:qwen3.5-2b → cert:glm-4.6v-flash → cert:qwen3.8-27b |

排查线:瓦片假设(internvl3.5-2b + deepseek-ocr),以及 pass2 命中 token
超出理论上限 2208 那个记账异常。
