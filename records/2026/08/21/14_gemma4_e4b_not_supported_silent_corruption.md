# Gemma 4-E4B:NOT_SUPPORTED —— 失败的 L1 读被当成命中,静默损坏

日期:2026-08-21(19:13–23:20)
分支:`multi_modal` @ `46213857`
证书:`certificate_gemma-4-e4b.json`(verdict **NOT_SUPPORTED**)
前置:`12_vllm_upgrade_batch_dissolved.md`(为什么 Gemma 4 能在 0.23 上跑)

## 结论

- 合成套件:**26 passed**,**连过 5 次**(12–16 分钟一轮)
- MME 全量 parity(2374 题,MP 路径):**FAIL**,而且不是"差一点":

| 项 | 实测 | 预算 |
|---|---|---|
| flips 命中 vs 未命中 | **1288 / 2374 = 54.3%** | 11.87 |
| flips 未命中 vs 纯 vLLM | **0 / 2374** | — |
| 分数 pass2 − pass1 | **−920.24**(1844 → 924) | 10.0 |
| 分数 pass1 − baseline | 0.00 | 10.0 |
| 装载覆盖率 | 1.0076 | 0.95 |
| lookup 命中率 | 0.957 | — |

**未命中 pass 与纯 vLLM 逐题相同(0 flip);一开始装载 KV,一半答案变成
乱码。** 这不是漂移,pass 2 的输出长这样:

```
pass1                 pass2(命中)
'No.'      (545)      '**The\nCorrect\nAnswer\nis'   (148)
'Yes'      (316)      '.\nsettings/modifiers/gd.'     (135)
'No'       (209)      '**The\nto\nspeak\nfrom'        (124)
'no'       (159)      'ข้อ'                            (79)
```

## 根因:`KEY_NOT_READABLE` 被吞掉

服务端日志里 **11516 条**错误,起于 23:11:42,一直到跑完:

```
LMCache ERROR: Failed to read prefetched object ObjectKey(chunk_hash=...,
  object_group_id=0) from L1 storage: The specified key exists but cannot
  be read.                                    (storage_manager.py:300)
LMCache ERROR: Some keys not found during retrieve!
                              (lmcache_driven_transfer.py:1366)
```

"exists but cannot be read" 是 `L1Error.KEY_NOT_READABLE`——**键被写锁占住**
(`storage_manager.py` 把它归进 `write_locked_keys`)。也就是说:并发的
store 持着写锁,retrieve 读不到。

**致命的一步在后面:这个失败没有变成 miss。** 那一跑仍然报
`hit_coverage=1.0076`、`external_cached` 把这些 token 算成"已装载",于是
vLLM 跳过了这些位置的重算,模型拿着**从未被写入的 GPU KV** 往下算——
输出成了乱码。失败的读被当成成功的装载,是**静默数据损坏**。

## 损坏是"到点就翻",不是随机竞态

按题号顺序统计 pass2-vs-pass1 的 flip 率:

```
    0- 1056:   0.0%     <- 干净
 1057:          第一个 flip
  948- 1185:  52.7%
 1185- 2374:  96–99%    <- 之后几乎全坏
```

前 1056 题一个都不错,之后再也没恢复。时间上与 23:11:42 那批读失败对齐。

规模是触发条件,不是模型:

| parity 规模 | 命中率 | flips | 分数漂移 |
|---|---|---|---|
| 400 题 | 0.951(真命中) | **0 / 400** | 0.00 |
| 2374 题 | 0.957 | 1288 / 2374 | −920 |

400 题命中充分且**完全正确**。所以不是 Gemma 4 的 KV 布局算错了,是
**规模压力下的读写锁竞争**。

排除项:没有驱逐(日志里只有 EvictionController 启动行);L1 池 280 GB
在 23:00:50 就已经全量物化(100%),不是容量耗尽;心跳已静音,不是降级
模式。

## 为什么 Gemma 4 先撞上

chunk 32 是它的硬约束(分页组 block 16/32 的公倍数,见 `12_`)。同样一条
800 token 的 prompt:

- GDN 混合(chunk 784):**1 个 chunk**
- Gemma 4(chunk 32):**~25 个 chunk**

对象数、lookup 数、store/retrieve 交叠面都是 25 倍,所以它在一个 benchmark
之内就把这个竞争跑出来了。**已认证的 7 个模型 chunk 都在 544–784,不代表
它们没有这个 bug,只代表它们到不了那个压力。** 这条要写进后续排查:
用小 chunk 重跑一个已认证模型,就能验证它是否与模型无关。

## 套件的盲区(比模型结论更重要)

合成套件**连过 5 次 26/26**,而同一个引擎在 MME 上一半答案是乱码。套件
够不到这个 bug,因为它跑的是少量、基本顺序的请求;这个 bug 要的是
"数千请求持续并发 + 数万小对象"。

- 这正是 parity 存在的理由:**绿的合成套件不是支持的证据**,证书要求
  两层同时绿。
- 反过来看也说明 `T0.7` 之类的存量审计对"并发规模"这一维覆盖不足。
  值得加一个"高对象数持续并发"的场景(小 chunk + 数百请求),让这类
  竞争在套件里就暴露。

## 一个说明:心跳静音不是原因

前三次 parity 都死在心跳误判(见 `harness.py` 常量注释与 commit
`468a906a`),都死在 pass 2 之前。把心跳静音**没有制造**这次损坏,只是
让跑能进行到能观察损坏的地方。400 题那次同样是静音心跳,结果 0 flip。

## 待办

1. **上游 bug**:`KEY_NOT_READABLE` 必须变成 miss(少报已装载 token 数),
   而不是被吞。这是静默损坏,优先级高于本项目的任何模型认证。
   顺带:`Some keys not found during retrieve!` 这条 ERROR 也没有让请求
   失败,连接器与 vLLM 之间的"实际装载了多少"契约需要复核。
2. 用 chunk 32 重跑一个已认证模型(例如 qwen2.5-vl-3b 强行小 chunk),
   验证与模型无关。
3. 套件加"高对象数持续并发"场景。
4. Gemma 4-12B 同架构,大概率同症;E4B 还带音频塔(当前 scope 之外)。
5. `achievable_hit_tokens` 分母偏小(这次 1.0076,与 `10_`/`13_` 同源)
   —— 注意它现在还掩盖了"装载失败仍计入 external_cached"这件事,
   两个问题叠在同一个指标上。
