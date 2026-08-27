# "vLLM 升级批"整批解散:两个模型都不需要升级

日期:2026-08-21(下午)
分支:`multi_modal` @ `da537bf0` + 新增 qwen3.8-27b spec(未提交时点)
推翻的记录:`2_model_priority_revision_aug2026.md` 第 4 项、
`5_gemma4_blocked_on_vllm023.md` 的"推迟到升级批次"结论

## 结论先写

按 `2_` 的顺序,剩下的是"**vLLM 升级批**":Qwen3.8-27B + Gemma 4,
前置是升级共享 venv `/home/bo/venvs/vllm-lazy`。我一直停在这里等确认,
因为别的工作依赖那个 venv。

**这个前置根本不存在。** 两个模型都能在 0.23.0 上跑:

| 模型 | `2_` / `5_` 的判断 | 实测 |
|---|---|---|
| Qwen3.8-27B | 0.23 撑不到,需升级 | **架构与 3.6-27B 完全同构**,0.23 直接跑,套件 26 passed |
| Gemma 4 E4B | 权重装载崩,实现落后于 checkpoint | **0.23 上加两个 config 属性即可**,文本+图像探针全对 |
| Gemma 4 12B | 同上 | 同上(1 个颜色探针是感知怪癖,与版本无关) |

共享 venv 一行没动,`vllm-ci-compat` 也一行没动(只读用来做对照)。

## 一、Qwen3.8-27B:研究结论 vs config 实测

`2_` 写"0.23.0 撑不到 Qwen3.8",依据是 web 检索的架构描述(还说它
"保留 DeepStack")。拉下 `Qwen/Qwen3.8-27B/config.json` 与已认证的
3.6-27B **逐字段 diff,只有一个字段不同**:

```
key                    3.6-27B    3.8-27B
transformers_version   4.57.1     5.8.0.dev0
```

`architectures` 都是 `Qwen3_5ForConditionalGeneration`,64 层
`layer_types` 完全一致(16 full + 48 GDN),hidden 5120 / head_dim 256 /
4 KV heads 一致,`deepstack_visual_indexes` 都是空(**没有** DeepStack,
`2_` 这一条也错)。也就是说 3.8-27B 是**重训的 3.6-27B**,不是新架构。

引擎实测坐实:3× MambaSpec(各 16 层)+ 1× FullAttentionSpec(16 层),
page size 全部 3211264 —— 和 3.6-27B 一模一样。12/12 颜色探针全对
(`enable_thinking: False`)。

所以 spec 直接继承 3.6-27B 的每一个数,注释里写清"这些是**继承**而非
重新测量,理由是两个 config 同构",并把 flip 预算标成**预测**:
同样 48 层 GDN 应当落在 ~0.5%,量出来差很远就是否证了 `10_` 的深度论。

教训(第三次同型):**"某版本支持不了"这种判断,必须落到 config/实测,
不能停在版本号和发布说明上。** `2_` 的三条 Qwen3.8 陈述里两条是错的。

## 二、Gemma 4:根因是"transformers 把属性藏了,vLLM 的 getattr 默认值把它吞了"

`5_` 记的两段式失败在 0.23 上复现无误。这次先做的事是**换版本对照**:
在 `vllm-ci-compat`(vLLM 0.27.1)上跑同一个 E4B ——
**报完全相同的 `stage2_weight_shape` 断言**。升级不解决问题,升级批次的
另一半理由也没了。

于是回头查真根因。checkpoint 逐 tensor 扫描(E4B,42 层):

```
q_norm/k_norm  35 个 (256,)   7 个 (512,)
q_proj         35 个 (2048,2560)  7 个 (4096,2560)
```

那 7 层正是 `layer_types` 里的 `full_attention`(下标 5/11/17/23/29/35/41)。
config.json 里写得明明白白:

```
text_config.head_dim:        256   # sliding 层
text_config.global_head_dim: 512   # full attention 层
```

而 vLLM(0.23 和 0.27 **都**)是这样建层的:

```python
head_dim = getattr(config, "global_head_dim", config.head_dim)
```

关键实测:transformers 5.15 把 Gemma 4 的逐层维度收进了
`per_layer_config`,**不再暴露扁平的 `global_head_dim` 属性**:

```
getattr(tc, "head_dim", ...)                   -> 256
getattr(tc, "global_head_dim", ...)            -> ABSENT
tc.per_layer_config[5].head_dim                -> 512   # 真值在这里
```

于是那句 `getattr` **静默取了默认值 256**,7 个 full attention 层按
sliding 的几何建出来,512 的权重装不进 256 的参数——断言只是症状。

12B 是同一个病,深一层:`num_global_key_value_heads` 也 ABSENT。它的
full attention 层是 `attention_k_eq_v`(**没有 v_proj**,K 当 V 用,
KV head 数 1):

```
full 层:q (8192,3840)=16x512   k (512,3840)=1x512   v 不存在
```

`num_global_key_value_heads` 被吞成 8,于是 fused QKV 的 K 分片按
8x512=4096 去 narrow 一个 512 的张量 → `start (0) + length (4096)
exceeds dimension size (512)`。

### 修法:把被藏起来的扁平属性还给 vLLM

值不用硬编码,从 `per_layer_config` 的第一个 full attention 层读出来:

```python
tc = AutoConfig.from_pretrained(model).text_config
tc.allow_global_per_layer_attribute_access = True
full = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
layer = tc.per_layer_config[full[0]]
override = {"allow_global_per_layer_attribute_access": True,
            "text_config": {"allow_global_per_layer_attribute_access": True,
                            "global_head_dim": layer.head_dim,
                            # 仅 k_eq_v 模型需要
                            "num_global_key_value_heads": layer.num_key_value_heads}}
```

实测(vLLM **0.23.0**,`hf_overrides` 传入上面这个 dict):

| 模型 | capital | 2+2 | 颜色探针 |
|---|---|---|---|
| gemma-4-E4B-it | Paris | 4 | **3/3** |
| gemma-4-12B-it | Paris | 4 | 2/3(blue→"Black") |

0.27.1 上同一组:12B 拿 1/3(red→"Brown" 也错),文本两问一致。
**0.27 并不更准**,颜色探针的差异是小色块上的感知怪癖(套件对每个模型
先量 baseline,正是为了容纳这种怪癖),不构成选版本的理由。

### 这个 bug 值得上游报

`getattr(config, name, <sliding 值>)` 这种写法在"上游改名/搬家"时
**不报错、只降级**。E4B 的运气是形状对不上直接断言了;要是某个模型的
sliding 与 full 维度恰好能装进去,它会**安静地按错的几何跑**,输出照样
成句。vLLM 侧该做的是:发现 `per_layer_config` 存在时从那里读,或者
读不到扁平属性就显式报错——而不是回落到同构假设。

(附带一条:`AmbiguousGlobalPerLayerAttributeError` **不是**
`AttributeError` 的子类,所以 `getattr(o, k, default)` 不会吞掉它。
我先怀疑的是这条,实测排除。真凶是属性彻底不存在。)

## 三、Gemma 4 的缓存形状:能缓存,但套件的一条假设要改

装得上不等于能认证。E4B 的 KV 布局实测(`gemma4_geometry_*.json`):

```
6 个 KV cache group,只覆盖 24 层(不是 42 层)
  x5  SlidingWindowSpec  4 层  block=32  page=65536  sw=512
  x1  FullAttentionSpec   4 层  block=16  page=65536  sw=None
cache_config.block_size = 16
56 KB/token(24 层)
```

三条结论:

1. **`num_kv_shared_layers: 18` 是真的**:42 − 18 = 24 层有独立 KV,
   其余 18 层复用别人的。所以"层数 × 头数 × 维度"算不出它的每 token 开销。
2. **6 个 group ⇒ 必须走 MP 路径**,和 GDN 家族同一个理由(进程内
   `LMCacheConnectorV1` 不接受混合 KV cache manager)。
3. **两种 block size 并存**:full attention 层 head_dim 512 → 4096 B/token/层
   → block 16;sliding 层 head_dim 256 → 2048 B/token/层 → block 32。
   vLLM 用"改 block size"把两者的 page size 都拉成 65536。
   **这是套件没见过的形状**——GDN 家族的分页组只有一个 block size。

MP 探针(`gemma4_mp_probe_*.json`,独立脚本,不动仓库,与认证跑并行):

| chunk | 结果 |
|---|---|
| 16 | `ValueError: LMCache chunk size 16 must be a multiple of engine group 0 tokens_per_block 32` |
| **32** | **注册成功;pass 1 存,pass 2 装回 2304 token,文本一致,`local_cached=0`** |

LMCache 的真规则写在 `vllm_multi_process_adapter.py:1255`:**chunk 必须是
每个 `tokens_per_block > 0` 的 engine group 的整数倍**(状态组报 0,
所以 Qwen3.6 的 chunk 784 不受 Mamba 组 8192 约束)。

而套件的 `_validate_block_size()` 现在校的是
`hybrid_block_tokens == cache_config.block_size`。对 GDN 家族恰好成立
(784 == 784),对 Gemma 4 会误判:正确的 chunk 是 **32**,
`cache_config.block_size` 是 16。所以这个校验要改成 LMCache 的口径——
**对每个分页组(排除 MambaSpec)要求 `chunk % block_size == 0`**。
改完对已认证的三个混合模型仍然成立(它们的分页组 block 就等于 chunk)。

`hybrid_block_tokens` 这个名字对 Gemma 4 已经不准(它是"LMCache chunk
size",不是"统一 block size"),但它同时驱动 MP 路径选择、padding、
所有命中容差,改名是纯 churn;这次只把语义写进 docstring。

## 四、状态与下一步

- Qwen3.8-27B:合成套件 **26 passed**(1207 s),MME parity 正在跑。
- Gemma 4:E4B / 12B 都已验证可加载可用,E4B 的 MP 缓存也已验证通,
  **但套件还接不了**,要改两处:
  1. `ModelSpec` 加 `hf_overrides` 字段,串到 baseline 引擎、LMCache
     引擎、isolated 子进程和 parity runner(baseline runner 已经会转发
     `extra_engine_kwargs`,所以从 `compute_baselines` 注入即可覆盖
     isolated 场景);
  2. `_validate_block_size()` 换成上面那条"分页组整数倍"的口径。

  这两处都等 parity 跑完再动:parity 阶段还在拉子进程,中途改
  `tests/e2e_mm/` 会污染正在出的那份证书。
- E4B 还带**音频塔**(`audio_config`,`gemma4_audio`)。真要认证音频,
  探针机制得新设计,当前套件的 scope 只到 image/video,证书里要写明。
- `2_` 的顺序表需要修:"vLLM 升级批"这个批次不存在了,Gemma 4 与
  Qwen3.8 都回到当前 venv 的可做队列。
