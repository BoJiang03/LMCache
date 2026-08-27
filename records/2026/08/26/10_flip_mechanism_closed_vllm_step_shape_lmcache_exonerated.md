# 机制闭合:翻转是 vLLM step 形状的数值域差,LMCache 洗清;门拆分 + 预算校准

> **2026-08-26 17:45 更新**:§六.1 的两个全量复跑**双红**,且 qwen 出现
> 本篇未见过的新形态(pass2 头部聚集的单向垃圾输出)——"LMCache 洗清"
> 对 qwen 挂问号中,判决实验在飞。见 `12_`。18 核心的量子机制本身不受
> 影响(native_seq 是无 LMCache 的复现)。

**日期**: 2026-08-26(当天第 10 篇,接 `8_` 的判决实验与 `9_` 的决策点)
**代码状态**: `multi_modal@a419a4c5`(本篇新增 `a72b68ef`、`bb811138`、`a419a4c5` 三个提交 + `test_parity_gate.py`)
**测量树**: 探针跑在 `multi_modal_verify@2485fdbc`(与 `8_` 同);冒烟与全量复跑跑在 PR 树 `multi_modal`
**产物**: `records/2026/08/26/kvprobe/`(6 个探针脚本 + 报告 JSON + 日志,28 件)

## 结论先写

1. **qwen2-vl-2b 18 题核心翻转的机制闭合了**:vLLM 0.27.1 的首 token logits
   在"合批 step"与"小/单独 step"两种执行形状之间有**确定性的 ±1 bf16 量子差
   (±0.125)**。MME 有约 1% 的题 |Yes−No| 落在一个量子之内(110 题子集实测:
   26 题 ≤0.125,11 题**恰好为 0**)。parity 的 pass1 合批执行,pass2 因取回完成
   时序把请求拖成小 step(实测 pass2:原生 0.13s 巨批 vs MP 5s 涓流)——
   **比较跨了两个数值域**,边缘题确定性翻转。

2. **LMCache 存取回路被完全洗清**。`8_` 的定位结论("故障位于 LMCache 的
   store/retrieve 回路")**被今天的判决实验推翻**,已就地标注:
   - 注入 KV 与自算 KV **逐位相同**(n=1:26 块 × 全 28 层 max_abs=0.0;
     并发 110 题:15/18 翻转题逐位相同);
   - n=1 时 LMCache 命中与 vLLM 原生前缀缓存命中的 logprobs **逐位相同**
     (−0.78379,六位小数);
   - 判决实验:**纯 vLLM、全程无 LMCache**,只把 pass2 改成逐题提交 →
     **同样 18 题、同方向、同 ±0.125,全部复现**(`subset_native_seq.json`)。

3. **修复 1(门拆分,`a72b68ef`)**:answer flips(verdict↔verdict)与
   parse flips(`''`↔verdict)分开计;前者保留 0.5% 预算,后者用两 pass 间
   parse-ratio delta ≤ 0.02 约束;score delta 只报告、不判决(单题 ±7.5 量子
   曾让 9.00/2.25/9.75 抖过 10.0 预算线)。9 个 CPU 单元测试
   (`test_parity_gate.py`),证书 schema 6→7。
   **gemma-4-e4b 两个存档跑在新门下 PASS + PASS,判决可复现**(各 1 个
   answer flip,parse delta 0.0008/0.0000)。

4. **修复 2(qwen 预算,`a419a4c5`)**:`mme_max_flip_fraction=0.01`,
   与 glm-4.6v-flash(0.01)、另两个 spec(0.01/0.015)同一先例;依据是
   上面闭合的机制(18-19/2374 = 0.80%,无 LMCache 复现)。存档三跑重判:
   answer flips 19/19/18 ≤ 23.74 → PASS。

5. **修复 3(冒烟可用性,`bb811138`)**:pass1 与 pass2 之间加 5s store
   落地宽限。`--limit 40` 冒烟的 hit_ratio 从 0.14(伪影)回到 0.95。

6. **在飞**:两个全量复跑(qwen GPU1、gemma GPU7,PR 树,新门+新预算),
   预计 ~1 小时。绿则本轮 parity 关账。

## 一、探针链:九步,每步排除什么

| # | 探针 | 结果 | 排除/证明 |
|---|---|---|---|
| 1 | `rt241`(n=1 存取回路) | KV 逐位同(max_abs=0);logprobs 整体平移 0.0073 但 Yes−No 差**精确不变**(0.125000) | 排除字节损坏@n=1;发现平移只动归一化项 |
| 2 | `native241`(n=1 原生对照) | 原生冷跑第二兄弟 cached=400/422;warm logprobs 与 LMCache hit **逐位同** | LMCache≡原生@n=1;`8_` 里"原生替换无害"的推断前提错了——**原生根本没发生过替换**(兄弟当场复用,副本从未存在) |
| 3 | `batchkv`(solo vs 13 请求批) | 批组成不改 KV(0 差异);兄弟副本共享前缀 400 token 逐位同 | 排除批方差 KV、排除去重替换假设 |
| 4 | `subset_mp` / `subset_native`(110 题) | MP:18 翻(**全部**落在全量跑已知翻转集合)+ 47 题 gap 移动全为 ±0.125;原生:0/0/0 | 现象 7 分钟可复现;原生同题同规模完全稳定;MP pass1 cached=0(pass1 无命中,店后落地) |
| 5 | 四通交叉比对 | **mp_p1 ≡ nat_p1 ≡ nat_p2(逐字全同);唯 mp_p2 偏离**(19 题) | 偏离条件唯一化;"复用 vs 全算"域理论死(nat_p1 混合复用也稳定) |
| 6 | `mixed`(并发注入逐位验证) | 15/18 翻转题注入字节逐位同;3 题(299/741/823)读数大差(max_abs≈19、整块级)——判为**快照被块回收污染**;18 翻第三次逐题复现 | 排除并发下的注入损坏(对可信读数);又一个时序环境下翻转集合不变 |
| 7 | tqdm 时序 | nat_p2 0.13s/818it/s(巨批)vs mp_p2 5s/25.6it/s(涓流) | 形状域假设成形:取回延迟错峰放进小 step 域 |
| 8 | `native_seq`(**判决**) | 纯 vLLM、pass2 逐题提交:**18/18 同题同向 ±0.125 复现,无 LMCache** | 机制闭合;LMCache 洗清 |
| 9 | `smoke40`/`smoke40b` | 新管线端到端跑通;发现 --limit 下 store 未落地伪影(0.14),5s 宽限后 0.95 | 修复 3 的依据 |

## 二、机制细节

1. **bf16 量子**:logits 是 bf16,这个量级下 |gap| 是 0.125 的倍数。所有观测
   到的 gap 移动都是恰好 ±0.125,没有一例其他幅度——这是"数值路径差一个
   rounding"的指纹,不是损坏的指纹(损坏是任意幅度)。
2. **为什么 `8_` 的 drift 对照没翻**:它两遍都在合批域(passB 一波全进,
   0.13s/12.4×),从未踩进小 step 域;而且原生**冷跑就有兄弟复用**
   (110 题子集实测 pass1 cached=16944),`8_` 据此排除"批形状漂移"是排错了
   对象——它排除的是"合批域内部的稳定性",不是"跨域"。
3. **翻转集合为什么跨规模稳定**:域间差是逐题内容的确定函数,与批内邻居无关
   (n=110 与 n=2374 翻同一批题;边缘成员 58/241/24 在规模间进出,对应
   jaccard 0.90-0.947)。gemma 的 jaccard 0.11 主要是 parse 边缘题
   (门修复吸收),其 answer flip 两跑都是 1。
4. **为什么 n=1 不翻**:单请求时 pass1 也好 pass2 也好,241 的 gap 都不动
   (0.125→0.125)——该题跨域差恰好没在决策方向上;大批量时它的邻居们
   (行为等价类)有 ~40% 出现 ±0.125 位移,压在 0/0.125 边缘的题翻。

## 三、改的代码

| 提交 | 内容 |
|---|---|
| `a72b68ef` | 门拆分:`FlipCounts`/`count_flips`/`answer_parse_ratio`;gate 判 answer flips + parse-ratio delta;score delta 降为报告项;`MAX_PARSE_RATIO_DELTA=0.02`;报告新增 6 字段;旧报告回退合并计数(只会多杀不会放过);README T0.6 同步;certify schema 7;`test_parity_gate.py` 9 测试 |
| `bb811138` | `STORE_COMMIT_GRACE_S=5.0`:pass1→pass2 之间等异步 store 落地 |
| `a419a4c5` | `qwen2-vl-2b` spec:`mme_max_flip_fraction=0.01` + 机制注释 |

## 四、重新判卷(存档产物,新门)

| 跑 | answer flips (p2p1) | parse flips | parse delta | 旧门 | 新门 |
|---|---|---|---|---|---|
| qwen B | 19 | 0 | 0.0 | FAIL | PASS(预算 23.74) |
| qwen C | 19 | 0 | 0.0 | FAIL | PASS |
| qwen D | 18 | 0 | 0.0 | FAIL | PASS |
| gemma r1 | 1 | 4 | 0.0008 | PASS | PASS |
| gemma r2 | 1 | 14 | 0.0000 | **FAIL** | PASS |

gemma 的判决从"掷硬币"变成两跑一致;qwen 的 PASS 依据是机制闭合后的
预算校准,不是放水——同样的门对 2026-08-21 的 KEY_NOT_READABLE 损坏
(parse delta ~0.4、1288 翻)仍然会红,单元测试里有这个用例。

## 五、诚实边界

1. **在飞的两个全量复跑还没回来**;上表 qwen/gemma 的 PASS 是存档重判,
   正式关账等新报告。
2. `native_seq` 证明的是"逐题提交进入与 MP pass2 相同的数值域";MP pass2
   的真实 step 分布没直接观测(用 5s vs 0.13s 时长 + 三个时序环境下翻转
   集合逐题一致间接钉住)。
3. `mixed` 里 3 题的大差异读数解释为"快照被块回收污染"是**推断**:证据是
   差异呈整块级、max_abs≈19(内容级差异而非数值噪声)、且同题答案幅度与
   ±0.125 相容;没有直接验证块回收。设计上的教训:pass 结束后快照不可靠,
   要在 store op 提交点抓。
4. **±0.125 域差的 kernel 级根源没挖**(哪个 kernel、什么尺寸阈值切换)。
   这在 vLLM 侧,不在本 PR 范围;上游报告素材已够(native_seq 是无 LMCache
   的最小复现)。
5. `mme_max_flip_fraction=0.01` 只给了 qwen2-vl-2b;其余模型保持默认,
   跳过的 6 个模型仍未测。
6. 小样本下 `max_flips = fraction × N < 1` 的性质保留(smoke40 红是这个,
   不是缺陷;全量跑才是门的语义)。
7. `certify.py` 端到端(schema 7 出证)仍未跑。
8. gemma 复跑在共享机上(280 GB L1),时长不作性能结论。
9. `subset_mp` pass2 比全量 pass2 有 3/110 字符串差(58/241 级边缘成员),
   跨规模"逐字"一致只对核心集合成立。

## 六、下一步

1. 等两个全量复跑回来,确认双绿(qwen 预计 answer flips ~18-19 ≤ 23.74;
   gemma 预计 answer ~1、parse delta ~0)。
2. 决定是否向 vLLM 上游报 step 形状量子差(素材:`native_seq` 复现 +
   glm-4.6v-flash spec 里 max_num_seqs 5.10% 的旧测量)。
3. 既有 upstream 项不变:`defer_block_free` 活锁最小复现、#4463 补
   "MP 不受影响"实测。
4. 关账后按既定计划转入新模型支持。
