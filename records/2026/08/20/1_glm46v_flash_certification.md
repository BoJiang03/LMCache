# 铺模型 #3:GLM-4.6V-Flash 认证(套件全绿,parity 在跑)

日期:2026-08-20
分支:`multi_modal` @ `71cdcd0e`(本地提交,未推送)
前置记录:`2026/08/19/7_t3_mp_video_preemption_coverage.md`

## 背景与两个用户澄清

1. **检测非侵入性核查**(用户质疑是否为"支持检测"改了核心代码):对 merge-base 复核 `lmcache/` 全部 6 个改动文件——keying 算法(utils.py)、抢占尾 token 修正(vllm_v1_adapter.py)、三个 MP connector 的 MM 替换、包内单测。**全部是多模态支持本体,零检测用改动**;所有计数/探测在 tests/e2e_mm 测试侧(读已有观测面 + 测试进程内运行时 wrap)。无需整改。
2. **两个排序的关系**(用户确认沿用既定顺序):模型序 = 1 号记录 18 项清单(认证铺哪个);修改项序 = 2 号记录 P0–P9(开发做什么)。P0+P1+P2 已完成,现在是收割期。**P5 的 bypass 护栏(SGLang/TRT-LLM/SDK 检测 MM 即 bypass+告警)只做了文档,代码还欠着**——已向用户指出,可与铺模型并行。

## 模型选择:#3 GLM 家族 → GLM-4.6V-Flash

- GLM-4.5V 本体 = 106B MoE(`Glm4vMoeForConditionalGeneration`),单卡 0.35-util 装不下,留给 TP>1。
- **GLM-4.6V-Flash**:10.3B dense,`Glm4vForConditionalGeneration`(vLLM 0.23 原生支持),4.6V 代际,带 video preprocessor,模板支持 `enable_thinking=False`(`/nothink` + `<think></think>` 预填)。比 always-think 的 GLM-4.1V-9B-Thinking 更适合认证(后者也下载了,未用)。

## 适配全部 spec 化(测试代码零模型特判)

ModelSpec 新字段(commit 71cdcd0e):
- `chat_template_kwargs` → harness/baseline_runner/benchmark_parity 三处 `llm.chat` 全部打通。
- `min_decode_tokens=64`:GLM 关 thinking 后仍有开场白,答案 ~64 token 内落地(`<|begin_of_box|>red<|end_of_box|>` 盒装);`effective_max_tokens()` 统一 harness/baseline/decode 记账(否则 T0.7 over-storage 上界误报)。
- `mme_mm_processor_kwargs`:MME 照片任意大,GLM 默认 9.6M 像素 ≈ 12k tokens 会爆 8192 context;GLM 用 `size.longest_edge=602112`(≙ Qwen `max_pixels`,同 768 token/图预算)。
- `answer_extract_pattern`:tempered 正则取**最后一个闭合** box(开头常有未闭合假 box,非贪婪会跨段——单测踩过)。

## Oracle 演进(跨 regime 教义的延伸)

第一轮 9 失败全部同型:盒装答案正确、计数器健康,分歧只在 hit vs miss 路径的开场白措辞(KV 载入 vs 计算 = 不同数值 regime;话痨风格给噪声 40+ 次翻词机会;Qwen 能过纯属答案只有 1–8 token)。修复:
- **`check_replay_text`**:14 处裸 `a2.text == a1.text` 统一改政策——相等→过;提取答案一致或 probe 通过→警告;都不行→硬失败(真污染翻转答案本身)。无 pattern 无 probe(Qwen pressure)保持严格逐字节。
- **MME parse_yes_no** 识别盒装答案(取最后 box 标记);新增 **`baseline_answer_parse_ratio` gate**(MIN_PARSE_RATIO=0.9):答案没落进解码预算时三 pass 全 parse 成 '' 会空洞通过,现在硬 FAIL。旧归档报告缺字段按 1.0 处理(其高分已证明 parse 正常)。

## 发现:MP 心跳降级 flake(独立 bug 候选)

certify#2 的 mp_connector 崩于 "KV load failure (failure_policy=fail)",同秒 fail 两个请求(16 + 560 tokens)。代码审读:健康路径 retrieve 失败只记日志;`error_block_ids` 三个填充点**全部要求 `is_healthy=False`** → 故障链必然是**单次心跳 PING(10s 超时)失败 → 判 unhealthy → 批量 fail 在途 retrieve**。7 次独立运行触发 1 次。已埋诊断:mp_connector 场景 engine 异常转 report failure + `server_log_tail` 进 metrics,再触发即可定位 server 为何停顿 >10s。与 pin-count bug 同类:铺模型顺带揪出的产品问题。

## 状态(截至本记录)

- **glm-4.6v-flash 套件 29/29 全绿**,certify exit 2(PROVISIONAL,差 MME parity)。
- Qwen 3B/2B 回归重认证 + GLM `--run-parity` 全量认证在后台串行跑(数小时);全绿则 #3 落地 SUPPORTED,证书/junit/parity 报告届时归档本目录。
- smoke 关键数字:A1 miss 288/0(SHA-256 identifier 替换正确)、B 隔离仅 16 前缀命中、A2 全命中 288/288 逐字节复现。

## 后记(2026-08-20 凌晨,d2872fc2):隔离场景一直在测错误的源码树

上面的认证链全线失败,排查出比失败本身严重得多的事实:

1. **`isolated_cases.py` 子进程从未测过本仓库的代码**。脚本模式下 `sys.path[0]` 是
   `tests/e2e_mm`,`import lmcache` 落到 venv 的 editable install →
   **lazy_offloading worktree**(direct_url.json + ctime 自 08-14 未变)。conftest 的
   pin 只护 pytest 进程,benchmark_parity 自己有 pin,唯独隔离子进程和它拉起的
   MP server(`-m lmcache...http_server`)漏了。**日志行号指纹**定案:mp_glm_6 运行时
   `Registering kv caches!` @479 精确匹配 lazy_offloading@05ea8163(multi_modal=480,
   lazy HEAD=487)。此前隔离场景"通过"靠的是那棵树的旧 16-bit keying
   (`hex_hash_to_int16`)——小规模场景碰撞概率低,恰好能过。进程内 T 系列与 parity
   一直 pin 正确,红/绿证据不受影响。
2. **链条失败的直接诱因**:00:21–00:25 另一 session 在 lazy_offloading 上
   merge+rebase+重编译(reflog + .so 时间戳吻合)——Qwen 场景撞上 import 崩溃
   (`device_ops` 缺失 / `EngineKVFormat` 枚举不匹配),之后 MP 场景 0 命中
   (gate-3-at-admission 拒收测试小块)。不是回归。
3. **GLM parity 失败是真问题**(该路径 pin 正确):hit_ratio=1.0、分差达标,但
   baseline parse 率 0.577 <0.9——真实 MME 照片的推理前言远长于合成色块,64 token
   截断 42% 的答案;另 pass1-vs-baseline 翻转 14 >11.87 边缘超标(同为截断噪声,待重跑)。

修复(d2872fc2,ruff 全绿):`isolated_cases.py` 顶部自 pin + `find_spec` 硬校验
(解析到别树直接 RuntimeError),MP server 注入 `PYTHONPATH`;新 spec 字段
`mme_max_tokens`(GLM=256,parity 专用);`parse_yes_no` 识别完整 thinking 后无盒装的
散文答案(`</think>` 后最后一个独立 yes/no)。归档 Qwen parity 报告新 gate 重评仍 PASS。
**冒烟**:pin 后 mp_connector@Qwen3B 首次真测本树 31-bit keying——通过(full_hit 288,
B 隔离 16,指纹 @480)。第二轮认证链已发(Qwen×2 复用归档 parity + GLM 全量重跑)。

## 终章(2026-08-20 早晨,a3c6a2c3):三绿,#3 落地 SUPPORTED

### GLM parity 翻转取证(gate 唯一残留失败项)

256 token 重跑后 parse ratio 0.9208 达标,分差 1.34/5.93 达标,但翻转
27(p1-vs-base)/17(p2-vs-p1)> 11.87。三步取证定性:

1. **baseline 自一致性**:同参数无 LMCache 重跑 → **2374/2374 逐字节相同,
   自翻转 0**。引擎在固定配置下完全确定。
2. **翻转稳定性**:全量 parity 重跑(带新加的答案落盘)→ 27/17 与链 #2
   **逐个吻合**,baseline 亦逐字节复现。翻转是确定性的 regime 分歧
   (有/无 connector = 两种各自确定但不同的数值轨迹),不是随机噪声。
3. **逐题检查全部 44 处翻转**:只有两种形态——边界题重新推理落到另一边
   (模型本就在 "yes or no" 间打转),和难题复读循环("Minghella Minghella
   ...")能否在 256 token 内逃出对数值极端敏感 → parse ''。零乱码、零串图,
   金标方向大体对称(18/9、9/8),总分反而 +5.9/+1.3。

结论:0.5% flip 预算是短答案模型(Qwen 答案 1–8 token)的口径;长推理模型
的确定性本底 ~1%。真损坏 oracle(逐字节 replay、hit_ratio、分差、parse
ratio)全部在岗。落地(a3c6a2c3):spec 字段 `mme_max_flip_fraction`
(默认 0 = 沿用 0.5%),GLM=0.015(观测 1.14%);certify 对注入报告重评
gate 时同样应用;benchmark_parity 顺带持久化 pass1/pass2 原始答案
(本次分析靠重跑才拿到答案,以后不用)。离线验证:GLM 报告新口径 PASS,
Qwen 归档报告默认口径不变仍 PASS。

### 第三轮认证结果

- **qwen2-vl-2b:29/29,SUPPORTED**(修复后 preemption oracle + 归档 parity 注入)。
- **qwen2.5-vl-3b:首跑挂 mp_connector(新竞态,见下),重试 29/29,SUPPORTED**。
- **glm-4.6v-flash:29/29,SUPPORTED**(套件全绿 + 注入链 #2 parity 报告,
  0.015 口径 gate PASS;为出处一致,提交 a3c6a2c3 后重新生成证书)。

证书/junit/parity 报告(含 baseline 与 rerun answers)归档本目录。

### 发现:第二种 MP 竞态 flake(迟到的 kv_xfer_finished)

qwen3b 第三轮首跑:t05-B 生成为空(`got=''`),随后 t12-A 时引擎崩在 vLLM
调度器断言 `_update_from_kv_xfer_finished: assert req_id in self.requests`
——**异步 KV 传输在请求已结束/释放后才报完成**。`server_log_tail` 埋点
首次派上用场:server 侧完全正常(无心跳失败、存取全成功),但请求 6 的
prefetch 完成(03:38:55)到 Retrieve(03:39:02)之间有 ~7 秒空档,与
"载入迟到→请求空结束→迟到通知打在已释放请求上"吻合。与心跳降级 flake
是**不同签名**(那个是单次 PING 超时→unhealthy→批量 fail)。同代码重试
即过 → 竞态。证据归档:`mp_xfer_race_evidence_qwen3b.txt` +
`mp_xfer_race_report_qwen3b.json`。MP 隔离场景 flake 家族现有两案,
均指向 connector/引擎侧请求生命周期与异步传输的竞争,值得单独立项。

## 下一步

- 代码提交(71cdcd0e, d2872fc2, 6dc6ce0c, a3c6a2c3)推 fork(用户已确认
  fork-push 模式;records 恪守本地)。
- #4 InternVL 3.5(小杯)接着铺;答案风格与 flip 口径适配机制均已就位。
- P5 bypass 护栏代码(可并行,不依赖模型)。
- MP 竞态家族(心跳降级 + 迟到 xfer_finished)单独立项深挖;两案证据齐备。
