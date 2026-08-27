# 根因闭合:torch 兜底同步 cudaMemcpy 无流排序,staging 槽竞态污染 KV;修复已提交

**日期**: 2026-08-26(当天第 13 篇,接 `12_` 的判决实验)
**代码状态**: `multi_modal@fc5755ca`(本篇新增 1 个修复提交)
**状态**: 根因在原语层 100% 实证;e2e 验证跑在飞(qwen_fixrace,GPU 7)

## 一、判决实验结果(12_ §五的 Run A/B)

1. **Run A(verify 树,GPU 1)崩溃**,非门红:MP server 正常起在 26319,
   但 vLLM worker 300s 内 ZMQ 连不上,server 日志 51 行全是启动段、从未
   见到客户端。原因未明(与本篇根因无关的独立事故),已在 GPU 2 重拉
   (qwen_vtree2,在飞)。
2. **Run B(PR 树、宽限=0,GPU 3)红,签名与 qwen_full3 完全一致**:
   98 answer + 70 parse flips,parse delta 0.0295;自算 59/59 parse 翻转
   100% 单向、全部在索引 0–335(前两个十分位);同一套垃圾词表
   (`''`×17、`'system'`×6、`'assistant'`×2、题文回声)。
   → **5s 宽限提交(bb811138)出局**;pass1 与 qwen_full3 逐字节 0 差
   (store 输入侧稳定);垃圾集合跨跑重叠 26/59(部分时序抽签)。
3. **重大修正:gemma 也是头部聚集**。gemma_full2/full3 的 answer+parse
   flips 都集中在前 3–4 个十分位(fix: 60/79、fix2: 66/88 在 0–3 段)。
   `12_` §三"弃答抖动"的定性漏看了位置分布——四个红跑是**同一现象**。
4. 关键线索:不同题出现**相同的垃圾输出**且索引相邻
   (`'The painting you provided is a portrait of'`@68,72;
   `'Is this artwork titled garden in fonten'`@334,335)——
   同一块陈旧缓冲喂给了不同请求。

## 二、排除:IPC event 生命周期假说(先证后修的价值)

server 的完成事件是函数局部变量,export handle 后随 return 被 GC
(cudaEventDestroy),CUDA 文档称 exporter 销毁后 importer 使用是 UB。
两进程探针(`eventub/probe_event_destroy_ub.py`,GPU 7):importer 在
拷贝飞行中轮询 700–1100 万次,keep/destroy/destroy_churn 三变体全部
**精确在拷贝完成时刻**才翻 True——本机驱动对 IPC event 有跨进程引用
计数,**假说证伪**。

## 三、根因(原语层 100% 复现)

**`lmcache/v1/platform/torch_ops.py::lmcache_memcpy_async` 指针模式用
ctypes 同步 `cudaMemcpy`——跑在 legacy 默认流,与 torch 的 non-blocking
流互不排序。** 而 MP server 的传输路径(`transfer_kv_per_object_group`
torch 兜底)在 `cache_context.stream` 上异步排 gather/scatter kernel,
staging 临时槽(4 个)跨批复用:

- **store(D2H)**:gather kernel(paged KV→槽)还在流积压队列里,
  host 已同步把槽的**旧内容**(上一请求的 KV)拷进 pinned 主机对象
  → **L1 入库即污染**。完成事件记录在流上,但真实数据搬运是 host 侧
  同步做的,事件形同虚设。
- **retrieve(H2D)**:批 N 的 scatter 未读槽,批 N+1 的同步 staging
  已覆盖 → 注入错位内容。

探针 `eventub/probe_sync_memcpy_race.py`(GPU 7):流上排 80ms 积压 +
gather 后,host 同步 cudaMemcpy 读槽 → **100% 旧字节**,无任何报错;
空流时 kernel 恰好赢(0% 旧);`cudaMemcpyAsync` 同流(修复方案)全对。

**为什么是"静箱"触发**:host 越快(无共租 CPU 争抢),从 enqueue 到
同步读的间隔越短、竞态窗口越大;pass 头部洪峰积压最深 → 头部聚集。
重共租拖慢 host → kernel 总赢 → 存档全净。**编译版 cuda_ops 从不受
影响**(原生 C++ 本来就是 cudaMemcpyAsync 同流 + 按 cudaHostRegister
边界拆分)——生产路径无此 bug,这是**纯 torch 兜底路径的缺陷**,本机
因 pyguard 屏蔽被劫持的 cuda_ops 而长期跑兜底,故首次暴露在这里。

一致性核对:pass1 文本逐字节稳(污染只进存储,不影响 pass1 计算);
server 日志零告警(无错误路径);垃圾内容=别的请求的 KV/模板位置
(槽的旧内容);gemma/qwen 形态差=模型维度/chunk 几何不同。

## 四、修复(`fc5755ca`)

- `torch_ops.lmcache_memcpy_async` 指针模式改为镜像原生 C++:
  `cudaMemcpyAsync` 发 `torch.cuda.current_stream()`,按
  `host_buffer_offset`/`host_buffer_alignments` 对齐边界拆分;
  无 async 符号的降级路径先 `current_stream().synchronize()` 再同步拷。
- 回归测试 `test_lmcache_memcpy_async_orders_with_current_stream`:
  非阻塞流上 80M cycles 积压 + gather,断言 D2H 不越序。
  **旧码 100% 旧字节红,新码绿**;test_torch_ops.py 全套 75 过;
  ruff check/format 干净。

## 五、验证在飞与判读

| 跑 | 码 | GPU | 期望 |
|---|---|---|---|
| qwen_vtree2(A2) | 旧(verify 树) | 2 | 共租已回场(~18:00 起占 0/1/5/6),host 被拖慢,可能净;红则双确认 |
| qwen_fixrace | **fc5755ca** | 7 | 绿:answer flips 回 ~18-19 ≤23.74、parse flips 0 |

- fixrace 绿 + A2 红 = 同期同负载 A/B,最强;双绿则 e2e 层在当前噪箱下
  不可判(原语层证据已足),等静箱窗口再复验一次。
- qwen 绿后同法跑 gemma_fixrace(期望 answer ~1、parse delta ~0)。

## 六、对既有结论的影响

1. `10_` 的"LMCache 洗清"**对垃圾现象反转**:垃圾确是 LMCache 兜底
   路径的 bug(已修)。但 **18 核心量子机制不受影响**(native_seq 是
   无 LMCache 复现),qwen 0.01 预算与门拆分的依据仍然成立——垃圾修掉
   后预算校准无需重算。
2. 门/预算冻结:待 fixrace/gemma_fixrace 双绿后解冻关账。
3. 上游报告素材新增:无(此 bug 在 LMCache 侧,不在 vLLM)。

## 七、诚实边界

1. e2e 因果链(修复→红跑变绿)**还没闭合**,在飞;原语层与单元层已闭合。
2. 共租 ~18:00 回场,今晚的绿可能是"负载抑制"而非"修复生效"——
   需要 A2 对照或静箱复验来区分。
3. Run A 首拉的"server 起立但 ZMQ 不可达"崩溃未解释,单独挂起观察
   (A2 重拉若复现再查)。
4. H2D 批间覆盖竞态推理成立但未单独实证(D2H 探针已足以定罪同一原语;
   修复同时覆盖两向)。
5. store 侧污染 vs retrieve 侧错读,两者对垃圾的相对贡献未分离——
   修复后无区分必要,记为一个机制的两个表现。
