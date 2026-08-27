# T3 MP 竞态根因:store 可见性滞后 + torch 回退 memcpy 流序缺失(KV 错位腐坏)

**日期**: 2026-08-25(当天第 5 篇,接 `4_`;本篇更正记录 4 的两处误判)
**代码状态**: `multi_modal@0040c6bd`,工作树干净,未提交(全部工作是插桩与
复现实验;插桩经 sitecustomize + import hook 注入,零仓库代码改动;
产物在 `vllm_upgrade/t3_root_cause/`,共 16 项)

## 结论先写

T3 mp_connector 的红是**两个独立的 base 侧问题叠加**,均与 vLLM 版本无关,
均与我们 mm 分支无关:

1. **命中塌陷 = store 可见性滞后(设计缺口,非键丢失)。**
   记录 4 说"存入的键几秒内丢失"——**错了,全程零删除**(DEL/CLEAR/
   finish_read 删除路径插桩,一次未触发)。真相:store 是异步的
   (请求答完即接受下一个,store 提交滞后 + 服务端 reserve_write→
   finish_write 写锁窗口实测 50–300ms),T3 背靠背发请求,下一个请求的
   lookup 稳定跑赢上一个请求的 store:
   - 撞上写锁 → 每键 `KEY_NOT_READABLE`(键在,不可读);
   - 跑赢 store 到达 → `KEY_NOT_EXIST`。
   两者都被前缀折叠放大成整段 0 命中(如 repeat-A 的 lookup 落在 RW 与
   FW 之间仅差 30ms,18 键全灭)。lookup 是一次性的(miss 即锁定不重查),
   协议里没有跨请求"读己之写"兜底。负载越高窗口越宽 —— 08-22 空闲全绿、
   今天全红的唯一自洽解释。
   **→ 更正(记录 6)**:"负载解释 08-22 与今天的差异"不成立。vllm-lazy
   环境(认证环境)今天的全零另有其因:12:19 的 editable 安装在子模块级
   劫持 cuda_ops 到外树旧 .so → store 崩 → 写锁永锁(第三种故障形态)。
   本条描述的可见性滞后仍真实存在(bisect venv 复现,插桩证据有效),
   但它只解释部分命中丢失,不解释 lazy 环境的全零。记录 4 的"18/18 → 1/18 键消失"实为**两条不同
   键链**(只共享 chunk-0,交叉 prompt 公共前缀),当时没打印键才误判。

2. **污染/答错 = torch 回退版 `lmcache_memcpy_async` 缺流序(真正的
   KV 腐坏 bug,已因果证明)。**
   - 位置:`lmcache/v1/platform/torch_ops.py:2130` 指针模式用**同步
     `cudaMemcpy`(legacy 默认流)**;而 gather/scatter 核在 PyTorch
     non-blocking 传输流上。两流互不同步 → 核未写完共享 temp 缓冲,
     memcpy 已把 temp 里**上一个 batch 的残留**拷进 host 对象 →
     chunk i 的键下存进 chunk i-1 的内容,错位跨 store 调用连续传染
     (`A[t05-chunk0] == 真值[t01B-pos17]`)。
   - 触发条件:仅当编译版 `lmcache.cuda_ops` 加载失败退回 torch ops 时
     (C++ 版用 `cudaMemcpyAsync(getCurrentCUDAStream())`,正确;
     `mem_kernels.cu:1340`)。bisect venv 因 torch ABI 不匹配 .so 加载
     失败 → 中招;vllm-lazy(系统 Python,cuda_ops 正常)同轮 T3
     **只有 miss、零污染** —— 交叉验证吻合。
     **→ 更正(记录 6)**:"vllm-lazy 的 cuda_ops 正常"是误判 —— 它加载
     的是 editable 劫持来的 lazy_offloading 旧 .so,其 T3 红是 store
     崩溃+写锁永锁的全零形态,不是滞后 miss。"零污染"观察本身成立
     (该环境根本没有成功的 store,谈不上污染)。本条的 memcpy 流序
     bug 与因果证明(bisect venv,A/B/C 三轮)不受影响。
   - **因果证明**(CRC32 逐 chunk 指纹,三轮对照):
     | 轮 | 补丁 | 链内错位指纹 | 相对 C 移位键数 | 污染失败 |
     |---|---|---|---|---|
     | A(原样) | 无 | 31 | 133 | t05 答 prompt 碎片、t05-B 空答 |
     | B(原样) | 无 | 36 | 128 | (早期空闲段幸免,尾段字节已错) |
     | C(memcpy 前加一行流同步) | 有 | **0** | **0** | **0**(t05 命中且正确答 'Paris') |
   - 两机制互相放大:滞后 miss → 引擎重算、请求节奏更密 → 传输流更忙 →
     错位概率更高(A 轮 store 间隔 ~100ms 全错位,B 轮早期间隔 ~3s 幸免)。

3. **对 mm support 的含义**:T3 红不是 mm 代码问题(dev_head 双侧复现,
   记录 4 归属实验仍然有效),但 MP 是重点支持模式,两个 base 问题都挡在
   mm 的 MP 路径认证前面。修复方向明确(见 §四),都在 base 侧。

## 一、发现路径(接记录 4 §一)

1. 用户指正重点是 MP 模式(非 in-process)→ 根因定位升为第一优先。
2. 读服务端代码列出全部键移除路径;发现 lookup 把 NOT_READABLE 与
   NOT_EXIST 一视同仁、`1/18 retained` 是前缀折叠(断点一个 chunk 即塌)。
3. sitecustomize + import hook 插桩(不动仓库树):L1Manager 全部
   reserve/finish/delete/clear 包一层,逐键错误码 + 键 id;首轮复现即推翻
   "键丢失"(nfail 全是 NOT_READABLE/NOT_EXIST,零删除),并抓到
   repeat-A lookup 落在 RW→FW 窗口内(差 30ms)。
4. t05 现"2/2 全命中却答 'Hello!'" → 腐坏另有其因。审计 store/retrieve
   事件链(store 捕获、块生命周期 device-future、retrieve 异步门控)
   纸面全闭合 → 转向内容指纹:FW 时逐 chunk CRC32,连跑两轮对照。
5. CRC 引爆:249 共享键 137 不一致,且成**移位等价链**(A 的 pos_i 字节
   == B 的 pos_{i-1});错位跨 store 调用连续 → 指向共享 temp 缓冲 +
   跨流竞速。逐层排查 planner(对)、staging(对)、批迭代器(对)→
   锁定 torch 回退 memcpy 用同步 `cudaMemcpy`(默认流)。
6. 因果实验:仅在 memcpy 前加 `torch.cuda.current_stream().synchronize()`
   (运行 C)→ 错位指纹 0、污染 0。C++ 版代码比对确认正确路径本就存在。

## 二、证据与产物(`vllm_upgrade/t3_root_cause/`,16 项)

| 文件 | 内容 |
|---|---|
| `sitecustomize.py` | v2 插桩:L1Manager 逐键错误码 + FW/UR CRC32;引擎侧 adapter 提交/完成时间线;运行 C 的 sync-before-copy 补丁(env `LMCACHE_T3DBG=1` 门控) |
| `run{0,A,B,C}.log` + `mp_server_run{0,A,B,C}.log` | 四轮引擎侧/服务端日志(0=首轮定性,A/B=原样对照,C=因果验证) |
| `result_run{0,A,B,C}.json` | 四轮场景报告(失败清单) |
| `analyze_crc.py` | 移位归因分析(含 2026-08-25 结果快照,可复跑) |
| `t3_wrapper.py` | 直跑 wrapper(补 pytest 环境) |

关键代码位置:
- 滞后:`vllm_multi_process_adapter.py` `maybe_submit_lookup_request`
  (一次性 lookup)、`lmcache_driven_transfer.py` store()(reserve→
  stream-callback finish_write 的写锁窗口;"Stored" 日志在 CPU 侧入队时
  即打出,不代表可读);`l1_manager.py` `available_for_read`(只看写锁)。
- 腐坏:`torch_ops.py:2130`(坏),`csrc/cuda/mem_kernels.cu:1340`
  (C++ 正确参照),`lmcache_driven_transfer.py` torch 回退循环
  (共享 temp buffer + 每 batch gather→memcpy 两步)。

## 三、教训

1. **"键消失"级别的结论必须打印键本身**——两条只共享 pos0 的键链在
   计数日志里与"同一批键丢了 17 个"无法区分,记录 4 因此误判。
2. lookup 的 NOT_READABLE 与 NOT_EXIST 语义完全不同(写锁中 vs 不存在),
   但对上层都显示为 miss;排障必须下探到每键错误码。
3. 内容级验证(CRC 指纹)比协议级审计更快定位数据腐坏——事件链纸面
   闭合不等于数据正确;三轮对照(两坏一修)一次给出因果。
4. "Stored N tokens" 日志≠数据可读:它在 CPU 入队时打印,finish_write
   是 stream 回调。看服务端日志时间线时要牢记。
5. `cuda_ops 加载失败退回 torch baseline` 这行 WARNING 实际是正确性
   分水岭,不只是性能损失——所有无编译扩展的安装(源码 pip 装、torch
   ABI 不匹配)都在跑一条会腐坏 KV 的路径。
6. sitecustomize + meta-path import hook 可以在不动仓库、不进引擎进程
   热路径的前提下对服务端类做手术级插桩,复用价值高(归档件即模板)。

## 四、修复方向(等指令;都在 base 侧)

1. **腐坏(必修)**:`torch_ops.py` 回退 `lmcache_memcpy_async` 补流序 ——
   最小修:指针模式 `cudaMemcpy` 前同步当前流(即运行 C 的补丁,已验证);
   正解:仿 C++ 版用异步拷贝挂当前流(需处理 cudaHostRegister 边界切分)。
   顺带建议把"extension not found"提为更醒目的告警。
2. **滞后(设计改进)**:跨请求读己之写 —— 候选:服务端 lookup 遇写锁键
   报"pending"而非 miss(调度器已有 None=稍后重查的语义,`check_lookup_result`
   现成);或引擎侧对 in-flight store 覆盖的前缀延迟判 miss。
   测试侧可先行:T3 子用例间加 store 落盘屏障(轮询服务端 resident API,
   conservation 检查已在用同一接口)——这能让 T3 在 cuda_ops 正常的环境
   稳定绿,把两个问题解耦。
3. **上游报告**:现在一份报告四个问题 —— (a) 0.26.0+ fused 布局 hit 腐坏
   (记录 4);(b) torch 回退 memcpy 流序缺失致 KV 错位(本篇,复现包
   `t3_root_cause/`);(c) store 可见性滞后无兜底(本篇);(d) CI 命中门
   空转(记录 3)。附健壮性小项:MP handler 崩溃应回错误包(记录 4)。

## 五、诚实边界

1. 运行 C 只有 t05 一次成功注入作为"修后正确性"的直接语义样本(其余
   命中仍被滞后吃掉);腐坏消失的主证据是错位指纹 31/36→0 与移位键
   133/128→0。多轮重复未做。
2. 滞后窗口的量化(50–300ms)来自本机今日负载,不同负载下未标定;
   "机器负载是 08-22 与今天差异的诱因"仍是推断(与两机制的负载敏感性
   自洽,但没做受控负载实验)。
3. 跨轮 CRC 有良性噪声:批次组成不同导致数值级差异(A vs C 169 处
   mismatch 中 133 处归因移位,其余为良性),分析时用移位归因而非
   裸相等计数。
4. C++ cuda_ops 路径"正确"基于代码审读(异步拷贝挂当前流)+ vllm-lazy
   环境零污染的旁证;未在 cuda_ops 环境做 CRC 级验证。
