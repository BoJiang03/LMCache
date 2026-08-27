# T3 MP 路径 + T2.3 video + T0.11 抢占:三条覆盖线落地,双模型重签

日期:2026-08-19
分支:`multi_modal` @ `dc6d6e05`(3dced28f 三线覆盖 + dc6d6e05 抢占 oracle 修正;均已推 fork)
前置记录:`6_qwen2vl2b_supported_design_closure.md`

## 任务

用户拍板:推进 T3 部署路径 + video/audio(T2.3)+ 抢占重算;特殊架构挂件留到需要时。

## P2 代码修复(生产代码)

- **主 `lmcache_mp_connector.py`**:`on_new_request`(eager prefetch)是最后一个用裸 placeholder token 的 key 调用点——查出来并改走 `tracker.get_token_ids()`(裸 token lookup 会假 miss + 锁泄漏)。
- **`_0180.py` / `_0201.py`**(版本锁定副本,vendor 进 vLLM 的源):整体移植 tracker 级替换(`mm_adjusted_prompt_ids` + `get_token_ids()`),lookup/store/retrieve/free_lookup_locks/allocation-report 全部改道。此前这两个文件 100% 必然串图。
- 单测:`tests/v1/test_mp_connector_mm_keys.py` 参数化覆盖两个 vendored tracker,9/9 过。
- 设计文档 `multimodal_cache_keying.md` 状态升级:所有 MP connector 变体已覆盖;未覆盖仅剩 SGLang/TRT-LLM/SDK。

## 三条新覆盖(tests/e2e_mm)

1. **T3 `mp_connector` 场景**(isolated_cases):真实 `lmcache.v1.multiprocess.http_server` 子进程 + `kv_connector_module_path` 指本仓库主版 connector。重跑 T0.1/0.3/0.5/0.8 + T1 全部 + T2.1/2.2 + 守恒(store 意图 vs server `/status` 驻留对象)+ **本路径独立负控**。计数来自包装 scheduler adapter 的 lookup submit/check 与 worker adapter 的 batched store(`MPHarness`,经 MMHarness 新钩子 `_kv_transfer_config/_install_transport_hooks/_setup_stats` 插入)。T0.4/T0.2 留在 in-process 路径:两路径共用同一 `apply_mm_hashes_to_token_ids`,keyspace 性质与传输无关(README 已写明)。
2. **T0.11 抢占场景**:`num_gpu_blocks_override=128` + `max_model_len=2048` + `ignore_eos` 112-token decode,6 图并发 → 调度器必然抢占(`vllm:num_preemptions` 计数器证实,0 次抢占判 vacuous;离线 LLM 需显式 `disable_log_stats=False`)。断言:batch 输出过 config-matched baseline、replay 全命中且再过 baseline、store 不欠账。**实测两模型均 2 次抢占,resume 各命中 576 tokens——被抢占请求靠 LMCache 免了重算,且输出正确。**
3. **T2.3 video**:catalog 合成 8 帧 224² 纯色 mp4(cv2,per-index 图案,1.7KB/个),`MMRequest.video_indices` + `video_color_request`;测试重跑 T0.1/T0.3/T1 于视频摄取路径。**模态门控 = pytest marker `requires_modality` + collection 反选(deselect 而非 skip)**——"0 skip 才发证"规则不破,claim 恰好与 spec 声明等宽(image-only spec 实测 28/29 collected, 1 deselected)。specs 两模型声明 `{"image","video"}`;audio 无注册模型,留在 known_not_covered。

## 过程中的坑

- 抢占场景首跑失败:128 块 < `max_model_len=8192` 最低需求 → 引擎拒启;`max_model_len` 随块池同缩。
- **2B 认证暴露断言过严**(dc6d6e05 修正):replay(单发全命中)vs 被抢占并发 batch 逐字节相等——跨数值 regime + ignore_eos 垃圾尾混沌放大,2B 尾部分歧但 probe 颜色全对(无污染)。改为 replay 对 config-matched baseline(同 regime)+ probe rescue;真污染仍会 probe 硬失败。3B 之前逐字节碰巧相同,属侥幸。
- **抢占场景确定性复现 pin-count 负数 bug**:preemption 中断 load → "Double unpin" 警告刷屏(memory_management.py:819)。该独立立项 bug 现在有了分钟级复现器:`python isolated_cases.py preemption <model>`。
- scratchpad 里前次会话的旧证书文件差点被当成新结果(等文件的 until 循环秒退)——已清理;判定一律以任务退出码 + 文件 mtime 为准。
- 会话 cwd 曾漂到 lazy_offloading worktree(那里的 lazy_offload_manager 改动是别的工作,未触碰);相对路径命令前必须显式 cd。

## 最终状态(本目录归档)

- **qwen2.5-vl-3b = SUPPORTED @ dc6d6e05**:29/29,0 skip,754s;MME gate 复核过。
- **qwen2-vl-2b = SUPPORTED @ dc6d6e05**:29/29,0 skip,730s;MME gate 复核过。
- 证书 scope(schema v2):deployment_paths 双路径(in-process V1 + MP connector/server)、modalities [image, video]、scheduling 四 regime(单步/分步 prefill、并发 batch、驱逐、抢占重算)。known_not_covered 收缩为:TP>1、remote/disk 后端、audio、MP 路径的 T0.4/T0.2(注明理由)、allocator 对账。
- 场景报告归档:`preemption_3b/2b.json`、`mp_connector_3b.json`(2B 的 mp_connector 在 certify 套件内跑,junit 为证)。

## 下一步

- 横向铺模型(流程含三新场景全自动)。
- pin count bug:用 preemption 场景做复现起点(单独立项)。
- 留待需要时:特殊架构挂件(extra_suites)、audio 模型、remote 后端 T3 第三条路径。
