# e2e_mm 套件混合(Mamba/GDN)模式改造:设计与快照

日期:2026-08-21(04:25–05:00)
分支:`multi_modal` @ `9de8f264`(快照提交,未推送)
前置记录:`4_hybrid_recurrent_state_investigation.md`(立项与可行性实探)、
`5_gemma4_blocked_on_vllm023.md`(Gemma 4 推迟,顺序让位给混合)

## 硬事实:混合模型只能走 MP 路径

进程内 `LMCacheConnectorV1` 带 Qwen3.5-2B 直接引擎初始化失败:
`ValueError: Hybrid KV cache manager is disabled but failed to convert
the KV cache specs to one unified type`。原因:vLLM 只对声明支持混合
的 connector 开放 hybrid KV cache manager,进程内 connector 没声明 →
vLLM 关掉混合管理器 → 又无法把 GDN state spec 和 attention spec 统一。
**结论**:混合模型的认证范围只能是 MP 部署路径,证书不得声称进程内路径。

## 设计:块粒度改变了几乎所有断言的含义

混合模型的可缓存单元是 vLLM 统一 block(Qwen3.5-2B: **N=544**),
不是 16 token 的 chunk。逐条影响:

1. **spec 声明**:`hybrid_block_tokens`(引擎启动时校验实际 block
   size,防止 spec 过时导致断言失去意义或平凡通过)、
   `hybrid_object_groups`(每块存几个对象)。
2. **强制引擎参数**(非调优,来自 hybrid_models.rst):
   `mamba_cache_mode="align"` + `enable_prefix_caching=True` +
   `max_num_batched_tokens>=N`,**测试引擎与基线引擎同时施加**
   (align 改变数值 regime,基线不匹配则输出差异归因错)。
3. **每次 run 前清 vLLM 本地前缀缓存**:align 强制开 prefix caching,
   不清则重放由 GPU 前缀缓存服务,LMCache 根本不被问,计数器测的是
   另一个缓存。
4. **提示整形(最关键的洞察)**:图像 span(196 token)小于一个块。
   若图像落在最后一个偏块,**不同图像的块粒度命中数完全相同**——
   套件的头号检测器(计数器对比)会全盲。因此:
   - `pre_pad`(2 块):给"只差图像"的两请求一段共享可缓存前缀,
     否则第 0 块就分叉,命中恒为 0,同样测不出东西;
   - `post_pad`(4 块):把整块放到图像**之后**,使图像差异传导到
     后续每个块 → 合法命中 = 2 块(1088),串图/致盲命中 = 全部
     (~3264),中间用 `image_span_margin = 2*chunk` 一刀切开;
   - `mid_pad`(2 块):多图请求的图间填充,让 T2.2 部分共享能命中
     到第一张图之后(否则共享前缀凑不满含图的那个块)。
   pad 通过环境变量在 **collection 之前**(`pytest_configure`)设好:
   `test_mm_acceptance` 在 import 期建目录,晚设会让基线与测试拿到
   不同 salt。混合模型必须单独选,混选直接 RuntimeError。
5. **公差全部从 harness 读**(`chunk`/`image_span_margin`/
   `objects_per_chunk`/`expected_full_hit`),非混合模型的数值与
   行为**逐字节不变**(已用 collection 对照:qwen2.5-vl-3b 仍 29/34)。
6. **隔离场景**:三个进程内场景对混合模型不适用(见上),显式从
   参数化里排除;`mp_connector` 场景改为按 spec 取 chunk/对象组/
   更大的 L1(GDN state 页 ~13MB/块)。附带认知:**混合模型的
   chunked prefill 是天然覆盖的**——padded 提示在
   `max_num_batched_tokens=N` 下每次都要跨多个 scheduler step。

## 快照提交

`9de8f264`(7 文件,+531/−127):ruff check/format 全绿,三种模型的
collection 均验证(hybrid 26/31、deepstack 34/34、baseline 29/34),
混选报错。全套件运行**待完成**(冒烟中:GPU 3 正在算全目录基线)。

## 过程经验

- **triton CPATH 坑在临时脚本里复发两次**:`harness.configure_
  environment()` 会设 CPATH,但直接手写的实验脚本不走它,多图/新
  kernel 触发 JIT 时 gcc 找不到 Python.h。临时脚本一律显式带
  `CPATH=/home/bo/venvs/vllm-lazy/include`。
- **先实探再改代码**的收益再次兑现:pre/post/mid pad 的必要性是从
  块粒度算术推出来的,但"进程内不可用""每块 2 对象""不串图"这三条
  都是实探数据,省掉了一轮猜错方向的全套件运行(每轮 ~30 分钟起)。

## 下一步

1. 冒烟子集(T0.1/T2.2/T1.2/T2.1/T0.5/负对照)绿 → 跑全套件;
2. MME parity 的 MP 化(当前 runner 只有进程内 connector 路径);
3. certify.py 的 `CERTIFIED_DEPLOYMENT_PATHS` 按 spec 收窄(混合模型
   只claim MP 路径),再出 Qwen3.5-2B 证书;
4. 之后:Qwen3.6-27B(同机制,大杯)→ 升级 vLLM 批次(Qwen3.8 +
   Gemma 4)。
