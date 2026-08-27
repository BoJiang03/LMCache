# 隔离场景源码树污染排查:子进程从未测过本仓库代码

日期:2026-08-20
分支:`multi_modal` @ `d2872fc2`(本地提交,未推送)
前置记录:`1_glm46v_flash_certification.md`(其"后记"是本记录的摘要)

## 起点:第一轮认证链全线失败

后台链(Qwen 3B/2B 重认证 + GLM `--run-parity`)三个 exit 全为 1:

- Qwen 3B:chunked_prefill / capacity_eviction 子进程 import 崩溃
  (`ImportError: device_ops`、`AttributeError: EngineKVFormat has no attribute
  NL_X_TWO_NB_NH_ONE_BS_HS`),mp_connector 全部 0 命中(repeat A 0/297、
  blind-B 负控制都没 trip)。
- Qwen 2B / GLM:mp_connector 同样失败。
- GLM parity:hit_ratio=1.0、分差达标,但 `baseline_answer_parse_ratio=0.5771 < 0.9`
  (新 gate 首次触发),另 pass1-vs-baseline 翻转 14 > 11.87 边缘超标。

崩溃 traceback 里的路径全是 **`/home/bo/LMCache-worktrees/lazy_offloading/`**。

## 取证链(按时间顺序)

1. **import 解析实测**:从 `tests/e2e_mm` 目录 `python -c "import lmcache"` →
   lazy_offloading。从仓库根 → multi_modal(cwd-first)。脚本模式 `sys.path[0]` =
   脚本目录,不含仓库根。
2. **editable install 溯源**:`direct_url.json` =
   `file:///home/bo/LMCache-worktrees/lazy_offloading`,finder/.pth/dist-info 的
   **ctime 与 mtime 都是 08-14 17:06**(硬链接数 3,mtime 本不可信,ctime 定案)——
   不是今天被重指,而是**一直如此**。
3. **"昨晚为什么能过"的矛盾**:负控制(cross-image 隔离)在旧 keying 下应当失败……
   实际不然:lazy_offloading 树带的是**旧 16-bit keying**(`hex_hash_to_int16`,
   与 multi_modal 的 31-bit `mm_hash_to_token_values` diff 99 行)。16-bit 替换在
   小规模场景(个位数图片)碰撞概率极低,负控制照样能过——所以一直没暴露。
4. **行号指纹定案**:mp_glm_6(00:17 运行,通过)日志
   `Registering kv caches! (lmcache_mp_connector.py:479)`。三棵候选:
   multi_modal HEAD=480,lazy_offloading HEAD(924e2c1c)=487,
   **lazy_offloading@05ea8163=479 精确命中**。reflog:05ea8163 是当时的 HEAD。
5. **链条失败的直接诱因**:lazy_offloading 的 reflog 显示 00:21:01 merge
   origin/dev、00:25:55 rebase 完成;.so 重编译时间戳 00:23:44–00:24:02。
   链内 Qwen 场景 00:21/00:23 恰好撞进重编译窗口 → import 崩溃;之后的 mp 场景
   跑的是 924e2c1c(gate-3-at-admission),经济门槛在准入处拒收测试小块 → 0 存储
   0 命中。**不是 multi_modal 的回归。**

## 波及面评估

- **进程内 T 系列测试:不受影响**(conftest.py pin + 硬校验,红/绿证据都来自这里)。
- **benchmark_parity:不受影响**(自带 pin)。GLM parity 失败是真问题(见下)。
- **受影响的是全部 isolated 场景历史结果**(chunked_prefill / capacity_eviction /
  preemption / mp_connector,三个模型):它们认证的是 lazy_offloading 树的旧
  16-bit keying。行为近似(场景全过),但**从未真正测过本仓库 HEAD**。

## 修复(d2872fc2)

1. `isolated_cases.py` 模块级自 pin(仓库根插 `sys.path`)+ `importlib.util.find_spec`
   硬校验——解析到别的树直接 RuntimeError(放在 test-local imports 之后避开 E402;
   catalog/harness/specs 均无模块级 lmcache import,已验证)。
2. `_start_mp_server` 给 `-m lmcache.v1.multiprocess.http_server` 子进程注入
   `PYTHONPATH=仓库根`(PYTHONPATH 先于 site-packages)。
3. **GLM MME 预算**:新 spec 字段 `mme_max_tokens`(GLM=256;0=回落 min_decode_tokens),
   certify.py 转发。真实 MME 照片(OCR 代码题、艺术品)的推理前言远长于合成色块,
   64 token 截断 1004/2374 条 baseline 答案(其中 851 条连 box 都没开)。
4. **`parse_yes_no` 第三种答案形态**:完整 thinking 后无盒装的散文答案
   (`...</think> The code is not Python. So the answer is no.`)→ 取 `</think>`
   后**最后一个** `\b(yes|no)\b`。截断答案仍解析为 ''(parse gate 本职)。
   归档 Qwen parity 报告在新 gate 下重评仍 PASS(离线验证)。

## 验证与当前状态

- ruff check/format 全绿;pin 校验单测:`isolated_cases` import 后
  `_LMCACHE_SPEC.origin` = multi_modal。
- **冒烟(修复后首次真测本树隔离 MP 路径)**:mp_connector@Qwen3B 通过——
  full_hit 288/297、B 隔离仅 16 前缀块、日志指纹 @480 = multi_modal。
  31-bit keying 在隔离 MP 路径行为正确。
- **第二轮认证链在跑**(scratchpad `recert_chain2.sh`):Qwen 3B/2B 套件重认证
  (复用归档 parity 报告)→ GLM `--run-parity` 全量重跑(256 token 预算)。
  预计数小时。全绿则归档证书,GLM-4.6V-Flash 落地 SUPPORTED。

## 经验沉淀

- 长期记忆 `vllm-lazy-venv-build` 已追加:**pin 必须覆盖每一个子进程**;测试子进程
  行为诡异时,第一件事查它的 lmcache 解析到哪棵树。
- 共享 venv + 多 worktree 并行开发 = 定时炸弹:editable install 指向谁,谁的
  mid-edit 状态就会污染别人的测试。per-进程 sys.path/PYTHONPATH pin 是不冲突的解法
  (重装 editable 会反过来炸对方 session)。
- "测试通过"不等于"测对了代码"——行号指纹(日志 `file.py:NNN` 对三棵树 `git show`)
  是廉价而决定性的事后取证手段。

## 追记:真树第一跑暴露 preemption oracle 口径错误(6dc6ce0c)

第二轮链 Qwen 3B/2B 均**只挂 preemption 一项**(各 28/29 过;mp_connector、
chunked_prefill、capacity_eviction 在真树上全绿):
`under-storage: batch missed 1888 but only 1728 store-requested`,缺口 160,
换 GPU 独立复现数字**逐位相同**(确定性,非竞态);replay 全命中与文本校验全过,
缓存最终完整。

**破案**:stored_delta 1728 = 6 请求 × 288 prompt chunk(首轮 prefill 存储),分毫不差。
2 个被抢占请求恢复时把已解码的 80 token 当作输入重新 lookup(计 miss),但 save 路径按
`save_decode_cache=False` 的设计从不存 decode 来源的 token(tracker 的
`is_decode_phase` 跨抢占恢复保持 True)→ 2×80=160 恰为缺口。**不是存储丢失,
是 oracle 在错误的树上校准时没暴露的口径缺陷**。真 oracle 是 replay 全命中检查。

修复(6dc6ce0c):under-storage 松弛加上
`preemptions × chunk_align(PREEMPTION_MAX_TOKENS)`(decode-relookup 项),
注释写明机理;换 GPU 验证 failures=[]。chunked_prefill/MP 的同型 oracle 无抢占,无需动。

设计观察(非 bug,记录备查):抢占恢复的重算段语义上是 decode 输出,跳过存储与
`save_decode_cache=False` 意图一致;若未来想让重算段可复用,需在恢复分支重置
`is_decode_phase` 并按 prefill 处理——那是产品决策,不是缺陷。

## 下一步(已完成,见 1 号记录"终章")

- ~~等第二轮链 GLM 腿~~:parity 报告产出,parse 0.9208 达标,翻转 27/17 超标
  → 取证(baseline 自一致 0 翻转、重跑逐个吻合、44 处逐题全为良性漂移/复读
  截断)→ spec 级 `mme_max_flip_fraction`(GLM=0.015,a3c6a2c3)。
- ~~第三轮三绿~~:qwen2b ✅、qwen3b 重试后 ✅(首跑撞上**第二种 MP 竞态
  flake**:迟到的 kv_xfer_finished 打在已释放请求上,证据已归档)、glm ✅。
  三张 SUPPORTED 证书 + junit + parity 报告归档本目录。
- 推 fork 待办;MP 竞态家族(两案)单独立项。
