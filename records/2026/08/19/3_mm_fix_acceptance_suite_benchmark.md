# P0 修复 + 验收套件 + MME 分数等价(开发实施)

日期:2026-08-19
分支:`multi_modal`(commits `939d9f89`、`c976b120`、`51be407f`);`multi_modal_repro`(`8aacf2c0`)
前置记录:`2_lmcache_modification_priority.md`(修改项排序)

## 会话内容

用户确认验收标准设计后开始开发。完成 P0 修复、`tests/e2e_mm` 验收套件、红绿验证、复现脚本、MME benchmark 分数等价层。

## 交付

1. **P0 修复(`939d9f89`)**:`mm_hash_to_token_values()` 取代 `hex_hash_to_int16`——placeholder 每位置一个从完整 mm_hash 派生的 31-bit 值(SHA-256 计数器模式),位置偏移编码、前缀截断稳定、int32 安全。两条替换路径(in-process + 主 MP connector)一次修复,零接口改动。顺带修 lookup 路径丢 preemption 后 decode token 的 bug。设计文档:`docs/design/integration/vllm/multimodal_cache_keying.md`。
2. **验收套件(`c976b120`)**:`tests/e2e_mm/`,T0 正确性 / T1 有效性 / T2 场景 × 模型 spec 表,`LMCACHE_MM_E2E=1` 显式开启,Qwen2.5-VL-3B 22/22 全绿(含 N=800 碰撞压力)。
3. **复现脚本(`multi_modal_repro` @ `8aacf2c0`)**:`repro/mm_hash_collision_repro.py`,录制真实 identifier → 找 16-bit 碰撞对 → 问色验证串图。
4. **MME 分数等价层(`51be407f`)**:`benchmark_parity.py`,三组对照(基线 / miss / hit),标准 MME 感知+认知分数,阈值:翻转 ≤0.5%、分差 ≤10/2800、命中率 ≥0.8。60 题冒烟:hit_ratio=1.000、0 翻转、三组分数逐分一致。**全量 2374 题结果:PASS**——baseline / pass1(miss)/ pass2(hit)三组分数逐分一致:感知 1586.55、认知 613.21、总分 2199.76(14 个类别逐一相同);0 翻转;pass2 命中率 1.000(全部 KV 自缓存恢复)。报告存档 scratchpad `mme_full.json`。

## 红绿证据(#3301 实锤)

- **红**(旧代码 + 本套件,N=800):**6/800 请求全量误命中且答错颜色**(黄图答紫等)。6 起恰好吻合 16-bit 生日期望 800²/2/65536≈4.9。日志存档 scratchpad `red_evidence_old_code_n800.log`、`red_evidence_probe_substituted_values.log`(插桩确认整段填充值 = identifier 低 16 位)。
- **绿**(修复代码):22/22 全绿,N=800 零异常。

## 调试中发现的三个坑(已处理/记录)

1. **LMCache 统计线程偷 interval**:`LMCacheStatsLogger`(`cache_engine.py:2114`,硬编码 `log_interval=10`)每 10s `get_stats_and_clear()`,偷走测试窗口统计(表象:lookup_tokens=0)。harness 改为 Prometheus 累计 counter + 未清 interval 的守恒差分(`harness.cumulative_lookup_stats`)。
2. **模块解析劫持(最险)**:从 `tests/e2e_mm` 子目录跑 pytest 时,`import lmcache` 被 editable 安装解析到 `lazy_offloading` worktree——**e2e 一度在测错误代码树**(所以"绿跑"曾在 N=800 失败:那其实是旧代码的真实碰撞)。conftest 现在强制 pin 本仓库根到 `sys.path[0]` 并断言 `lmcache.__file__` 在仓内。已写入持久记忆(vllm-lazy-venv-build)。
3. **vLLM/NCCL 退出挂死漏 GPU 显存**:冒烟进程输出报告后挂死 25 分钟占 87GB,导致全量首跑 OOM 启动失败。`benchmark_parity.py` 现在报告落盘后 `os._exit`。

另观察到:长跑中 LMCache 刷 1802 条 "Double unpin / pin count negative" 告警(`memory_management.py:819`)——与本修复无关的存量 bug,建议单独立项。

## 环境备注(新增)

- python3.12 dev 头文件已手工装到 `/home/bo/venvs/vllm-lazy/include/`(deadsnakes deb 解包);triton JIT 需要 `CPATH` 指向它(harness 自动设置)。
- 本 worktree 原生扩展已就地编译(`setup.py build_ext --inplace`)。
- `datasets` 5.0.1 已用 uv 装入 venv;`lmms-lab/MME` 已入 HF 缓存。

## 下一步

- P2:`_0180`/`_0201` MP connector 补多模态处理。
- 可发 PR:修复 + 套件两个 commit 在 `multi_modal`,repro 在 `multi_modal_repro`。
