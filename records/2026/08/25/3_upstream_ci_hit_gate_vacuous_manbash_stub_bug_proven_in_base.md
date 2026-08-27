# 上游 CI 的命中门是空转的:man-bash 存根让它每天绿;用上游自己的测试证明 bug 属于 base

**日期**: 2026-08-25(当天第 3 篇,接 `1_`、`2_`)
**代码状态**: `multi_modal@0040c6bd`,工作树干净,未提交(本会话没改任何仓库代码;
所有工作是探针、CI 复刻、docker 验证,产物归档在 `vllm_upgrade/`)

## 结论先写

1. **bug 属于 lmcache base,不属于本分支 —— 这次是用上游自己的测试证明的。**
   把 buildkite `vllm-correctness.sh` 的 Phase 2+3 逐字复刻(Qwen2.5-14B、
   chunk 256、`--enforce-eager`、`FLASH_ATTN`、`VLLM_BATCH_INVARIANT=1`、
   `vllm serve` + curl 四步 man-bash 流程),跑在 vLLM 0.27.1 + 上游 HEAD
   `c1ef01b9` 上:**`[CI-GATE] FAIL (outputs differ)`**。且 provenance 是实的:
   STEP 4 重问时 LMCache 报 hit 6144 tokens、真实取回 3072、装载 2944。
   整条链路没有一行我们的代码 —— 不是分支,连我们的探针都不是。

2. **上游 GPU CI 每天在 0.27.1 上标 "tested" 是绿的,原因找到了:它的唯一
   命中路径检查在 CI 环境里是空转的。** CI pod 镜像
   `nvidia/cuda:13.0.2-devel-ubuntu24.04` 从不安装 `man-db`;minimized
   Ubuntu 24.04 的 `man` 是个存根,**输出 46 个词的提示并 exit 0**(在
   `ubuntu:24.04` 父镜像里实测)。于是 `CONTEXT` 和 `HALF_CONTEXT` 都是同
   一段 ~60 token 的文字,小于一个 256-token chunk → LMCache 一个 chunk 都
   存不下 → 永远没有取回 → STEP 4 全新计算 → 两次输出平凡相等 → PASS。
   `set -euo pipefail` 不会拦:存根 exit 0。ShareGPT 阶段(100 条新请求发
   给新 server)只走 miss 路径,miss 在坏版本上本来就是对的。
   **所以 `buildkite_latest_tested_vllm` 分支上每天的 "tested: 0.27.1"
   对 KV 装载路径的覆盖是零。**

3. **CI 的两个配置旋钮假设全部证伪**(都在上游 HEAD、0.27.1、探针带
   provenance 闸门、`valid: true`):

   | 配置 | hit 准确率 |
   |---|---|
   | chunk 16(我们的默认,已知红)| 0.0 |
   | chunk **256**(CI 的值) | **0.0** |
   | chunk 16 + **`VLLM_BATCH_INVARIANT=1`**(CI 的旗标) | **0.0** |

   所以 CI 绿不是配置躲开了 bug,是测试根本没测到。第 2 条才是真解释。

4. **两条 CI 流水线的钉法弄清了**:GPU 侧(buildkite k3)钉的是
   **vLLM 0.27.1 stable**,每天标 tested(空转);CPU 侧(GitHub Actions)
   钉 `vllm-cpu-nightly 0.28.1.dev202608250650`,只测 CPU,碰不到 GPU
   装载内核。`记录 1` 里"上游 CI 测的是 0.28 nightly"只对 CPU 侧成立,
   GPU 侧修正为 0.27.1 —— **也就是说上游 GPU CI 声称支持的版本正是我们
   证明坏掉的版本**,"版本覆盖缺口"的岔路就此关死:这是 bug,不是缺口。

## 一、发现路径(按时间)

1. 读上游 pin 分支:`github_nightly_tested_vllm`(CPU,0.28.1.dev)、
   `buildkite_latest_tested_vllm`(GPU,**0.27.1**,CSV 里 08-19 那天
   "tested" 的 lmcache commit 恰是我们复现过 bug 的 `09bc14c0`)。
2. 读 `vllm-correctness.sh`:发现它的门其实很严 —— man-bash 四步要求逐字
   相等,ShareGPT 一条不同就 fail(temperature 0)。**门不弱,是喂进去的
   语料是空的。**
3. 跑两个旋钮实验(A: chunk 256;B: batch-invariant)→ 都红 → 配置假设死。
4. 逐字复刻他们的 man-bash 流程(带行号标记做 provenance)→ **FAIL** +
   真实装载证据。
5. 查 pod 镜像 Dockerfile(无 man-db)→ 在 `ubuntu:24.04` 实测存根行为
   (46 词,exit 0)→ 空转机制闭环。

腐坏形态与 `记录 1/2` 完全一致 —— miss 正确、hit 胡说:

```
OUT1(冷,正确): "may be quoted inside double quotes by preceding it..."
OUT2(命中,坏): "It seems like you're referring to add punctuation. Q: What is..."
```

## 二、证据与产物(`vllm_upgrade/`)

| 文件 | 内容 |
|---|---|
| `ci_replica/run_ci_manbash.sh` | 逐字复刻脚本(带行号标记 provenance) |
| `ci_replica/ci_manbash_14b_gate_result.txt` | `[CI-GATE] FAIL` + LMCache 装载日志行 |
| `ci_replica/run14b/out1.txt / out2.txt / vllm.log / cpu.yaml` | 两次输出、完整 server 日志、CI 原样配置 |
| `textacc_devhead_0271_chunk256.json` + `runA_chunk256.log` | 旋钮 A(hit 0.0,16/16 各装载 256 token) |
| `textacc_devhead_0271_batchinv.json` + `runB_batchinv.log` | 旋钮 B(hit 0.0) |
| `text_accuracy_probe_chunkparam.py` | 探针加 `PROBE_CHUNK` 环境变量的版本 |

## 三、诚实边界(报上游之前要说清)

1. 存根行为验证于 `ubuntu:24.04`(CI 镜像的父镜像)+ 他们 Dockerfile 的
   apt 包清单审计(无 man-db、无 unminimize),**没有拉 8GB 的 nvidia 镜像
   本体实测**。报告里要照实写。
2. `vllm-correctness.sh` 在 `.buildkite/pipelines/*.yml` 里找不到引用;
   k3 版本 `k3_tests/correctness/scripts/run-correctness.sh` 同样用 man bash
   (只是换了 `sed 's/.\x08//g'` 清控制符)。**具体哪条流水线、多高频率跑
   correctness,是从仓库里看不全的**(可能钉在 buildkite 服务端配置里)。
   但两个版本的语料来源相同,空转机制对两者都成立。
3. 我们的复刻没有跑 ShareGPT 两个阶段(它们是 miss 路径,且要 100 条请求;
   对命中门的判定不需要)。

## 四、顺带发现(不阻塞,记下)

1. **探针进程会在引擎 teardown 时挂死**:三个已完成写盘的探针进程(HND 轮
   两个 + 旋钮 A)挂着不退,各占 ~100GB 显存(GPU 1/6/7),手工 kill 才回收。
   旋钮 B 则是写完结果后被 OOM-kill(exit 137,结果完整)。教训:**每轮探针
   收完结果必须查 `nvidia-smi` 回收,"进程还在"不等于"还在算"。**
2. chunk-256 运行时 LMCache 自己刷了个 WARNING:
   `Pin count of MemoryObj ... is negative: -1. Double unpin occurred`
   (`memory_management.py:819`)—— 独立线索,可能与装载路径 bug 有关也可能无关。
3. **nightly 版本号陷阱**:GPU nightly 轮子版本 `0.26.1rc1.dev1202+g7de96050c`
   (今天 main 的 commit)在版本序上**小于** stable 0.27.1,`uv` 会解析到
   PyPI stable;要装 nightly 必须精确钉版本。`/home/bo/venvs/vllm-nightly`
   已装好该 nightly(torch 2.13 与 dev_head 的 native 构建同版),用户叫停
   后未使用,**留着**(后续要验 "最新 vLLM 上还坏不坏" 时是现成的)。

## 五、下一步(等指令)

1. **报上游**:一份报告两个问题 —— (a) 0.27.1 fused 布局装载腐坏
   (复现物:我们的探针 + 他们自己的 correctness 流程);(b) CI 命中门空转
   (修法:镜像装 `man-db`+`manpages` 或把语料换成仓库内静态文件,并给
   correctness 加 provenance 断言 —— 取回 token 数 > 0,否则测试作废)。
2. 实际缺陷点(`*_CS` 装载内核路径)仍未定位 —— 上游修还是我们修,等指令。
3. `vllm-nightly` venv 现成,想验"今天的 vLLM main 上还坏不坏"是 6 分钟的事。
4. 12 张证书不受影响(0.23.0);DeepSeek-OCR / Mistral 的推进仍卡在这个 bug 后面。
