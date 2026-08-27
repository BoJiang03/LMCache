# 心跳的四次尝试、一次 git 失手,和跨会话状态

日期:2026-08-21(19:00–23:40,接 `11_` 之后的第二段)
分支:`multi_modal` @ `e33973a8`,工作区**完全干净**(无 tracked 改动,
也无 untracked——run artifact 现在被 exclude 挡住了)
本条不重复 `12_`–`14_` 的技术结论,只记三样:**心跳那一串失败尝试**
(结论进了代码注释,但"试过什么、为什么不行"值得留下)、**一次 git 失手**
和它没做完的收尾、以及**跨会话状态**。

## 一、心跳:四次尝试,只有第四次有用

Gemma 4 的 parity 连死三次,死法完全一样:连接器判服务端死亡 → 报
`error_block_ids` → vLLM 在混合模型上走
`_update_requests_with_invalid_blocks` → `ValueError: too many values to
unpack`(`8_` 记过的那条上游缺口)。四次处置:

| # | 改动 | 结果 |
|---|---|---|
| 1 | `--max-cpu-workers` 1 → 4(PING 与 LOOKUP 同池) | **仍死**,而且死在**完全相同的偏移**(心跳线程启动后 300 s) |
| 2 | 心跳窗口 60 s → 300 s,worker 数 → 16 | **仍死**,偏移变成 600 s |
| 3 | 心跳窗口 → 21600 s(等于静音) | 服务端**拒绝启动**:reap timeout 不能超过 registration grace |
| 4 | 同时抬 `--worker-registration-grace-seconds` | **过了**,pass 1/2 跑完 |

从 1、2 学到的东西比"修好了"更重要:**耐心和线程数各自把死亡时刻推后了,
但都没有阻止它**。

- 窗口 60 s → 死于第 5 个 ping(300 s);窗口 300 s → 死于第 2 个 ping
  (600 s)。两次都是"**开跑之后发出的第一个 ping 再也没回来**"。
- 如果只是排在数据面后面,那么把池子从 1 加到 4 应当把等待除以 4 ——
  但死亡偏移**一点没动**(还是 300 s)。

所以它不是"排队慢",而是**客户端那个 future 在饱和时根本不完成**——与 `8_`
记的"retrieve 完成延迟 0.3–20 s(服务端 3 ms)"是同一套响应分发机制。
Gemma 4 只是把它放大了 25 倍(chunk 32 vs 784)。

一条设计观察值得单独记:**心跳的 interval 同时就是 ping 的超时**
(`send_ping(timeout=self._interval)`),而且**一次不回就判死,没有重试、
没有连续失败计数**。所以这个参数没法只调"多久探一次"而不动"多有耐心",
两个语义绑在一个数上——这是 `8_` 那条"应改成连续 N 次失败才判死"的
后续证据。

### 为什么"静音心跳"不算把测试改绿

- 没有任何 oracle 读那个 health event;
- 服务端真死了,所有装载都会超时,跑还是会失败,只是消息不同;
- 降级模式本来就在证书的 not-covered 里。

而且**静音之后才看见真正的 bug**(`14_` 的静默损坏)。前三次都死在 pass 2
之前,等于被这个误判挡着看不见。400 题那次也是静音心跳,0 flip ——
说明静音没有制造损坏。

## 二、一次 git 失手:`git add -A` 扫进 28 个 run artifact

两次提交(`7e1d435a`、`4cd7dfee`)用了 `git add -A tests/e2e_mm`,把
**一直故意不跟踪**的 28 个文件扫上了分支:各模型的 `certificate_*.json`、
`parity_*.json`、`suite_*.xml`。这些是"某台机器某一次跑"的证据,历来只
拷进 `records/`。

处置(`e33973a8`):

- `git rm --cached` 全部 28 个(文件留在磁盘上);
- 把三条 pattern 写进 `/home/bo/LMCache/.git/info/exclude`(和 `records/`
  同一个文件、同一个理由:本地产物),这样 `git add -A` 再也扫不到。

**没做完的部分**:blob 还在 `416fdaa2..e33973a8` 这五个提交的历史里
(约 1.7 MB)。要清掉得 rewrite 这几个**尚未推送**的本地提交;
`git filter-branch` 被沙箱拦了(破坏性操作),所以留给用户决定 ——
在这个分支公开之前值得做。已在汇报里点明。

教训:**在一个会产出 artifact 的目录里,永远不要 `git add -A`**。
该目录下 `git status` 长期有一大片 `??` 正是"这些不该进 git"的信号,
我却用了一个会把信号一起吞掉的命令。

## 三、这台机器上的工具坑(新增)

- **Bash 工具默认超时 2 分钟**(上限 10 分钟)。我几次 `until ... sleep`
  等待循环被静默截断,看起来像"检查完了没结果"。长等待要显式传
  `timeout`,且**单次最多 600000 ms**。
- **`A && B && C &` 会把整条链后台化**,不是只后台化 C。因此
  "lint && 提交 && 启动跑"那次 lint 输出丢了、提交没发生、跑也没起来,
  而我先看到的却是 `echo` 打出来的"已启动"。要分步发。
- **`pkill` + `sleep` 的复合命令**在这里会被拦(前台 sleep)。停进程要
  `kill` 之后用 `until ! kill -0 ...` 轮询。
- cwd 会被重置(`12_` 已记),这次又因此让一个 patch 静默没打上
  (`certify.py` 找不到)。**patch 脚本一律用绝对路径 + `assert s != orig`**
  ——后者这次救了我一次:没打上就直接报错,而不是"看起来成功了"。

## 四、跨会话状态

### 认证结果:8 个 SUPPORTED + 1 个 NOT_SUPPORTED

| # | 模型 | 路径 | 结论 | 记录 |
|---|---|---|---|---|
| 1–5 | Qwen2-VL-2B / Qwen2.5-VL-3B / GLM-4.6V-Flash / InternVL3.5-2B / Qwen3-VL-2B | 进程内 + MP | SUPPORTED | 更早 / `1_` / `3_` |
| 6 | Qwen3.5-2B(18 层 GDN) | 仅 MP | SUPPORTED | `9_` |
| 7 | Qwen3.6-27B(48 层 GDN) | 仅 MP | SUPPORTED | `10_` |
| 8 | **Qwen3.8-27B**(48 层 GDN) | 仅 MP | **SUPPORTED** | `13_` |
| — | **Gemma 4-E4B**(滑窗混合) | 仅 MP | **NOT_SUPPORTED** | `14_` |

### 未推送队列:21 个 commit,fork 仍在 `a3c6a2c3`

用户指示仍然有效:**不发 PR,只推自己的 fork,且要显式点头才推**
(已写进 memory `multi-modal-push-policy`)。今天新增 7 个:

```
416fdaa2  register Qwen3.8-27B
7e1d435a  sliding-window hybrid 支持 + 注册 Gemma 4-E4B
4cd7dfee  MP heartbeat 池 + per-model parse floor
47f9e183  心跳窗口按排队时间定
468a906a  心跳静音
46213857  registration grace 跟着 reap timeout 抬
e33973a8  停止跟踪 run artifact
```

### 优先级:最高的不再是"下一个模型"

`14_` 的静默损坏(失败的 L1 读被当成命中)是数据正确性问题,排在任何
模型认证之前。三条后续:

1. 上游修:`KEY_NOT_READABLE` 必须变成 miss;"连接器声称装载了多少 token"
   这个契约要复核。
2. 用小 chunk 重跑一个**已认证**模型,验证这个 bug 与模型无关
   ——现有 7 个 SUPPORTED 的 chunk 都在 544–784,到不了那个压力,
   **这是"没测到",不是"安全"**。
3. 套件加"高对象数持续并发"场景:合成套件连过 5 次 26/26 而 MME 一半
   乱码,说明并发规模这一维覆盖不足。

其余存量:`8_` 的两条 MP 家族课题、`achievable_hit_tokens` 分母偏小
(这次 1.0076,而且它现在还掩盖着"装载失败仍计入 external_cached")、
P5 bypass guardrail、MP race 家族 flake、extra_keys keying 重构。

## 五、一条可复用的判断

**绿的合成套件不是支持的证据。** 今天同一个引擎:26/26 连过 5 次,
MME 上 54% 的答案是乱码。证书要求两层同时绿,今天第一次真正兑现了
这个设计的价值 —— 而且是**在我自己刚改过套件之后**被兑现的,
更说明"我改的东西让它变绿了"永远需要第二层独立验证。
