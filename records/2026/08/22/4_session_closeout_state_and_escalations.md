# 会话收尾:状态交接、待上报事项、方法论账目

日期:2026-08-22 04:36 PDT

技术实质全部在 [`3_read_lock_ttl_root_cause_gemma4_certified.md`](3_read_lock_ttl_root_cause_gemma4_certified.md) —— TTL 根因、Gemma 4 认证、两个 hybrid 场景打通。**本篇不重复那些**,只记 `3_` 里没有的三件事:交接状态、必须上报给维护者的设计问题、以及这轮的方法论账目。

---

## 一、分支与提交状态

工作区干净。`/records` 调用时**没有创建提交** —— 本轮所有改动早已在下列提交里,空提交只会谎报状态。

```
09bc14c0 (dev @ 08-18)
 ├── fix_mp_load_error   e18e55f2   载入失败诚实上报(1 commit)
 └── multi_modal         b1836ce1   ← 当前
                         │
                         ├─ d43e817a  同一个诚实上报修复
                         ├─ ea5a84e1  Gemma 3 认证
                         ├─ befee285  read_lock_expired 标签 + 聚合日志 + L1Manager 属性 + 测试
                         ├─ 2c61c213  harness read-lock TTL(解开 Gemma 4)
                         └─ b1836ce1  hybrid 跑 capacity_eviction / preemption
```

**本轮三个提交(`befee285` / `2c61c213` / `b1836ce1`)全部只在本地。** 相对 `fork/multi_modal` 领先 5 个提交。

按长期约束:**分支只能推我的 fork,PR 用户自己手动做,我不做。** 推送需要明确指令,没有指令就继续在本地排队。

`records/` 已复验被 `/home/bo/LMCache/.git/info/exclude:19` 排除;三个提交的改动文件共 9 个,全是源码/测试,无 artifact。仓库里那 9 个 `.json` 是既有的 workflow matcher / Grafana dashboard / fixture,与本轮无关(查过了,不是我引入的)。

`backup/pre-artifact-rewrite`(`e33973a8`)仍是那 1.7 MB artifact blob 唯一的家,**永不推送**,等 fork 分支确认无误后删除。

---

## 二、必须上报维护者的设计问题(我故意没动)

**retrieve 路径是否应该在 transfer 时刻自己续读锁,而不是继承 lookup 时刻的时间戳?**

这是 TTL 根因的正解所在,但它有真实权衡,不属于一次认证任务的裁量范围:

- TTL 存在是**有理由的** —— 它是防止"客户端 reserve 完就死"把 L1 内存永久钉住的安全阀。
- 续锁就削弱这个安全阀:一个卡死但仍在轮询的消费者可以无限续期。
- 全局拉长或去掉 TTL,是拿一个失效模式换一个内存泄漏。

我做的只有两件不越界的事:把测试套件的超时变成无关项(`MP_SERVER_L1_READ_TTL_S`,理由和它旁边那条 reap timeout 一致),以及让诊断说真话(`read_lock_expired`)。**生产语义的改动标记出来,不擅自实现。**

一个重要的缓和事实:`d43e817a`(诚实上报)已经把这个隐患从"静默错答案"降级成"安全失败" —— 非 hybrid 上是 load error → 重算,hybrid 上是响亮打挂。所以这条上报是**性能/可用性**议题,不是正确性紧急事项。

---

## 三、这轮的方法论账目

**记账的意义在于:同一类错我今天犯了两次,而且第二次是第一次的升级版。**

`2_` 里的教训是"把一段代码当嫌疑犯之前,先证明它被执行过"(滑窗截断路径 `skip` 恒为 0,从未执行)。

今天的升级版:**还要证明你看到的分布不是控制流裁出来的。** 7699 条失败 100% 落在 `object_group_id=0`,我在 `2_` 里把它写成"读锁归属集中在一个 object group",实际上是 `lmcache_driven_transfer.py:1366` 的 `break` —— 六个 group 同时到期,group 0 永远先失败,循环走不到别的 group。**一个 `break` 被我读成了数据特征。**

**"跑得短" 是个我一直没当成变量的变量。** Gemma 3 干净、五张 chunk 16 证书全绿,我一路把它们当成"多组/小 chunk 无罪"的证据。真实原因是它们全是短跑(Gemma 3 整个双 pass 才 643 秒),从来没隔离出任何东西。**最该早点算的数是"首次失败距 pass2 开始 332 秒",就在 300 旁边。**

**做对了的三件事,值得复用:**

1. **先归档再重跑。** certify 的 parity 输出名由 `workdir / f"parity_{model_key}.json"` 决定,同模型重跑必然覆盖。上次因此只剩 200 行过滤样本,直接导致 `2_` 把起病形状记错。这次先把 4.8 MB 服务端日志压缩存到仓库外并校验条数,再启动重跑。
2. **两条红灯不靠"看起来无关"下结论。** 回归扫描 2 failed,我把改动 stash 掉跑了同一条扫描 —— clean HEAD 也是 2 failed,但**红的 parametrization 不同**。这才是判定既有 flake 的依据(已写入 memory `turboquant-roundtrip-flake`)。提交是在比对之后做的,不是绕过红灯做的。
3. **假设不成立就说不成立。** 我猜"挤紧 block 池能换到更多 preemption",实测 496 与 512 都只有 1 次 —— 假设是错的,于是保留更安全的 512 并把测量写进 spec 注释,而不是留一个看起来调过参的数字。

**差点犯的错(提交前自查抓住):** 我一度把 preemption 对所有 hybrid 打开,而三个 Qwen 没有测量过的 `preemption_gpu_blocks` —— 那会让三个模型的套件挂在我刚修好的那个 `ValueError` 上。教训:**"给某类模型开启某能力"必须按性质 gate,不能按"我测过的清单" gate**,否则新注册的模型默认落到错误的一侧。最终 gate 用的是 `hybrid_family` 与"是否有测量值"这两个真实性质。

---

## 四、下一步(接 `3_` 第八节)

最前面两条:

1. **给 recurrent-state hybrid 测两个数**:能装下 ~205 MB 状态页 object 的 eviction 容量(同时要维持 >2× 溢出,两个要求会互相拉扯),以及三个 Qwen 各自的 `preemption_gpu_blocks`(读 vLLM 拒绝启动时自己报的 "estimated maximum model length")。做完这两项,三个 Qwen 也能拿到这两个场景。
2. **Qwen3-Omni** —— 第一个音频模型,八张证书全部把 audio 列为未覆盖;需要 AIR-Bench/MMAU 而非 MME,探针要重设计。

`3_` 第八节有完整的优先级列表和 carry-over。

---

## 五、本次会话新增/更新的 memory

- `lmcache-silent-load-corruption` —— 从"根因待查"改为 **SOLVED**,写清机制、证明、以及"run length 而非 geometry 才是区分变量"
- `hybrid-models-are-mp-path-only` —— 两个场景已打通(`b1836ce1`),并记下两个**必须实测、无法推导**的旋钮(`preemption_gpu_blocks`、eviction 容量的 64 MB 分配粒度)
- `turboquant-roundtrip-flake` —— 新建,记下"失败条数相同但 parametrization 每次不同"这个既有 flake 的签名及判定方法
