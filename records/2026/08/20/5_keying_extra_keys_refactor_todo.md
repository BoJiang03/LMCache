# keying 设计债签收:extra_keys 重构记为 TODO

日期:2026-08-20
分支:`multi_modal` @ `755272e1`(本地提交,未推送;fork 上是 a3c6a2c3)
前置记录:`4_batch_invariance_control_glm_signoff.md`

## 起点:用户的设计品味之问

用户看完三模型总结后问:"这些修改看起来有点杂。有没有更加优雅的改法?
直击本质的那种?没有也可以,毕竟可遇不可求。"

## 回答:拆"杂",认本质,指终点

1. **"杂"是结构性错觉**:6 个文件里只有一个设计决策(token 替换恢复
   token 流的身份忠实性);三个 MP connector 重复改动是 vLLM 版本锁定
   副本的既有结构债,抢占尾 token 是顺路修的正交 bug,3800 行是测试设施。
2. **token 替换本身已接近本质**:把"支持多模态"归约为"修复一个边界
   不变量"(placeholder token 对不同图片相同 → 换成 mm_hash 的 31-bit
   切片),下游整个栈保持模态无知、一行不改。对"支持这三个模型"是
   正确的最小改动。
3. **但存在更本质的终点形态**:vLLM 式 `key = hash(tokens, extra_keys)`
   ——身份走显式带外通道而非带内走私进 token 流。分水岭是 **Phi-4 模态
   LoRA(P6)**:LoRA 影响全部 KV,没有 placeholder 可替换,身份必须
   带外;届时 token 替换走不下去,extra_keys 是唯一形态。当时没走是
   因为爆炸半径(所有哈希/查找调用点 + MP 线协议 + 存储格式),且对
   图/视频两者 key 空间效果等价。
4. **点名否掉的捷径**:直接消费 vLLM block_hashes——身份计算不重复很
   诱人,但把 key 空间焊死在 vLLM block size 和版本上,跨引擎共享
   (SGLang/TRT-LLM)、缓存可移植性、MP server 离线重算 key 全部报废。
   key 的 engine/transport 无关性是产品性质,不拿它换实现省事。

## 用户决策与落地

用户:"记 todos,等合适的时候我们就来重构。" 落地三处:

- **设计文档**(随代码推 fork):`docs/design/integration/vllm/
  multimodal_cache_keying.md` 新增 "TODO: migrate to an explicit
  extra_keys channel"——终点形态、三个触发器(**Phi-4 模态 LoRA /
  cache_salt / 第二引擎要共享 key 语义**,任一落地才动手)、四步迁移
  草图(扩 `_hash_tokens` 调用点 → MP 协议与 key schema 版本化 →
  mm_hash 降级为 extra-key 生产者 → 弃用 `apply_mm_hashes_to_token_ids`)、
  block_hashes 捷径的否决理由。提交 `755272e1`。
- **长期记忆**:`keying-extra-keys-refactor-todo`(跨 session 提醒)。
- 与优先级清单 #16 Phi-4 / P6(LoRA ID 进 CacheEngineKey)互相指认。

## 经验沉淀

- "优雅改法"之问的正确答复不是辩护现状,而是:分离结构债与设计决策、
  承认现方案的适用边界、写下终点形态和**明确的触发条件**——让重构
  启动时从已记录的决策开卷,而非从头论证。
- 设计债记在 `docs/design/`(随代码走、reviewer 可见)优于只记在
  records(本地)或记忆(只有我可见);三处互为索引。

## 下一步

- 待推送:`1129b2e2`(batch 对照注释)+ `755272e1`(设计 TODO),
  与下批工作一起推 fork。
- #4 InternVL 3.5 铺模型;P5 bypass 护栏代码;MP 竞态家族立项。
- extra_keys 重构:挂起,等触发器(P6 LoRA / cache_salt / 第二引擎)。
