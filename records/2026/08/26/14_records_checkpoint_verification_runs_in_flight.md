# /records 检查点:修复已提交,两个验证跑在飞

**日期**: 2026-08-26 18:33(当天第 14 篇,接 `13_` 的根因与修复)
**代码状态**: `multi_modal@fc5755ca`,工作树干净;本篇无新提交
(`fc5755ca` = torch 兜底 `lmcache_memcpy_async` 流排序修复 + 回归测试)

## 一、本检查点状态

`13_` 后无新代码变更、无新结果——本篇是 /records 快照,记录在飞验证的
精确状态,便于中断后接续。

## 二、在飞跑(18:32 采样)

| 跑 | 任务 | 码 | GPU | 阶段 | 预计 |
|---|---|---|---|---|---|
| qwen_vtree2(A2 重拉) | blxqcfm4d | 旧 verify 树 `2485fdbc` | 2 | baseline 渲染 58% | ~19:00 出 |
| qwen_fixrace | bm0yhsbp4 | **fc5755ca** | 7 | 数据集装载 | ~19:35 出 |

- 输出:`$SP/parityfix/parity_qwen_vtree2.json` / `parity_qwen_fixrace.json`
  (+ 同名 `.answers.json`);日志 `$SP/parity0271/qwen_vtree2.log` /
  `$SP/parityfix/qwen_fixrace.log`。
- 共租(vllm-lazy,GPU 0/1/5/6)~18:00 回场,两跑均在噪箱条件下。

## 三、判读(接 `13_` §五)

1. **fixrace 绿 + A2 红**:同期同负载 A/B 闭合,e2e 因果链成立 → 解冻
   门/预算,接 gemma_fixrace。
2. **双绿**:噪箱抑制了竞态,e2e 层今晚不可判——原语层证据
   (探针 100% 复现 + 回归测试旧红新绿)已足以支撑修复正确性;
   等静箱窗口对旧码复验一次红、新码绿再正式关账。
3. **fixrace 红**:修复不完整或另有第二机制,回 `13_` §七.4-5 拆
   H2D/D2H 两向单独实证。
4. A2 若再现"server 起立但 ZMQ 300s 不可达"崩溃(首拉事故),升级为
   独立问题排查;单次不复现则继续挂起。

## 三.5、A2 出结果(18:43 补记)

qwen_vtree2(旧码)跑完,exit 1 是**旧树门限 0.005→11.87 的门红,非垃圾**:
verdict 翻转恰 18(10 Yes→No / 8 No→Yes,近对称)、parse 翻转 0、
全分位散布(十分位 [6,18,3,7,8,6,10,4,0,1] 为原文差异含大小写变体;
verdict 级 18 个无头部聚集)、无模板碎片词表——量子核心指纹,实质净。
score_delta 9.75,hit 0.9842,p1 vs baseline 0/0。首拉的 ZMQ 崩溃未复现,
维持单次事故挂起。

→ 与 `13_` §五预言一致:共租 ~18:00 回场拖慢 host,竞态窗口关闭,旧码
也净。**今晚 e2e A/B 不可判**(判读矩阵进分支 2 的前半):fixrace 若绿,
按原语层证据(探针 100% 复现 + 回归测试旧红新绿)支撑修复正确性,
正式关账等静箱窗口对旧码复验一次红。

## 四、复验清单(fixrace 出报告后)

- gate PASS;answer flips ≈ 18-19 ≤ 23.74;parse flips = 0;
  parse delta = 0;p1 vs baseline 0/0;hit ≈ 0.9842。
- answers.json 自检:无单向 parse 翻转、无头部聚集、无模板碎片词表。
- 绿后:gemma_fixrace(GPU 7,期望 answer ~1、parse delta ~0)→
  双绿解冻门/预算 → certify.py schema 7 端到端(既有 open 项)。

## 五、fixrace 出结果(19:06 补记):实质全绿,qwen 关账(噪箱保留项除外)

- **answer 19 / parse 0 / parse delta 0.0 / p1 vs baseline 0+0 /
  hit 0.9842 / score delta 2.25**——复验清单逐项命中。
- exit 1 是 launch 遗漏:没传 `--max-flip-fraction 0.01`(qwen 校准预算,
  a419a4c5),gate 按默认 0.005→11.87 判的。离线以报告 JSON 重跑
  `parity_gate(report, 0.01)` → **PASS**(19 ≤ 23.74)。数据无损,无需重跑。
- **翻转集合 A/B 完全闭合**:fixrace 19 = vtree2(旧码)18 全部 + 索引 24
  一个新增;±1 正是存档 19/19/18 的抖动。新旧两码同晚同核心集、双方
  0 垃圾——修复码在真实 e2e 路径上无回归,垃圾签名消失
  (但因共租回场,"垃圾消失"归因于修复还是负载抑制,今晚不可分——
  静箱复验保留项不变,见 §三.5)。
- 自检:10 Yes→No / 9 No→Yes 近对称;pass2 全部可解析;分布与 vtree2
  形状一致(核心集本身按 MME 类目排序略偏前,非垃圾头部聚集)。
- **gemma_fixrace 已拉起**(bely6gsyb,GPU 7,fc5755ca,19:10;参数照
  spec:hybrid 32 sliding_window、hf_overrides、parse 门 0.85、L1 280GB,
  flip 预算默认 0.005=11.87)。期望 answer ~1、parse delta ~0.003;
  红过 11.87 或再现单向头部垃圾 = 修复不完整,回 13_ §七.4-5。

## 六、gemma_fixrace 出结果(19:50 补记):门 PASS,双绿达成

- **exit 0,gate PASS**:answer 1(=存档)、parse 5 双向
  (1 答→弃 / 4 弃→答,全是连贯拒答句)、parse delta 0.0012、
  hit 0.9569(=存档)、p1 vs baseline 0+0、score delta 3.95(仅报告)。
- 自检:6 处 verdict 差十分位 [0,0,2,1,0,3,0,0,0,0] 全卷散布,无头部
  聚集、无模板碎片——与红跑(77-86 answer + 141-165 parse 头部垃圾)
  判若两物,就是 `8_`/schema-7 门设计时测过的弃答边缘抖动。
- **判读矩阵落分支 2(双绿)**:修复正确性由原语层证据(探针 100%
  复现 + 回归测试旧红新绿)+ 同晚 A/B 核心集完全一致支撑;
  "垃圾消失归因"的静箱复验保留(共租负载在今晚两跑间还在变化)。
- **门/预算解冻**:a72b68ef 门拆分与 a419a4c5 qwen 0.01 预算维持原样
  生效,无需改动(12_ §七.3 的冻结解除)。
- **certify.py schema 7 已拉起**(19:52,并行):qwen2-vl-2b GPU 7
  (bxqe51g5r)、gemma-4-e4b GPU 2(bc1kpiojm),均
  `--parity-report` 复用今晚 fixrace 报告(load_parity_report 会按
  spec 的 0.01/0.85 重新过门,qwen 那次 launch 漏旗不影响证书)。
  证书落 `tests/e2e_mm/certificate_<key>.json`(未跟踪,覆盖不脏树)。
  启动器 `$SP/parityfix/certify_pr.sh`。

## 七、certify 出结果(20:25 补记):两模型 schema 7 双 SUPPORTED

- **gemma-4-e4b: SUPPORTED**(exit 0,一次过):套件全绿,parity 门
  PASS(answer 1,parse delta 0.0012),commit fc5755ca,tree stable,
  runtime 块 vllm 0.27.1 / torch 2.13.0 / lmcache 0.5.4.dev112
  解析自 worktree(pyguard 生效)。
- **qwen2-vl-2b: SUPPORTED**(第二拉):首拉 NOT_SUPPORTED 是环境
  竞态非套件红——`test_isolated_scenario[mp_connector]` 撞上 vLLM
  显存 profiling 一致性断言(profiling 中 GPU 7 有 ~35GB 被套件内
  上一测试的异步 teardown 释放,98.43→133.33 GiB free),37 过 1 败,
  错误信息自证环境因素;同参数重拉全绿。证书:套件绿 + parity 门
  PASS(19 ≤ 23.74 按 spec 0.01,parse delta 0),fc5755ca stable。
- 至此 `12_` 双红事件的处置闭环:根因(13_)→ 修复(fc5755ca)→
  同晚 A/B 核心集全同 + 双 fixrace 绿(§五-六)→ 门/预算解冻 →
  schema 7 双证(本节)。唯一保留项:静箱窗口对旧码补一次红,
  正式闭合"垃圾消失归因"(§三.5)。
