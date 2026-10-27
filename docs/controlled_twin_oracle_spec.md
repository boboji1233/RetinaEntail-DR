# Controlled Twin Oracle 规范 v1

本规范把“source 中是否有病灶”“观察是否足以辨认”和“是否属于声明域”分开，防止循环地用
RetinaEntail 自己的分数定义其正确答案。版本化测试接口位于
`retinaentail.interventions.oracle`。

## 三层真值

1. `SOURCE_TRUTH`：`PRESENT` / `ABSENT`。描述 controlled source/twin 的命题状态。
2. `SUPPORT_TRUTH`：
   - `IDENTIFIABLE_PRESENT`
   - `IDENTIFIABLE_ABSENT`
   - `OBSERVATION_AMBIGUOUS`
   - `ORACLE_UNRESOLVED`
3. `DOMAIN_TRUTH`：`IN_DOMAIN` / `OUT_OF_DOMAIN`。描述观察是否落在预注册数据、退化与病灶范围内。

`OBSERVATION_AMBIGUOUS` 必须由独立 oracle 工件产生。禁止把 RetinaEntail 的 positive/negative score、
feasible flags、阈值或四状态当成 oracle 输入，否则形成自证循环。

## 独立 oracle tier

- **Tier A — `EXACT_COLLISION`**：规范化解码后的观察数组形状、dtype、值逐元素完全相同。
  `SAME_THETA` 表示 present/absent source 使用同一成像参数；`TWO_THETA` 表示分别存在两组已记录参数。
- **Tier B — `CALIBRATED_SET_OVERLAP`**：使用独立采集/重复测量建立的误差集合，两种 source 的观察集合
  存在交集。集合及校准数据不得由 RetinaEntail 分数构造。
- **Tier C — `EMPIRICAL_EQUIVALENCE`**：冻结的外部指标或盲审判定近似等价，仅为探索性证据。

确认性主分析只接受 Tier A/B。Tier C 不能写成严格不可辨识证明，`ORACLE_UNRESOLVED` 也不能被转换为
ambiguous 或 determinate truth。

## 最小记录合同

每条记录至少绑定：schema version、source/support/domain truth、oracle tier（若适用）、collision mode
（Tier A 必需）、独立 oracle method ID、evidence SHA-256 和 `retinaentail_independent=true`。确认性资格
函数仅在 `IN_DOMAIN + OBSERVATION_AMBIGUOUS + Tier A/B + independent` 时返回 true；其他情况 fail closed
为不具确认性资格。

本 S0 只实现规范与纯测试接口，不声称 S3-pilot 已建立正式 ambiguity benchmark。
