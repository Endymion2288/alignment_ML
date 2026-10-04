# WB132 — native CKF 状态、测量关联与 strip prediction 审计

## 前瞻合同

### FACT FROM REPOSITORY

起点为 `4station` HEAD `d02d55f569060082c392c517e74dcec25fdf01f5`。WB131 的四维 single-step 实验为 3 PASS / 1 FAIL / 2 UNKNOWN；event12 derivative 不稳定且诊断解超出局部范围。原 P source 从持久参数平铺列表按距 S0 最近选择，没有 exact TSOS 状态类型和 measurement 对照。

当前 `CombinatorialKalmanFilterAlg::makeTrack` 将 Hole 保存为 predicted、Outlier 保存为 filtered、Measurement 保存为 smoothed，并转换到 curvilinear parameters；这些是当前源码规则，不能直接证明 MC24 历史执行。原 parameter loc0/loc1 的 frame 不等于 sensor frame。本阶段不默认 P 是 smoothed/fitted anchor。

### INFERENCE

native 测量状态的全局位置及 exact PRD link 可以绕过本轮 noMaterial ACTS transport/FD，提供另一条模型证据。即使 native residual 小，也只是同 hit 条件预测，不能作为独立 track prior、独立 validation 或 covariance calibration。

### RECOMMENDATION / 本阶段实施

只读同六事件和原 selected CKF track，导出完整 TSOS→measurement/ROT→PRD 与 exact flattened-parameter-index mapping、类型 flags、参数/协方差、native 和 measurement surfaces、fitQuality/TrackInfo。对原 allowlist 的 exact linked Measurement rows，在实际测量面坐标计算 loc0 residual。查清当前 selected P source 究竟关联何种 TSOS；不按结果换 source。

## Frozen inputs / allowed changes / forbidden changes

六个 indices `1,4,8,12,16,20`，147 个 cluster/wafer IDs、WB125 selected container `CKFTrackCollection`、track index 0、selected persistent parameter index 和所有原序列化字段固定。保存源文件 stat、父级 response/provenance/export/manifest/hash、现有 geometry/conditions/identity payload/runtime。原 memberships 支持 exact selected PRD 数 `18/18/18/16/18/23`，总111；原所有 S0 rows（35）和 event20 S2 已知 missing member（1）必须原样保留为36个 unmatched，不以最近 surface/位置补造匹配。

只增加 bounded read-only EDM exporter 与保存复核。预计六次 input-event execution，可本地并发2个独立 Athena processes。**新增 reconstruction、propagation、algorithm field query、truth/SDO、held-out/sealed access 均为0**。不构造 extrapolation tool，不改旧 workbook/outputs、q/p/geometry/field/material/covariance/FD/tolerance，不拟合或求 alignment update。

## Scientific question / hypothesis / metrics

问题一：原 P source 在 exact TSOS 中的类型、是否有测量、参数 covariance/fit metadata 是什么？问题二：原 CKF 的 native measurement-state predictions 是否能解释其实际使用的 selected downstream strips？

主 H：所有六事件的 eligible Measurement（非 Hole/Outlier）exact linked rows，在每个 occupied downstream station 的 native loc0 residual RMS ≤ `.1 mm`，最大绝对残差 ≤ `.5 mm`。这是执行前冻结的 **gross inconsistency screening**，不是校准的精度接受标准。PRD 与 ROT 两种 loc0 residual 均保存；以原 PRD component0 为主量，以 ROT loc1 交叉核查 calibration/interface。

保存每条原 allowlist row 的 exact TSOS index、parameter index、type、原 PRD/ROT local measurement/covariance、native global state、sensor-local prediction/residual、plane distance、finite bounds、q/p。按 station 保存 raw/ROT residual RMS/max、条件 measurement-weighted Q（仅描述，不计算 pull/p-value），native q/p range 和 covariance diagonal。缺 covariance 不伪造；当前 P source covariance 不当独立 prior。

native curvilinear 位置/momentum 在 Athena `TrackParametersCnv_p2` 中以 float 持久化。执行前采用 `max(1e-6 mm,4*epsilon_float32*max(1 mm,|position|∞,|sensor translation|∞))` 作为 **保存位置的几何平面量化容差**，同时保存 strict on-surface 与实际 plane distance；不调整 ACTS stepTolerance，也不传播/外推到邻近平面。该余量不能吸收测量残差或改变 gross residual 阈值。surface transform 与 WB127 的对照阈值仍 `1e-9`，同参数旧字段要求 exact。

## Controls / negative controls / evidence hierarchy

- 既有同 track 全部 native 参数序列精确 identity；flat index 与 TSOS 参数指针对照，不重新用 nearest-z。
- 原测量 memberships 精确闭合；event1 为已知 source/association 对照；全部 S0 absence 和 event20 S2 missing member 为保留缺失对照。
- 合成旋转 sensor / curvilinear frame 差异、错误 residual sign、duplicate/wrong wafer PRD、改变原 parameter、nearest-z 假配、off-plane state 必须拒绝。合法 float persistence rounding 应通过预定量化容差，gross off-plane 必须拒绝。
- 独立保存复核用 sensor axis dot product 计算 prediction，与 exporter inverse-transform readout比较；source index、matched/unmatched coverage、所有 residual/站聚合/q/p 重新计算。
- TSOS flags 是实际 EDM 证据；filtered/smoothed/predicted 语义还需历史 producer/persistence identity 证明。若只有当前源码和 WB125-era freeze，标记 `CURRENT_RULE_CONSISTENT_HISTORICAL_UNKNOWN`，不追认历史算法或材料模型。没有 material-effects TSOS 不能证明 fit 无材料。

## Success / FAIL / UNKNOWN / stop rule

execution PASS 只证明六个输入成功读取并保存；interface PASS 要求原参数/PRD/frame/count 与 exact links 闭合。主 H 需全部111 eligible linked rows可评估且全部站与最大值满足阈值；数值可用的 gross residual 反例为 FAIL，缺关联/参数/有效测量面为 UNKNOWN。36 个预定 unmatched 保留为 `NOT_IN_SELECTED_CKF`，不能称四站 association PASS。P source 结构类型与是否为 measurement anchor另行报告，不预设成功。

无论结果如何，不在本阶段补 propagation、改 q/p、换 fitted state、调 covariance 或修 production。native prediction 自身若不可信，停止以 P 作 physical anchor；native downstream 闭合但 P 语义/模型不兼容时，下一轮才设计明确 physical-model/seed 合同。不得自动进入 Hθ、WB103 bulk、solver 或 ML。

## Artifact / 防 post-hoc tuning

机器合同 `configs/research_review/wp132_native_ckf_state_contract.json`；exclusive namespace `outputs/mc24_four_station_wb132_native_ckf_state_v1/`。执行前源码/合同 commit，并冻结隔离 binary、源/父级实际 artifacts、已选输入及合同 workbook 快照。保存 commands/logs/exit、完整 native export、audit decisions、独立重算、manifest；失败不覆盖、不重试。结果只追加在本文件末尾，freeze 使用执行前 `contract_workbook.md` 快照。

## 保存接口恢复合同（原六次执行之后，恢复计算之前）

### FACT FROM REPOSITORY

六次 exporter execution 均 PASS。原 observer 全部因 `old selected membership changed` 中止分析，原 `summary.json` / `status.json` 均保留 UNKNOWN。前两个 raw exports 显示 S0 有 exact ROT→PRD links，但 TSOS flag 为 Outlier；它们不在 `track->measurementsOnTrack()` 的 accepted Measurement 列表中。原 observer 错把两个不同集合相等当成 identity gate。这是 aggregation/interface failure，同时产生了“原无 accepted membership 不能推导无 exact TSOS link”的新来源证据。

### RECOMMENDATION / 限定恢复

机器恢复合同 `configs/research_review/wp132_typed_coverage_recovery.json`。在新 `saved_typed_coverage_v1/` 内，分别构造 all exact links 与 `Measurement AND NOT Outlier AND NOT Hole` accepted links。后者与原 WB125 accepted membership 逐 ID 闭合，仍为原主 H 规定的111 rows。原147 rows全部保留；36个 non-accepted rows 必须报告实际是否 linked、原 flags、原参数及 residual，不再用 `NOT_IN_SELECTED_CKF` 暗示没有 TSOS link。

不改主 H、gross 阈值、plane precision allowance、参数/source index、任何 numeric 字段，不覆写原失败、raw export、协议或脚本；新 reader 的唯一接口变化是明确 accepted Measurement 集合。源状态诊断保持实际 Outlier/测量字段，不伪造 flags，不删除 excluded states。四项 typed coverage 回归控制 PASS。执行前另行 commit/freeze 新 reader、原 freeze、六个 raw responses/原失败收据；新增 event read、reconstruction、propagation、field query 全为0。
