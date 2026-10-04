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

## 最终结果 — 2026-10-04

### FACT FROM REPOSITORY：执行与证据范围

本阶段起点 `d02d55f569060082c392c517e74dcec25fdf01f5`；原实现与前瞻合同提交 `6cdfa9a`，保存接口恢复合同提交 `c44d2884ed3517341ae0e64aa1950cbbe17c6976`。六次本地只读 Athena execution（并发上限2）全部 exit 0；没有追加 execution，没有提交 HTCondor job。新增 reconstruction / propagation / algorithm field queries / truth-SDO / held-out-sealed access 均为0。恢复和独立审计仅消费已保存 JSON，不再次读取事件内容。

| seen index | actual run | actual event | 原 rows | accepted Measurement | all exact links | truly unlinked |
|---|---:|---:|---:|---:|---:|---:|
|1|100043|2141|24|18|24|0|
|4|100043|2310|24|18|24|0|
|8|100044|2327|24|18|24|0|
|12|100044|2268|21|16|21|0|
|16|100047|2305|24|18|24|0|
|20|100048|2287|30|23|29|1|
|合计|||147|111|146|1|

证据根：`outputs/mc24_four_station_wb132_native_ckf_state_v1/`。原 `events/NN/native_response.json` 是 exact EDM export，`events/NN/exit.json`、`athena.log`、`status.json` 保存执行与原分析失败；原 `summary.json` 保留 UNKNOWN。恢复结果在独立的 `saved_typed_coverage_v1/events/NN/audit.json`、`status.json`、`summary.json`，标量 sensor-axis 点积复核在 `saved_typed_coverage_v1/independent_audit.json`。最终 hash 清单为 `docs/wb132_native_ckf_state_result_manifest.json`。

### FACT FROM REPOSITORY：原 UNKNOWN 是接口失败，恢复不改科学阈值

原 observer 六次均在 `old selected membership changed` 中止：它将 all TSOS ROT→PRD links 与 `measurementsOnTrack()` accepted Measurement 集合相比较。恢复 reader 明确采用执行前主 H 已规定的 `Measurement AND NOT Outlier AND NOT Hole`；111 个 accepted IDs 与原 membership 逐项闭合，原147 rows全部保留。35 个 S0 rows 实际有 exact link，类型全部是 Outlier；event20 S2 cluster `10469004361915695104` 真正没有 exact link。

因此须分别报告：原 execution PASS；原 analysis/science UNKNOWN（保留）；恢复 interface PASS；恢复 native hypothesis `PASS_GROSS_SCREEN_ONLY`；独立 artifact consistency PASS。原失败属于 aggregation/interface failure，不能当作 native 物理模型的 scientific falsification，也不能事后抹成原执行 PASS。前瞻文本中“36个 unmatched / NOT_IN_SELECTED_CKF”的表述只对应 accepted membership 缺失；实际 TSOS 缺失仅1个。这个来源认知被数据纠正，原文和原 artifact 均未改写。

### FACT FROM REPOSITORY：native strip prediction

以下均为 accepted Measurement 的 PRD loc0 residual；单位 mm。计算使用 native global position 和其 exact linked measurement 的实际 sensor frame，不将 curvilinear loc0/loc1 当成 sensor 坐标。

| index | S1 RMS | S2 RMS | S3 RMS | accepted max abs |
|---|---:|---:|---:|---:|
|1|0.018978886|0.022748419|0.020414248|0.034952457|
|4|0.025333884|0.032194770|0.021447786|0.071254089|
|8|0.012094607|0.018091885|0.022566471|0.036810440|
|12|0.019298953|0.011985028|0.021064782|0.034195276|
|16|0.009642288|0.024995827|0.014380084|0.054702935|
|20|0.017599921|0.015918588|0.018121160|0.025826957|

全部111 rows有限且通过冻结的 bounds / persistence-plane / frame gates；所有 occupied downstream station RMS ≤0.1 mm、max abs ≤0.5 mm。ROT loc1 与原 PRD loc0 在本 corpus 相等，两条 residual 一致。独立标量点积与 exporter prediction 最大差为 `9.77e-15 mm`。这些是粗大不一致筛查和表示复核；threshold 没有改变，未获得独立预测精度或 covariance calibration。

Event12 被排除于主 H 的五个 S0 Outlier residual（原 row 顺序）为 `[-0.0816382262,-0.1423457240,-0.1262074798,-0.1109143483,-0.1610421747] mm`，全部保存在 excluded diagnostics。不能丢掉这些 rows 后宣称四站闭合。

### FACT FROM REPOSITORY：六个原 P source 全是 Outlier

原持久参数 index 与 exact TSOS mapping 未改变。六个原 P 均有 measurement/ROT→PRD，但都不是 accepted Measurement anchor。当前 CKF `makeTrack` 源码规则为 Outlier→filtered、Measurement→smoothed、Hole→predicted；source SHA `c79e404fc18f62b7cb32352623200f44901202c8dd67e2202755ae8969ba7905` 与 WB125 freeze 一致。实际 Outlier flags 是 EDM FACT；历史 MC24 producer execution identity 未保存，不能将当前源码规则升级为历史执行证明，语义仍为 `CURRENT_RULE_CONSISTENT_HISTORICAL_UNKNOWN`。

为描述来源差异，以下“首个 accepted”按 eligible exact linked Measurement native global z 最小选择，同 z 时 TSOS index 最小；这是保存数据对照，**本轮没有替换 source 或运行新 prediction**。q/p 单位 MeV⁻¹；相对变化为 `(q_accepted-q_P)/q_P`。

| index | 原 P persistent index | 首个 accepted TSOS index | 原 P q/p | 首个 accepted q/p | 相对变化 |
|---|---:|---:|---:|---:|---:|
|1|2|6|−7.26409282746e−6|−7.23708213192e−6|−0.37184%|
|4|3|6|−7.48408417500e−7|−8.22714368527e−7|+9.92853%|
|8|2|6|+2.22964156043e−6|+2.26588421581e−6|+1.62549%|
|12|2|5|+4.49243307949e−5|+3.59273488730e−5|−20.02697%|
|16|2|6|−8.47231664902e−7|−9.04584789189e−7|+6.76947%|
|20|2|6|+3.35762022358e−7|+5.36225580577e−7|+59.70406%|

Event12 对应 source momentum 22.25965 GeV 与首个 accepted 27.83395 GeV。六条 track 的 global fitQuality 实际存在；按上述事件顺序 `(chi2,dof)` 为 `(14.61864,13),(24.36628,13),(11.09909,13),(11.26342,11),(10.83160,13),(13.18443,18)`。全部 material-effects state count 为0，**不证明 CKF fit 无材料**。S0 Outlier 保存的 per-state chi2=0、dof=1，不能解读为完美拟合或据此推导 rejection 原因。

原 covariance 未更改。当前 producer 对 sensor-bound→Trk curvilinear 转换的 covariance 处理仅复制 top-left 并缩放 q/p 单位，未明确做完整 local-frame covariance 变换；历史执行与物理校准均未关闭。ROT dimension=2 也不构成独立二维 strip likelihood，未测方向不能加入额外拟合约束。

### INFERENCE：建立了什么，仍未建立什么

数据排除了“原 accepted membership 缺失等于没有任何 exact TSOS link”的解释，也排除了这些 accepted native states 对已使用 strips 存在本合同定义的 gross residual。它并未证明 all147 rows 为同一粒子、S0 是 accepted CKF 测量、一个 single common track 能在当前 noMaterial 模型下贯穿四站，或 station3 production navigation 已修复。

原 P 具有持久来源，但它不是已验证的 accepted Measurement 起点。来源类型和 q/p 对照足以降低继续围绕这个起点追 float-cache 的优先级；**不证明 q/p 单独导致 WB131 event12 derivative instability**。位置、方向、条件信息及可能的材料处理同时不同。native predictions 在多个 conditional fitted states 上计算，不是一次从共同起点传播的 trajectory，也不是独立的留出验证；native 小 residual 不能推出 Hξ、Hθ、profiled residual、alignment update 可信。

当前最高优先级仍是：明确的共同轨迹起点与冻结物理预测模型是否能解释原四站测量。独立 q/p prior / covariance、particle association、navigation、solver/gauge、bulk 与 formal qualification 均未关闭。本阶段不产生 alignment result 或 ML readiness。

### RECOMMENDATION：唯一 NEXT STAGE（本轮未实施）

**NEXT STAGE = 同六事件 exact accepted Measurement-anchor 的有界四站 nominal common-track prediction 对照。**

未来先提交前瞻机器合同，再用全局 z 最小、同 z 按 TSOS index 最小的 exact accepted Measurement 状态作为新起点；选择仅基于状态身份和坐标，禁止基于 residual 或拟合效果挑选。完整使用该状态的 global pose、direction、q/p，在既有实际 sensor prediction 路径上预测原147 rows；同时保留保存的原 Outlier-P baseline 与本阶段 native per-state predictions 对照。起点位于 S1，必须明确 S0 所需 backward propagation 和下游 forward propagation 的合同；不得靠 z 最近替代 surface/link。

科学问题是“完整起点干预后，冻结 common-track 模型是否具有四站名义预测相容性”。这不是 q/p 单变量因果实验。信息增益来自：区分原 source 语义/初始化问题与仍存的 transport/material/association 不相容，而不先为可疑起点优化导数。该接受测量起点同样是同-track posterior，不能包装成独立 prior。

前瞻设计要求：

- **Frozen inputs**：同六 seen indices、原147 rows与顺序/IDs/measurement covariance、selected track、WB132 raw/export hashes；原 geometry、field、noMaterial model、stepTolerance 与 runtime；旧 P/native 结果保留。
- **Allowed change**：仅预定 exact accepted 起点的完整 pose/direction/q/p 与到达四站所需明确 propagation direction；记录 coordinate conversion 和单位，不引入 covariance prior。
- **Forbidden changes**：自由拟合 q/p/track、FD/derivative campaign、改材料/field/tolerance/covariance、删 S0 Outliers或event20 unlinked row、按效果换起点、多轮调参、truth/held-out、alignment solver、ML。
- **Metrics**：全部147 rows的执行/到面/bounds coverage、每事件每站 signed residual、RMS/max、对原 P 的 residual 改变量；exact linked rows另存 propagated-minus-native prediction；S0 excluded rows和唯一 unlinked row单独标记，但保留在全样本判定。
- **Controls / negative controls**：起点精确身份及 frame/单位 roundtrip；保存 baseline hash identity；重复/错误 PRD或wafer、错误source flags、错误坐标分量必须拒绝；S0 Outlier与event20 missing-link保留，native reference 缺失不能伪造。不能把后验 native residual 当成 independent control sample。
- **Success / failure / UNKNOWN**：未来合同须在运行前冻结 call budget 和到面判定；建议沿用本轮0.1 mm station RMS / 0.5 mm max作为全147rows粗筛，仅称 model-screen PASS。有效 prediction 超阈值为 gross-model FAIL；没有有效 prediction 或接口身份不闭合为 UNKNOWN。覆盖和 residual 判定分开，少算任何原 row 不允许报告整体 PASS。
- **Artifacts / 防事后调参**：保存前瞻源码/合同 commit、input/runtime hashes、逐source identity、完整request/response与每次call/exit/log、逐row残差与站聚合、原baseline差值、独立保存复算、所有失败和manifest。固定一次执行，不在输出后放宽阈值；意外机制只记录，另开合同。

若全四站 model-screen PASS，才设计同六事件局部导数及 nuisance 的预算验证；不能直接恢复 WB103 或声称 independent qualification。若 FAIL，先根据 native 对照与站分布定位 source/model/material/association，形成可证伪的下一合同；若 UNKNOWN，先解决明确的 execution/interface/navigation 缺口。两者都不自动转去调 FD、float cache 或 solver。

### Stop rule / 本轮边界

本轮在六次只读 export + 一次 typed saved-only recovery + 一次独立保存重算后停止。8 项原 reader 控制与4项 typed reader 回归控制通过；独立复算 PASS、冻结输入和恢复 hash gates验证后封存。没有 production 修改、重建轨迹、训练、bulk campaign、held-out/sealed 或推荐阶段的新传播。

原 P 的 numerical 下钻暂停。未来只有在明确 source 和冻结模型已经能解释四站、且数值变化超过预先定义的 prediction/derivative预算并影响 nuisance/alignment 结果时，才值得恢复 self-consistent numerical causal experiment；单个 trial accept/reject 变化不满足恢复条件。

CURRENT STATE: exact TSOS coverage 与 accepted native gross screen 已闭合；原分析 UNKNOWN 保留，四站 common-track 尚未验证。

PRIMARY BLOCKER: 原 P 全是 Outlier，可信共同起点与冻结物理预测模型的四站相容性仍未建立。

NEXT STAGE: 同六事件 exact accepted Measurement-anchor 的有界四站 nominal common-track prediction 对照。

WHY: 一次完整起点干预能直接检验原source是否限制预测，并区分仍存的模型/关联问题。

DO NOT DO YET: 原P float-cache下钻、q/p自由拟合、协方差prior、WB103 bulk、Hθ/solver、held-out qualification或ML。
