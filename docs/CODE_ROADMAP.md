# CODE ROADMAP

执行基准：远端 master `0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56`。科学依据、已核实数值和限制见 [PROJECT_MASTER_AUDIT.md](PROJECT_MASTER_AUDIT.md)。本文件是下一轮开发计划；**本轮只交付文档，不执行下面的 production、训练或 geometry 变更**。

## 单一主线与资源退出规则

主线：**真实 measurement → field-aware global track likelihood → track/geometry nuisance 消元 → 经认证外部约束 → uncertainty-aware association/alignment 交替 → shadow validation**。

Frozen V2、MLP、原 chi2/route packing 只作 controls。没有 V4、GNN、differentiable network 或新模型调参任务。唯一备选是 **维持原 geometry 的 residual/DQ monitoring，发布限定 observable/model 的负结果与可复现证据包**。主线失败后不再自动派生新 parameterization/source campaign。

```text
T00 evidence/baseline
 └→ T01 immutable/access boundary
     ├→ T02 bootstrap + numerical contract
     └→ T03 geometry/chart contract
         ↓
T10 reference-oracle falsification（依赖 T02；与 T11 无数据依赖）
T11 covariance/transport falsification（依赖 T03）
         └→ T12 measurement-level profiled information pilot
              └→ T20 constrained estimator + survey interface
                   └→ T21 assignment uncertainty
                        └→ T22 bounded physical iteration
                             └→ T30 simulation campaign
                                  └→ T31 confirmatory uncertainty/failure envelope
                                       └→ T40 real-data shadow validation
                                            └→ T41 deployment gate/rollback
```

每个任务先交付单独 reviewable change、短测和机器可读判定，再开始依赖任务。T10 的原 estimator 失败不是整个项目立即终止：它指明必须用 T12 的正确 likelihood。T11 的契约不能建立，则不允许 T12 用未经验证的 covariance 假装继续；T12 的新 estimator 在明确预算内失败，则停止 tracker-only 无约束部署，只有真实合格外部信息可支撑 T20 的受约束分支。没有外部信息时走唯一备选。

T12 的 `conditional_go` 只批准在固定预算内继续 T20–T30 的 development 实现/验证；后续任务的“通过”须区分 contract/feasibility 与 precision/coverage。它不放宽任一最终 bias 或 uncertainty 门，也不允许提前进入 T40。若 T22 smoke 仍无法排除统计涨落，保留 unverified 状态，仅按事先计算的样本预算作一次 T30 判定。

若实施时本地代码已更新：T00 必须列出与此 baseline 的 diff 和已有 fixes；**不得重写用户未提交文件，也不得把后续 workbook 的结论倒填成这次 master audit 的证据**。本地已存在的正确修复可复用，经本任务验收后标记 `already_satisfied`，无需重复实现。

## 全阶段统一科学契约

- 保留旧 S、rank_tolerance=0.01、capture thresholds、负结果和 sealed sources。新的统计定义以新 schema/run ID 补充，不覆盖旧结果、不给旧失败重打 pass。
- 所有 calibration 开发使用 train；开发 validation 已被多次查看，称 development validation。最终 confirmatory study 需要另行预注册来源/条件及一次性开封策略；本路线前期不生成新 test bank，不读取历史 sealed test。
- source UID 包含 immutable original file identity + run/event；不以 overlay event 当独立事件。重采样与 uncertainty 至少按原事件聚类，并报告 source/run 层变化。
- 报告 `structural_gauge`、`numerical_rank`、`legacy_operational_rank`、`absolute_precision`、`model_misspecification` 五类信息，不用一个 rank 替代全部。
- 所有新报告默认 `geometry_write_allowed=false`。MC sandbox payload、real-data shadow payload、official conditions 三个能力分开。零 residual、较低 chi2、SVD 满秩都不构成 official write 凭据。
- 闭环 input 不能有每事件正确 geometry 的 residual；truth 只能在独立 evaluator 和 Stage-1 truth-association control 中使用。选边不含 truth 仍不足以证明 estimator 无 oracle。
- 不能从原 Nov-22 scatter 生成 covariance、把历史 5 mm dz prior 说成 measured survey，或把已作 prior 的测量重新用作独立 validation。

### 新实验的预注册指标

保留旧工程尺度：`|dx bias|,|dy bias| ≤0.1 mm`，旋转暂用 `≤1 mrad`，仅作 simulation pilot 指标，**不宣称已满足 FASER physics analysis 的最终需求**。真实部署前 T40 必须将要求转换为 track extrapolation/momentum/efficiency 的物理误差预算；该预算需领域责任人认可。

新 estimator 同时报：每参数 bias、RMSE、完整 covariance、按 source/run clustered interval、68%/95% coverage、未测弱方向的区间或不可用状态、预测 sensor motion/covariance、objective/predictive residual、assignment purity/efficiency/fakes 和实际用到的原事件数。不得只给 posterior sigma 或 projected norm。小 pilot 无足够 pseudoexperiments 时，只判 bias/contract/feasibility；标 `coverage_not_established`，不得宣称误差已校准。

## Stage 0 — Repository / evidence cleanup

### T00 — 锁定代码、上游构建与证据，重建唯一任务入口

```text
Task ID: T00
Priority: P0
Scientific purpose: 防止把不同代码、数据和 conditions 的实验拼成一条证据链。
Files to inspect: README.md; README_cn.md; docs/project_audit_and_next_plan.md;
  docs/global_alignment_multidof_loop.md; scripts/setup_environment.sh;
  scripts/build_physical_curriculum_corpus.py; scripts/run_physical_refit_capture_scan.py;
  configs/physical_curriculum_v3_expanded_trainval.yaml; 本审查 E01–E09 及其 manifest。
Files likely to modify: README.md; README_cn.md; docs/project_audit_and_next_plan.md;
  docs/global_alignment_multidof_loop.md; pyproject.toml（仅显式依赖/extra）；上述生产脚本的 provenance 部分。
New files if needed: docs/CANONICAL_PIPELINE.md; datasets/evidence_registry.py;
  scripts/audit_evidence_registry.py; configs/audit_evidence_registry_v1.yaml;
  tests/test_evidence_registry.py; environments/ 下的实际版本记录。
Implementation: 明确 inference、paired-response、monitoring 三条入口；旧报告加 dated/superseded 导航。
  为每项 claim 建 code SHA、dirty patch SHA、resolved-config SHA、input GUID/SHA、output SHA、
  schema、seed、source/condition split、UTC、完整 argv、job ID、closure 类型、negative/supersession 状态。
  外部 Calypso 是仓库外依赖：保存其 commit/dirty patch、Acts/Athena/LCG 版本、实际加载库 hash、
  field/material/geometry/conditions tag、IOV 和 overlay payload 的 override 语义。
  不解析 sealed event 文件；只用已有安全 manifest 标识注册封存来源。
Tests: registry 重复 source/缺 hash/不同 config/input 映射应拒绝；缺产物应 MISSING，不能 PASS。
  当前 master 与之后修复的差异必须可审阅；无新 production。
Expected artifact: outputs/master_audit_reconciliation_v1/evidence_registry.json;
  claim_reconciliation.json; external_runtime_manifest.json; 一个可离线恢复的非 event 摘要包。
Acceptance criteria: 本审查 9 项证据状态逐项明确；E09 13 项 hash 重核；每个 active control
  有唯一 checkpoint/config/condition lineage。无法恢复历史 upstream 时标 historical_runtime_unverified，
  不补造 commit，也不将其产物用于新 estimator 的认证。
Do NOT: 重读 sealed test；修改失败数值；自动 git clean/reset；覆盖本地未提交工作；改写历史。
Dependency: 无。
Estimated compute class: 本地 metadata/hash 短任务；大型 ROOT 校验若必要按文件 HTCondor job。
```

正结果：进入 T01–T03。负结果：能恢复的 controls 保留；缺 upstream 认证的部分先解决 reproducibility，不发大规模 refit 作替代。资料确实不可恢复时明确历史结论的可信级别。

### T01 — 统一数据访问与不可覆盖的 artifact 生命周期

```text
Task ID: T01
Priority: P1（防止未来 P0 方法污染）
Scientific purpose: 防止研究入口误开 sealed 数据，或重跑报告悄悄覆盖冻结结论。
Files to inspect: datasets/physical_curriculum.py; datasets/root_loader.py;
  scripts/audit_field_candidate_coverage.py; scripts/run_curriculum_mlp_baseline.py;
  scripts/run_station_pair_threshold_baseline.py; scripts/run_global_assignment_mlp_baseline.py;
  scripts/report_tracker_only_identifiable_subspace.py;
  scripts/report_rigid_station_only_identifiability.py;
  scripts/report_cluster_local_observable_cross_run.py; tests/test_expanded_control_boundaries.py。
Files likely to modify: 上述 loader/CLI/report writer，限制在共享入口与必要调用点。
New files if needed: datasets/access_policy.py; evaluation/artifact_store.py;
  tests/test_sealed_access_policy.py; tests/test_immutable_artifacts.py。
Implementation: 显式传入 AccessScope(train/development_validation/real_data_monitoring/frozen_evaluation)；
  默认不允许解析 test paths；真实访问由 provenance registry 判定，不仅匹配文件名“test”。
  Frozen evaluation 需锁定 plan/checkpoint/calibration/source set 的 capability，并记录开封账本。
  对现有三个 legacy 默认保护保持兼容；审计工具也遵守同一策略。
  新 report 使用唯一 run ID、独占目录、临时文件与同文件系统原子 finalize；
  resume 只重试失败 shard，config/input/runtime hash 不同必须新 run ID。
Tests: 用不存在/触发异常的 fake sealed path 验证 open/resolve 根本未被调用；
  所有 train/validation 入口测试；已有 output 第二次写应 FileExistsError；中断结果不能标 complete。
Expected artifact: access_policy_audit.json; immutable_writer_smoke.json; tests 的安全 fixture。
Acceptance criteria: T00 canonical 全入口默认 fail closed；不能通过 legacy 默认或 report rerun 绕过。
  原 negative artifacts byte-for-byte 不变；不运行真正 sealed evaluator。
Do NOT: 把 --allow-sealed-test 当普通开发许可；修改封存来源；为访问权限测试读取真实 test。
Dependency: T00。
Estimated compute class: 本地 unit/smoke，数十秒。
```

### T02 — 修 bootstrap multiplicity 和数值前提，保持历史判据

```text
Task ID: T02
Priority: P0
Scientific purpose: 让 statistical stability 的含义正确，阻止非法 covariance/null basis 进入后续模型。
Files to inspect: alignment/tracker_only_identifiable_subspace.py::bootstrap_subspaces;
  alignment/cross_source_stable_core.py::event_bootstrap_core_persistence;
  alignment/cluster_local_observable_cross_run.py::official_event_bootstrap;
  alignment/identifiable_subspace.py; baselines/field_chi2_matching.py;
  geometry/propagation.py; tests/test_tracker_only_identifiable_subspace.py;
  tests/test_physical_jacobian.py; tests/test_field_candidate_coverage.py。
Files likely to modify: bootstrap_subspaces 与 subset API；identifiable_svd 的欠定矩阵分支；
  field candidate covariance validation；独立报告的 statistical method 标记。
New files if needed: alignment/resampling.py; tests/test_event_bootstrap_multiplicity.py;
  tests/test_identifiability_numerical_contract.py。
Implementation: event key=(original_source_uid,run,event)；有放回抽样保留 event multiplicity，
  可重复 row indices 或以 sqrt(multiplicity) 缩放整事件 A/b；不得 boolean 去重。
  输出 n_draws/n_unique/effective_multiplicity、seed、invalid replicate 与原因；不静默丢 replicate。
  source bootstrap 同理，cluster-local 已正确的 row-list 分支不回退。
  用经济型 SVD 保留 tall case；m<n 另取完整 right-null complement，验证 dim=n-rank。
  所有用于概率/chi2 的 covariance 要有限、对称、SPD；非法直接拒绝并计数，不 eigenvalue clipping。
Tests: 强制抽样 [event0,event0,event1]，normal contribution 必须 2*N0+N1；
  源重号不能合并；4×2/tall、2×3/wide、零矩阵 null/orthogonality 测试；
  xy block [[1,2],[2,1]] 应拒绝，不能返回 chi2=-2。
Expected artifact: bootstrap_semantics_regression.json；numerical_contract_tests；旧报告标记补充记录。
Acceptance criteria: 历史 S 与 0.01 完全不变；重复 multiplicity 等价于整数权重；
  A*V_null≈0（仅 exact null 测试）、V_id/V_null 完整正交；非法 covariance 明确失败。
Do NOT: 将 bootstrap 修复后 rank 的变化自动转换为 deployment pass；给旧 weak modes 零误差；
  增加真实数据或改变 200/200/80 等旧统计门。
Dependency: T01。
Estimated compute class: 本地数学 unit tests；真实 bank 重采样在 T12 的 shard 中处理。
```

### T03 — 统一 global/center-pivot chart、单位与 finite transform 契约

```text
Task ID: T03
Priority: P0
Scientific purpose: 保证 physical payload、cluster-local Jacobian 和 survey 指向同一物理参数。
Files to inspect: alignment/gauge_equivalence.py; alignment/payload.py;
  alignment/physical_jacobian.py; alignment/true_cluster_local_residual.py::perturb_surface;
  alignment/nov22_metrology_provenance_station_ry.py::station_alignment_transform/extract_alpha_beta_gamma;
  alignment/cad_survey_nov22.py; scripts/write_station_alignment_payload.py;
  tests/test_gauge_equivalence.py; tests/test_nov22_metrology_provenance_station_ry.py。
Files likely to modify: 新模型使用的 chart adapter；上述 extraction/软件 residual 的显式 convention API；
  不静默改变已冻结 campaign 的输出定义。
New files if needed: geometry/alignment_charts.py; tests/test_alignment_chart_contract.py;
  docs/ALIGNMENT_PARAMETER_CONTRACT.md。
Implementation: 一个明确 active/passive、axes、global-origin pivot、mm/rad、rotation order 的 SE(3) API；
  native report mm/mrad 仅在边界转换。区分 absolute payload 与 left-composed increment。
  center-pivot map 用 t_origin=t_center+(I-R)c，并对 Jacobian、prior covariance 作完整线性/非线性传播。
  cluster-local 当前 ry 反号且绕 station z，必须显式命名旧 chart，不能裸复制为 Stations ry。
  native five-column J 可从原 FD 列选取并重算 A；禁止裁剪既有 V/Σ 当新模型，但不把合法 J 列限制污名化。
Tests: 单轴和同时非零 Rx/Ry/Rz 的 build→extract→build round-trip；
  origin/centroid charts 对相同 sensor points 的预测相同；station/layer 不同 pivot 反例；
  covariance cross-terms 正确；J_chart map 和直接几何 FD 一致；正负 mm/mrad 单位检查。
Expected artifact: alignment_chart_contract.json; geometry-only finite_difference_checks.json。
Acceptance criteria: 普通混合角在 1e-10 matrix tolerance 内 round-trip（避开预声明 Euler singularity）；
  analytic/FD discrepancy 随 h 减小，达到相对 1e-6 的数学 fixture 精度；
  不以重新对角化 S 改善旧 rank。旧与新 chart 的关系可逆且可审计。
Do NOT: 写 SQLite/POOL；将 pivot 改变说成新增物理信息；把所有 Euler 参数直接相加。
Dependency: T01；T00 upstream source contract 已记录。
Estimated compute class: 本地几何与小矩阵 unit/smoke，不运行 Athena。
```

## Stage 1 — Critical scientific falsification

Stage 1 只回答三个问题：旧闭环依赖什么 oracle；covariance/transport 是否物理自洽；正确 measurement likelihood 是否具有足够信息。每个实验有固定样本规则和一次判定，没有 rank rescue 分支。

### T10 / Experiment E-A — paired-reference 与无 reference 的最小消融

```text
Task ID: T10
Priority: P0
Scientific purpose: 证伪“truth-free route selection 的 paired closure 已等同于可部署 alignment”。
Files to inspect: alignment/physical_jacobian.py::solve_physical_finite_difference;
  scripts/run_route_selected_multidof_update.py（target_residual/self_nulling_update）；
  scripts/run_multisource_refit_multidof_local_step.py;
  outputs/mc24_multidof_ift_iteration01_route_selected_update_{train,validation}_physical_edge_deduplicated_v1/。
Files likely to modify: 独立审计 driver；现有 solver 的 objective/response 类型 metadata；
  不改变旧 solver 的默认数值。
New files if needed: scripts/audit_reference_oracle_gap.py; configs/reference_oracle_gap_v1.yaml;
  tests/test_reference_oracle_boundary.py。
Implementation: hypothesis=paired-target 成功可以推广到不使用正确 geometry reference 的 estimator。
  Input=上述两个固定 train/development-validation 数组和所有原事件/边，保留 frozen selection。
  在相同 J/W 上计算 delta_pair=N^-1 J'W(r_ref-r_anchor)、delta_zero=-N^-1 J'W r_anchor；
  拆出 N^-1 J'W r_ref，按原 source/event 聚类；同时报告 old operational rank，不调门。
  第二个 entry 接口禁止接收或解析 target residual、target payload 和 truth displacement；
  真值仅由独立 evaluator 事后 join。重算现有数组即可，不做新 refit。
Tests: 与 E02 保存 delta 逐分量 1e-8 一致；去除 reference path 的估计器仍可运行；
  给 reference 放 poison/任意替换，delta_zero 不变；truth 标签置乱不改变同一固定选边后的解。
Expected artifact: outputs/reference_oracle_gap_v1/paired_vs_absolute.json;
  source_effects.csv; input_hashes.json; decision.json。
Acceptance criteria: metric=estimated increment、remaining injected error、reference-projection bias。
  成功=无 reference 与 paired 在既有 0.1 mm/1 mrad 门均合格，且 source 分解无被 pooled 抵消的系统偏差。
  失败=zero-target 任一目标门失败或依赖 reference 才工作；不得更改 covariance/selection 找过门版本。
  本审查 validation 467-edge 算例已失败；本任务主要固化重现并补 train/source 分解。
Do NOT: 把零目标重新跑出的结果叫正确 likelihood；它只是诊断；读 sealed test 或新增 test bank。
Dependency: T00–T02。
Estimated compute class: 本地小 NPZ 短任务；大 source 拆分如超过约一分钟则 HTCondor job。
```

正结果下一步：仍需 T11/T12，因为 paired response 的噪声语义与 field model 尚未认证。负结果下一步：降级旧 closure claim，禁止旧 WLS 进入 data correction；只继续 T12 的 measurement-model 主线。失败不允许 V2 retraining。

### T11 / Experiment E-B — upstream covariance/transport 契约证伪

```text
Task ID: T11
Priority: P0
Scientific purpose: 判断 mode-0 极端各向异性源于合理 nuisance model，还是坐标/单位/实现失配。
Files to inspect: datasets/schema.py; datasets/propagation_loader.py;
  scripts/convert_ntuple_tracklets.py; scripts/convert_ntuple_tracklet_propagations.py;
  scripts/audit_propagation_uncertainty_budget.py; scripts/audit_propagation_pull_calibration.py;
  scripts/audit_field_global_fit_contract.py; configs/physical_curriculum_v3_expanded_trainval.yaml。
  外部依赖在 T00 pin 的 Calypso 内，按类定位 SegmentFitAlg、NtupleDumper、
  FaserActsExtrapolationTool 及其 state/covariance conversion；不能猜当前安装库等于源码。
Files likely to modify: 新 contract/adapter/audit；若证实 upstream 错误，在隔离 Calypso checkout 单独 patch，
  将 patch 和 build fingerprint 引用回本仓库，不能直接改官方安装或旧产物。
New files if needed: datasets/transport_contract.py; scripts/audit_transport_covariance_contract.py;
  configs/transport_covariance_contract_v1.yaml; tests/test_transport_covariance_contract.py。
Implementation: hypothesis=C_pred 在声明的 [x,y,tx,ty] 坐标中与实际传播 derivative/process noise 一致。
  先纯数学检查 local/bound/curvilinear/global chart、angular→slope、q/p 单位转换：
  C_out=F C_in F' + Q，Q 定义、reference surface 与各 state component 均显式。
  分 source-state covariance、transport F、Q、target C；保存维度/单位与主要 cross terms。
  Input=18 个既有非 sealed source，按文件固定顺序取每源前 20 个原事件（不足全取并报不足），
  所有 failures 留在 denominator；不按 residual/角度/rank 挑源。
  只需 nominal/current-condition export，不开多 DoF bank。每源 1 job，源内抽 5 条 eligible track
  做 local state FD h/2,h,2h；state perturb 尺度在 config 冻结，rotation 单位明确。
  若缺可用 exporter，先实现 generic schema，未经批准的条件写入始终禁止。
Tests: diagonal 与 non-diagonal covariance、非零 tx/ty、带符号 q/p、单位换算、
  Jacobian 与数值 propagation 对照；不能靠 branch 名含 jacob 判断 global fit 支持。
Expected artifact: outputs/transport_covariance_contract_v1/<source>/contract.json;
  transport_fd_summary.json; coordinate_closure.json; aggregate/decision.json。
Acceptance criteria: 数学 conversion relative covariance/F error ≤1e-6；物理 transport FD relative norm
  error ≤1e-3 且三步长呈收敛，接近零导数用已注册 absolute floor；过程噪声/坐标语义无缺项。
  任何源非有限/非 SPD/坐标不符均列 failure，不删除。真值 pull 仅作后验诊断，不能用于反调 F 或 Q。
  这些为新 contract 指标；旧 rank/capture 门保持原样。
Do NOT: 用 core-width rescale/新增 Q 超参数来掩盖失败；用 truth q/p 作部署 seed；
  为恢复漂亮 rank 改 covariance；对当前未 pin 的本地 Calypso 作科学断言。
Dependency: T00、T01、T03。
Estimated compute class: 数学测试本地；18 个 nominal export/propagation 检查为 HTCondor job。
  首次 1 CPU、4–6 GB/job，依据既有运行资源调整，不启动训练。
```

正结果下一步：明确哪些 covariance 是 measurement uncertainty、哪些是 propagation-seed nuisance，再进入 T12。负结果下一步：只修有独立解析/FD 证据的 upstream contract，独立 build 后重做同一 smoke；不能恢复契约则停止主线并保留 monitoring。不能用误差 rescale 参数搜索代替 repair。

### T12 / Experiment E-C — 正确 measurement-level information 是否改变结论

```text
Task ID: T12
Priority: P0
Scientific purpose: 区分 compression/weight/track-model limitation 与真正信息不足；这是主线资源门。
Files to inspect: models/field_route_fitter.py; models/track_fitter.py;
  alignment/true_cluster_local_residual.py; alignment/identifiable_subspace.py;
  alignment/tracker_only_identifiable_subspace.py::load_fd_only_bank;
  alignment/rigid_station_only_identifiability.py; alignment/hierarchical_v1_leakage.py;
  scripts/run_true_cluster_local_dump.py; scripts/dump_true_cluster_local_for_sample.py;
  configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml。
Files likely to modify: 新独立 likelihood/adapter；原 campaign 只接受 diagnostics-only reconciliation，
  不改其冻结结论或改造成新的 fit driver。
New files if needed: datasets/alignment_measurements.py; models/field_track_likelihood.py;
  alignment/profiled_information.py; scripts/run_alignment_information_pilot.py;
  configs/alignment_information_pilot_v1.yaml; tests/test_profiled_information.py;
  tests/test_field_track_likelihood.py。
Implementation: hypothesis=剖面化真实轨迹 nuisance 后，measurement model 可以给出与原 rank gate
  不同且可验证的绝对 alignment 精度；也允许比原方法更少的信息。
  Input A=既有 hierarchical V1 FD bank 全18源（只 reference+axial probes，不读 withheld target
  来定义 SVD）；Input B=T11 同一原事件规则的 cluster local-u/surface/track measurements。
  若现有 cluster dumper仅支持 selected-route real data，新增 generic MC/条件感知 adapter，
  不复用其 straight-line prediction 或自动构建默认行为。
  第一臂 truth-association，只用 label 固定测量归属；不把 truth q/p、trajectory 或 residual target给 fitter。
  track q 至少包含两位置、两方向、q/p，以及必要 scattering nuisance；各站真实材料/磁场。
  Measurement schema 必含 original_source_uid/run/event、physical_track/cluster/sensor IDs、
  local-u 与 measurement dimension、R、surface/frame/units、geometry/field/material/IOV hashes；
  derivative 输出区分 G=dh/dalpha、H=dh/dq、transport F 和 process-noise Q。
  G/H 在当前anchor计算，不从正确geometry的reference residual中取得；旧FD响应矩阵仅作对照。
  对每源前5条eligible track验证7个已声明global/nuisance方向的h/2,h,2h central difference：
  最多1+2×7×3=43个measurement-prediction evaluations/track，18×5×43=3870次上限。
  这是局部模型/传播求导检查，不是43个完整Athena physical-refit points；不新建7D bank。
  如导数确需额外full refit，在资源门之前单独列出缺失point/预计CPU，未经新计划不得自动提交。
  物理模型预先声明 IFT rigid 六分量与 C_dx 内部 nuisance，S1–S3参考的假设显式；
  只报告约束与协方差，不求一个能过门的参数子集，不写7D payload，不另开7D scan。
  建立 block normal equation 与 Schur complement；对同一测量集比较 fixed-q 与 profiled-q，
  检查后者信息不能无物理理由更大；给 dx/dy/Rx/Ry/Rz 的 nuisance-profiled covariance，
  对 dz/C_dx 的弱方向给 finite interval或 unconstrained，不强制零。
  对旧 A 保存 σ绝对值/比例、κ(A)、κ(N)、逐事件贡献和条件数；重新跑 T02 正确 bootstrap。
  新模型用现有可用 joint physical points作 development闭环验证，不用它们挑观测/参数/阈值。
Tests: 小 toy 的 full joint least squares 与 Schur 解/协方差 1e-8 相符；
  true gauge 已知解析null；共享轨迹相关性；nuisance projection不会虚增信息；
  单独加强一个模式不应在 absolute-precision报告中“消灭”其它模式；full-span projector平凡性。
Expected artifact: outputs/alignment_information_pilot_v1/<source>/normal_blocks.npz;
  absolute_precision.json; fd_linearity.json; oracle_free_closure.json;
  aggregate/legacy_gate_reconciliation.json; resource_decision.json。
Acceptance criteria: fixed definitions 下的解析/FD checks通过；所有18源保留，subset不足明确报；
  no-reference truth-association 在预注册 pooled/held-source目标上达到0.1mm/1mrad，
  结果对合法 chart change及h/2,h,2h稳定，95%区间及nuisance/prior依赖被明确报告；
  不能以每个8-event slope bin不满rank自动判结构无解，也不能以 pooled pass掩盖source bias。
  单个源有显著偏差、无界机械参数、FD失稳或covariance无定义则不准tracker-only correction；
  原 operational gate的失败始终原样报告。小pilot不认证coverage或部署。
  若仅因20事件/源导致有限但过宽区间，分类为sample_size_limited，不能写physics no-go。
  用已验证信息的每事件尺度预估达到预算所需N，同时明确独立性与coverage不变的假设；
  只有预估N落在T30预注册资源上限内、且identity/bias/FD契约无失败，才可判conditional_go。
  不能通过边看新结果边加N推进；结构null、模型bias、非法covariance与小样本分开报。
Do NOT: 删失败源、选有利slope、裁剪null后把误差记零；增加模型；把MC参考残差传给估计器；
  用局部straight line代替磁場fit；追加任意新topology寻找pass。
Dependency: T02、T03、T10、T11 全部已判定；T11契约必须通过。
Estimated compute class: toy/小矩阵本地；数据export、18源fit、LOSO和bootstrap为HTCondor job。
  1 source/job；bootstrap最初40次仅reconciliation（旧数量），新精度评估固定200次、20次/shard。
  默认总预算≤18源×20原事件起步；只能在预注册统计理由下进入T30的大样本，不能边看结果边扩容。
```

正结果：存在值得验证的 estimator，进入 T20；仅统计量受限的 `conditional_go` 必须带预先计算的事件预算，不能当成 precision pass。若某些方向只有 qualified survey 下可用：只允许 T20 的受约束版本，并明确 track-only 不可部署。若没有外部约束且 measurement-level 模型仍达不到预设误差/偏差要求、或所需统计量超出冻结资源上限：**终止当前预算下的无约束 tracker-only 主线**，转唯一 monitoring 备选；说明是物理、模型还是资源限制，不再增加网络、源类别或任意 DoF 删减。

## Stage 2 — Core algorithm

### T20 — 将小规模 global fit 做成受约束 estimator；survey 是测量因子

```text
Task ID: T20
Priority: P0
Scientific purpose: 用同一个物理likelihood处理geometry、track nuisance和外部信息，禁止先后估计互相吸收偏差。
Files to inspect: T12新增field_track_likelihood/profiled_information；
  alignment/survey_derived_prior_interface.py; alignment/cad_survey_nov22.py;
  alignment/nov22_metrology_provenance_station_ry.py;
  alignment/external_constraint_iov_feasibility.py; alignment/physical_jacobian.py;
  configs/physical_refit_5dof_survey_dz_iteration.yaml。
Files likely to modify: 新estimator和survey adapter；旧5mm prior仅保留historical控制。
New files if needed: alignment/constrained_global.py; alignment/survey_measurement.py;
  configs/constrained_global_alignment_v1.yaml; tests/test_constrained_global.py;
  tests/test_survey_measurement_contract.py。
Implementation: 用全局normal与local Schur消元，不按“先station再C_dx”固定未知物理量。
  明确estimand、gauge和reference-geometry nuisance；解析gauge用exact constraint/null basis，
  有限survey用(s-h_s(alpha))'Cs^-1(s-h_s(alpha))及当前anchor处的非零prior RHS。
  Survey schema要求sensor/rigid参数ID、原始测量来源、frame/pivot、full covariance、units、IOV、
  mechanical state、与其它constraints的相关/重复使用标记；未合格输入不可填 measured slot。
  输出track-only/data-information、prior contribution、posterior、nuisance correlation和bias预算。
  缺真实survey时仅做明确标注的hypothetical sensitivity，不能发physics pass或geometry proposal。
Tests: linearGaussian解析posterior；非零prior mean；correlated prior；exact gauge与finite prior区别；
  frame/covariance变换；weak mode不以零variance输出；重复survey使用应拒绝；IOV不匹配拒绝。
Expected artifact: constrained_fit.json; information_budget.json; survey_ingest_audit.json;
  estimator_contract.json（默认geometry_write_allowed=false）。
Acceptance criteria: 与全分块direct solve在1e-8相符；geometry uncertainty含nuisance；
  survey缺covariance/IOV时明确blocked_prior_unavailable；无prior分支只有T12所支持的范围可用。
Do NOT: 把Nov22 Sigma当测量error；硬写旧central values；silent pseudo-inverse把weak modes藏掉；
  用一个subspace sigma宣称所有native DoF达到precision。
Dependency: T12信息/estimator门通过，或T12明确指出仅外部约束可行且确有合格数据；T03。
Estimated compute class: 本地analytic/unit；source级复核HTCondor job，不做survey precision参数搜索。
```

### T21 — 以物理 route likelihood 处理 ambiguity，冻结网络

```text
Task ID: T21
Priority: P1
Scientific purpose: 检验wrong association如何影响alignment bias，并传播assignment的不确定性。
Files to inspect: baselines/route_assignment.py; models/field_route_fitter.py;
  training/route_aware_transformer.py; training/structured_assignment.py;
  evaluation/route_metrics.py; alignment/route_selected_update.py;
  scripts/run_frozen_association_backbone.py; T20 estimator。
Files likely to modify: 新route likelihood adapter/assignment层；现有solver仅必要接口扩展。
New files if needed: baselines/probabilistic_route_assignment.py;
  evaluation/alignment_association_bias.py; configs/assignment_uncertainty_v1.yaml;
  tests/test_probabilistic_route_assignment.py。
Implementation: 使用T12通过的fixed corpus/measurement model，比较truth-association、
  frozen V2 hard assignment、physics-likelihood hard assignment和有界soft control。
  hypothesis=hard selection的不确定性在weak modes上产生不可忽略偏差；不用AP决定胜负。
  先在最大8个ambiguous endpoints的小fixture/冻结事件上枚举feasible assignments得归一化真值；
  对更大图用预注册近似/候选截断并记录被丢概率质量或上界，无法界定则标approximation_unvalidated。
  unmatched已有，不重复实现无语义dustbin；missing模式/partial与complete的utility在同一模型定义。
  保留endpoint capacity，增加shared-cluster conflict审计；需要skip-route时仅连接已有真实长边，
  是否纳入预先写入新模型contract，不改变历史route图。
  Frozen V2只能作proposal/control或有明确定义的辅助项，不将同一residual信息重复乘入likelihood。
Tests: 对小图枚举归一化/容量；空selection/unmatched/partial/重复hit；
  selected route不以truth feature产生；交换等价节点不改变posterior；近似与精确枚举误差。
Expected artifact: association_bias_budget.json; truth_hard_soft_comparison.csv;
  posterior_assignment_checks.json; raw_to_used_denominators.json。
Acceptance criteria: primary=held-source alignment bias/uncertainty；soft比hard不能使冻结0.1mm/1mrad门
  任一主要参数更差；对小图归一化误差≤1e-8，近似数值门在config预先固定。
  如soft无益，保留较简单hard physics/frozen control，不启动新网络。
Do NOT: 用test或反复validation选temperature；用高purity代替bias；把edge概率相加当联合posterior。
Dependency: T20；T12 truth-association estimator已通过。
Estimated compute class: exact toy本地；每source/payload的frozen inference与比较HTCondor job。
  CPU物理fit；若frozen V2需要GPU，单独GPU inference shard；无training。
```

### T22 — 有边界的 association/fit/alignment 交替与真实 refit

```text
Task ID: T22
Priority: P0
Scientific purpose: 检验可观测likelihood驱动的更新能否在重新重建和重关联后真正收敛。
Files to inspect: scripts/prepare_multisource_multidof_iteration.py;
  scripts/submit_multisource_multidof_iteration_condor.py;
  scripts/run_physical_refit_capture_scan.py; scripts/assemble_multisource_multidof_iteration_manifest.py;
  scripts/write_station_alignment_payload.py; T20/T21。
Files likely to modify: 迭代orchestrator与新estimator之间的adapter；payload接口的绝对/增量语义。
New files if needed: alignment/iteration_policy.py; scripts/run_global_alignment_iteration.py;
  configs/global_alignment_iteration_v1.yaml; tests/test_global_iteration_state_machine.py。
Implementation: 从已验证anchor开始，E/association→field track fit→global M-step→
  shadow MC payload→真实cluster refit/Acts→重关联，迭代最多5次。
  每次重新计算geometry依赖的transport和measurement derivative；source UID/seed稳定。
  内部optimizer接受步用当前data objective/trust region，不用truth或validation挑step；
  FD trust bounds由T11/T12建立，固定damping候选[1,0.5,0.25]，失败返回reject而非继续缩到过门。
  diagnostics另报paired响应，但禁止作为优化RHS；缺target也能运行。
  第k次accepted step的result hash是第k+1输入之一；中断只能从finalized父节点恢复。
Tests: state transitions、错误parent hash、identity/noise-only无漂移、超trust region拒绝、
  objective升高拒绝、IOV/geometry错配拒绝、partial shard不能聚合写payload。
Expected artifact: iteration/<k>/<source>/manifest.json; iteration_decisions.json;
  trajectory_of_objective_parameters_assignment.csv; sandbox_payload_manifest.json。
Acceptance criteria: 连续两次accepted step满足normalized motion≤0.1个预注册工程容差，
  data objective不增加，独立predictive checks稳定；最终真实refit后bias门通过；
  参数不可识别/关联不稳定/5次未收敛均fail，不能仅凭linearized residual下降pass。
Do NOT: 更新official geometry；用truth选择damping或stopping；沿用旧refit/residual冒充新geometry。
Dependency: T20、T21；T03绝对/增量composition通过。
Estimated compute class: 每个source×iteration为HTCondor job；新FD如确有必要源内成组运行，
  聚合/接受步是DAG屏障；没有交互长refit。
```

## Stage 3 — Closed-loop simulation validation

### T30 — 预注册双轴泛化与完整闭环 simulation campaign

```text
Task ID: T30
Priority: P0
Scientific purpose: 验证known injection、unknown association、真实迭代与source/condition泛化，
  防止只在建立Jacobian的点或已选validation上“闭环”。
Files to inspect: configs/physical_curriculum_v3_expanded_trainval.yaml;
  configs/physical_curriculum_ift_hierarchical_v1_trainval.yaml;
  alignment/five_dof_sampling.py; scripts/build_physical_curriculum_corpus.py;
  scripts/submit_physical_curriculum_condor.py; scripts/materialize_pooled_curriculum_synthetics.py;
  datasets/pooled_physical.py; datasets/synthetic_overlay.py; T22。
Files likely to modify: 新campaign config与通用manifest字段；已有frozen配置不改。
New files if needed: configs/global_alignment_closed_loop_development_v1.yaml;
  scripts/submit_global_alignment_validation.py; evaluation/global_alignment_closure.py;
  tests/test_global_alignment_campaign_manifest.py。
Implementation: 在production前登记全部sources、payloads、seeds、selection、metrics、失败处理与预算。
  先复用10 train/8 development-validation源；本阶段结果仍叫development。
  固定6个joint条件：identity、两个sign-paired small injections、两个不同方向held-condition、
  一个线性边界stress；具体六向量在T12的有效envelope内一次性冻结，不按结果重抽。
  内部C_dx/reference geometry nuisance加入声明的stress，不通过冻结其真值来制造5DoF精度。
  train/development sources均跑已冻结条件；报告source-held-out和condition-held-out两张表，
  不把共用payload当condition独立。优先用现有physical点匹配，缺点才生成MC sandbox refit。
  每source最多100原事件作第一轮完整campaign：18×6=108初始source-condition单元，
  每单元最多5次迭代；实际新增点及全部estimated CPU小时必须在submit manifest列出。
  每个条件比较truth-association与冻结unknown-association；真实physical多track样本另报，
  overlay 3track/missing/background仅作为受控压力臂，不冒充原始collision事件。
Tests: source/condition/seed不可变、joint角rotation composition、UID隔离、failure源保留、
  所有body/config checksum、无test路径；scheduler只dry-run测试。
Expected artifact: preregistration.json; campaign_manifest.json; per_source_condition_closure.csv;
  iteration_summary.json; candidate_to_alignment_count_waterfall.csv; failure_catalog.json。
Acceptance criteria: 全预注册单元都有终态；无未报告丢源；known injection经真实refit最终bias门通过；
  truth-selected通过但unknown失败需归入association bias，不改模型后仍称同一次确认；
  identity不生成统计显著假correction；stress失败允许，但明确标定failure envelope。
Do NOT: 在train验证后立刻开旧sealed test；生成新的test bank以追求过门；
  删除failed jobs/sources；按test结果选择模型；重新抽有利的heldcondition。
Dependency: T22小闭环通过；T00–T21 contract全部finalized。
Estimated compute class: HTCondor job；1 source×condition（或source内少量邻近points）/job，
  iterative dependency由DAG控制；GPU仅frozen inference，bootstrap分离。
```

正结果：进入 T31。负结果：根据预注册归因在 estimator contract 或 assignment uncertainty 发现具体 defect 才修；修后是新版本 development study，旧失败保留。不得自动扩大 source/seed/模型家族。如果满足需要不存在的外部约束才能闭环，停止到 prior availability gate。

### T31 — Statistical uncertainty、failure envelope 与一次性 confirmatory 计划

```text
Task ID: T31
Priority: P0
Scientific purpose: 把“有几次成功”提升为带coverage和明确适用条件的结论。
Files to inspect: evaluation/global_alignment_closure.py（T30）；alignment/resampling.py（T02）；
  evaluation/route_metrics.py; alignment/capture_criteria.py;
  scripts/register_5dof_capture_criteria.py；T30所有终态结果与T00 exposure registry。
Files likely to modify: 新统计报告/confirmatory登记器；旧capture thresholds不变。
New files if needed: evaluation/uncertainty_coverage.py; scripts/report_global_alignment_validation.py;
  configs/global_alignment_confirmatory_protocol_v1.yaml;
  tests/test_uncertainty_coverage.py。
Implementation: development阶段固定200个event-cluster bootstrap reps（20/shard），
  source-level LOSO逐源；对重采样每次重做association/fit/align，或明确近似只重做哪部分。
  bootstrap confidence不等于重复实验coverage：coverage验证需要模拟pseudoexperiments
  或独立data生成，单独预注册最多200个、每个固定seed，不把同一overlay重复当独立truth实验。
  评价68/95% interval coverage的binomial区间；mode truncation、prior-dominated方向单独列。
  nuisance/field/material/occupancy/angle/condition shift的stress值由独立系统误差预算预先固定；
  主导假设失败就停止，不能通过增加“更合理”的stress cuts淘汰难源。
  最后仅生成confirmatory计划：全项目exposure ledger、未看过的source+condition来源、
  checkpoint/estimator/config hash、一次性开封人/流程与failure publication规则。
  真正采集/生成/开封需另行获准并独立冻结；历史sealed来源永久不复用。
Tests: coverage分母包括失败fits；unbounded interval与fit failure不记成功；
  bootstrap multiplicity、seed恢复、source/event两层变异、absence of test access。
Expected artifact: uncertainty_coverage_report.json; failure_envelope.csv;
  confirmatory_protocol.json; development_vs_confirmatory_exposure.json。
Acceptance criteria: 校准的68/95% nominal levels在相应binomial 95%区间内，且coverage无明显
  source/parameter系统下降；区间宽到无实用精度也不能pass；bias满足冻结预算。
  coverage未建立时标unverified；confirmatory结果若尚未获得，deployment gate仍false。
Do NOT: 用bootstrap自证模型正确；根据新的confirmatory结果回调hyperparameters；
  把development validation改名test；无授权生成新test bank。
Dependency: T30全部终态且主门通过；T02正确重采样。
Estimated compute class: HTCondor job，bootstrap/pseudoexperiment按20 seeds/shard；
  需要反复physical refit的stress每source×scenario一个job；aggregate小任务。
```

## Stage 4 — Real-data deployment

本阶段并非“Stage 3 结束就可写 geometry”。当前 master 没有满足下面的前提。真实数据可以继续输出 selected-route counts、prediction/residual distribution、异常与适用范围；没有独立验证时只能 monitoring。能够推断的是已通过模型/约束/coverage验证的**相对 geometry estimands**，不能声称绝对装置位置、未知 dz、弱模或共同内部形变由 tracks 单独测量。

### T40 — 独立 IOV 的 shadow 验证与 DQ 效能

```text
Task ID: T40
Priority: P0
Scientific purpose: 检验simulation支持的方法在真实run/IOV上不把模型漂移拟合成geometry。
Files to inspect: alignment/real_data_operating_protocol.py;
  alignment/real_data_residual_dq_monitoring.py;
  alignment/real_data_residual_dq_monitoring_expansion.py;
  alignment/real_data_station_mode_failure_audit.py;
  alignment/survey_metrology_iov_infrastructure.py;
  configs/operating_protocol_v1_real_data_sources.yaml;
  configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml; T20–T31。
Files likely to modify: 新shadow report与DQ比较层；旧monitoring scale/threshold不改。
New files if needed: alignment/real_data_shadow_validation.py;
  configs/real_data_shadow_alignment_v2.yaml; scripts/report_real_data_shadow_alignment.py;
  tests/test_real_data_shadow_validation.py。
Implementation: 先写所需real-data estimands与physics impact预算，明确reference和survey依赖。
  使用residual-blind预注册run/IOV集合，含独立holdout runs；已经反复研究的14973/14974
  只作development/calibration参考，不能重新称独立确认。
  old official geometry与new sandbox shadow并行重建；明确authorized shadow IOV，
  不修改official tag。数据估计与验证事件按原事件分离，验证绝不以residual重新挑window。
  检查charge/angle/momentum/occupancy/时间依赖，prediction pull、track fit质量、
  内外站leave-out预测、matching效率/DQ代理、已有alignment方法或独立survey cross-check。
  同一survey用于prior后不能用于独立cross-check；需要另留外部测量或独立track observable。
  保留旧DQ robust-z as effect size，新增已知模拟drift/稳定run的power与false-alarm评估，
  新alarm policy需另行冻结，不能悄悄改变V1。
Tests: truth字段缺失仍运行；IOV/geometry/survey不匹配阻止solve；
  monitoring不得生成payload；零残差、低chi2、route数变多均不能单独解锁write。
Expected artifact: real_data_shadow_report.json; physical_impact_budget.json;
  iov_transfer_matrix.csv; independent_validation_report.json; dq_power_false_alarm.json。
Acceptance criteria: confirmatory simulation已合格；real-data heldout predictive checks和
  独立physical参考一致、无未解释run/charge/momentum偏差；survey各ingest gate通过；
  总systematic预算满足预注册physics requirement；否则保持residual_dq_monitoring_only。
Do NOT: 用self-nulling无reference的低residual直接写conditions；把monitoring z当高斯显著性；
  自动联系metrology负责人；把无IOV的历史survey用于当前data。
Dependency: T31真正的独立confirmatory结论完成；T20所需外部约束真实可用。
Estimated compute class: read-only report本地；current-vs-shadow两臂reconstruction为HTCondor job，
  1 run×segment×arm/job；real-data shadow需要事先明确授权与资源计划。
```

### T41 — Geometry publish gate、原子候选包与 rollback

```text
Task ID: T41
Priority: P0
Scientific purpose: 把科学通过与official conditions写入分开，确保可追踪、可拒绝、可回滚。
Files to inspect: scripts/write_station_alignment_payload.py;
  alignment/real_data_operating_protocol.py;
  alignment/operating_protocol_v1_final_closure.py; T22 state machine；T40最终报告。
Files likely to modify: 只新增strict candidate gate与explicit publisher入口；
  不让generic report或estimator持有official write能力。
New files if needed: alignment/deployment_gate.py; scripts/build_alignment_candidate_package.py;
  scripts/validate_alignment_candidate_package.py; tests/test_alignment_deployment_gate.py;
  docs/ALIGNMENT_DEPLOYMENT_RUNBOOK.md。
Implementation: 构建一个可review的候选包：旧/new full SE(3)、diff、reference/IOV、
  data/prior information、covariance/systematics、all-source validation、checksums、
  exact runtime、签署者/批准记录槽、previous tag/hash和rollback命令计划。
  validation工具只读；writer必须看到所有required gates true和明确授权的目标IOV。
  official发布在单独受控入口，成功后验证readback、tag/IOV、monitoring；
  失败/告警恢复原tag或停用新IOV，恢复不覆盖旧conditions历史。
Tests: 缺一gate/错parent/hash/过期IOV/prior不足/uncertainty未校准均拒绝；
  只能生成dry-run candidate；模拟rollback恢复原完整transform；没有测试触碰真实COOL/POOL。
Expected artifact: alignment_candidate_package/；deployment_gate.json；rollback_manifest.json；
  runbook与一次sandbox readback/rollback记录。
Acceptance criteria: 下述最终deployment gate全部通过；candidate已可审阅；
  official写入仍需明确负责人授权；未授权状态是ready_for_review，不是published。
Do NOT: 自动改official conditions；以--diagnostic-override绕过科研门；覆盖旧tag/negative report；
  把用户批准文档路线解释为批准官方geometry发布。
Dependency: T40通过；T00–T31 evidence完整；独立部署授权。
Estimated compute class: 本地candidate验证；readback/reconstruction校验HTCondor job；
  official发布是受控短事务，不在普通worker中执行。
```

## HTCondor 实施规范

对于 physical refit、dataset production、parameter/state FD scan、bootstrap、大 validation campaign，必须写入 `Estimated compute class: HTCondor job` 并按下述规范实施。没有“先交互试跑完整 campaign”步骤。

| 任务类型 | 推荐 job granularity | 关键产物 | 聚合/恢复 |
| --- | --- | --- | --- |
| nominal export/covariance contract | 1 original source，固定20事件；源内成组state probes | source manifest、成功/失败计数、state/cov summary、runtime hash | 单进程聚合18源；只重试失败source |
| 物理 geometry FD | 1 source×anchor；reference/±probe组共享明确身份规则 | 每point独立payload/refit/audit；common与missing event计数 | expected point清单全齐才建立Jacobian |
| frozen inference/global fit | 1 source×condition | route likelihood、normal blocks、selected/raw counts | CPU/GPU分开；不共享可写checkpoint |
| bootstrap/pseudoexperiment | 固定20 seeds/shard | multiplicity、每replicate结果、invalid理由 | 按seed完整集合聚合，不能只选成功replicates |
| iterative closure | 1 source×condition×iteration | parent hash、current geometry、fit result、proposed step | DAG屏障；全源齐后决定accepted step |
| data validation | 1 run×segment×old/shadow arm | provenance/IOV、DQ/fit summary | old/shadow按原event对齐，单独失败报告 |

统一路径建议：

```text
outputs/<campaign_id>/<immutable_run_id>/
  preregistration.json
  resolved_config.yaml
  runtime_manifest.json
  shards/<source_or_run>/<condition>/<iteration>/<attempt>/
    input_manifest.json
    stdout.log
    stderr.log
    completion.json 或 failure.json
    artifacts.sha256
  aggregate/<aggregation_version>/
    completeness.json
    results.json
    decision.json
```

manifest 列出 expected shards、source/condition split、seed 派生规则、原始 input GUID/checksum、代码与库 hash、requested CPU/memory/disk、timeout、重试上限、actual events、输出哈希、exit code、failure phase。seed 必须由稳定字符串哈希派生，不能使用 Python 随机化 `hash()`。EOS 上 finalize 采用同文件系统临时文件、完整校验后发布 success marker；若底层 atomic rename 语义不保证，改用不可变对象 + 单独完成标志，不声称事务性。

job scratch 只处理自己的 shard；worker 不写 shared corpus manifest。聚合必须验证父运行 hash 和全部终态，区分 `FAILED`、`INSUFFICIENT`、`MISSING` 与 `PASS`。失败最多按预注册 transient retry 上限重试，physics/contract failure 不靠不断重试消除。所有尝试日志保留，不能删除失败 source 或失败 attempt。内存不足可以以新资源记录重试同配置，不改变 selection。

## 最终 deployment gate

只有以下条件同时成立，T41 才能生成 `ready_for_review=true`；本审查时全部部署资格仍为 false。

1. Baseline、upstream binary、state/covariance/geometry chart、field/material 和 IOV 可恢复且数值验证通过。
2. 目标 estimands、true gauges、weak modes、内部与reference nuisance明确；没有以fixed=known的假设隐藏误差。
3. 推断完全不读取同事件truth/reference-geometry residual，identity与nonzero injection的真实闭环通过。
4. assignment容量、unmatched/shared hits、uncertainty和selection dependence被计入；没有AP升高替代bias验证。
5. 预注册source/condition/seed与全部failure结果齐全；独立confirmatory结果通过，development不伪装test。
6. 参数/预测误差、interval coverage、系统误差和failure envelope满足独立制定的physics影响预算。
7. 若模型需survey：measurement covariance、frame/pivot、相关性、IOV及机械稳定性证据真实可用；不与validation重复使用。
8. 独立真实run/IOV的shadow预测、DQ、外部或已有alignment方法cross-check一致，无未解释bias。
9. 新payload与旧tag完整可审阅、readback/sandbox rollback通过，publisher与普通research worker隔离。
10. 获得针对具体候选包、目标IOV和official条件写入的明确授权。批准研究路线、MC sandbox实验或shadow重建均不替代该授权。

若任一 gate 不满足：**保持当前 official geometry，输出 monitoring/diagnostic result 和缺失证据；不发布 alignment correction。**
