# 真实数据 Residual/协方差模型有效性与非线性响应前置审计 V1

Workbook 79。**状态：预注册** —— 所有 gate 在计算任何 audit 量之前冻结；执行后回填结果并单独冻结。

本战役是 **model adequacy 审计**，不是 geometry correction，也不做 nonlinear alignment。整个 Workbook 79：

- `geometry_write_allowed = false`
- `official_conditions_write_allowed = false`
- `real_data_candidate_alignment_authorized = false`
- `external_constraint_ingest_authorized = false`
- `held_out_accessed = false`（held-out 继续完全关闭）

## 科学问题（唯一）

WB78 的超大 candidate（|γ|max = 2.46，为冻结线性包络 0.15 的 16 倍）究竟是因为：

- **H1** — 真实 geometry 确实远离 nominal（residual/covariance 统计模型本身充分）；还是
- **H2** — covariance/statistical-model mismatch：combined 4×4 covariance 在真实 collision population 上的相关结构不适用，近奇异 eigen-direction 给 residual 赋予过大权重，从而产生巨大 χ² 与虚假的巨大 γ；还是
- **H3** — observable/Jacobian-transfer mismatch：station-pair mean MC Jacobian 对真实 collision topology 的 transfer 误差足以显著偏置 inferred alignment direction？

在回答这个问题以前，禁止启动 real-data nonlinear alignment。

## 冻结输入（不重新选 population）

- **Calibration**：run 14973 + 14974，以及被冻结的 route/event/source identity 与 split SHA `c9c359793c41b59df1296e3ec2075af652c3f585746ca832a9c2e37f4a4f61d3`。不得增加 run、删 event、重选 route。
- **Held-out**：14975/14976 + 7 个 monitoring run + 14977 report-only，继续完全关闭。Workbook 79 不得读取 held-out residual。
- 所有 model adequacy 研究只能使用 MC/control + 已打开的 calibration 14973/14974。
- Tracker information / gauge / transfer 合同完全继承 WB78（加载并 SHA 验证，不复制以防漂移）。

## 预注册设计

### Stage 0 — 精确复现 WB78 baseline（硬停止）

在任何 audit 量之前，用 WB78 同一确定性链重建 calibration bank + one-shot solve，逐项验证冻结 baseline（rtol/atol = 1e-9；bank SHA 精确匹配）：bank SHA256 `6e4eaae0...`、χ²_zero = 1999613.6054909483（378 pairs）、informed rank 3、information eigenvalues (8.04e-5, 1.70, 68.5, 117.3, 2608.0)、restricted condition 38.06、γ = (−2.460, +0.857, −0.403)、bootstrap 17/50 rank 改变、158.133°。任何不一致 → `real_data_model_adequacy_inconclusive`，立即停止。

### Stage 1 — covariance eigenstructure / whitening audit（H2 主线）

WB78 已观察：零 candidate χ² ≈ 2.00e6（1512 dof，χ²/dof ≈ 1322），但对角 marginal pull 均 < 0.1——强烈提示 χ² 由 combined covariance 的近奇异相关方向主导。

对每个 pair 的 combined 4×4 covariance 做完全只读审计：eigenvalues、condition number、correlation matrix、smallest-eigenvalue direction、residual 在 covariance eigenvector 上的投影、按 eigenmode 的 Mahalanobis 贡献、marginal pull、按 pair type、按 run。

**Whitening 检验（主 gate）**：z = L⁻¹(r − m_cell)，m_cell 为 per-(run, station-pair-type) coherent mean（diagnostic-only 的 alignment-bias 代理，绝不作为新 residual 定义或 alignment input）。若 frozen covariance 正确描述 coherent mean 之外的涨落，whitened χ²/ndof = O(1)。

- **预注册 gate（对模型宽容）**：`whitened_chi2_resid / ndof ≤ 4.0`，ndof 扣除估计的 cell mean。
- 报告（非 gate）：完整 eigenstructure、correlation、χ² 按 covariance eigenmode 集中度（尤其 smallest eigenmode 占比）、marginal pull、condition 分布、经验 Cov(z) 特征值/condition。

### Stage 2 — empirical covariance cross-check（14973 ↔ 14974）

建立 diagnostic empirical residual covariance（**不进入 alignment solve**）。按 station-pair type 在 (0,1)、(0,2)（统计量足够）上比较 14973 与 14974 的经验协方差结构；(0,3) 每 run 仅 ≤1 pair，report-only。

- **主 metric（基无关，对近简并谱稳健）**：两 run 经验协方差的 generalized-eigenvalue RMS log-deviation。**预注册 gate：≤ ln(3)（factor-3）**。
- 报告（非 gate）：principal eigenvector 夹角、逐 eigenvalue log-ratio、marginal scale、correlation 符号、empirical vs propagated 对比。
- 为避免同数据自证：14973→predict/check 14974，及 14974→predict/check 14973。
- 若 empirical 与 propagated covariance 严重不一致 → 冻结 `real_data_covariance_model_not_adequate`，且必须另开独立预注册 covariance-model campaign，**不得**在 Workbook 79 内直接换 empirical W 再求 candidate。

### Stage 3 — alignment score contribution decomposition（report-only）

把 frozen transfer model 的 normal equations 分解：每个 informed direction γ_k 的 score b = JᵀWr，拆成 run / station-pair / x,y,tx,ty / covariance eigenmode 贡献。报告 cumulative contribution（top 1%/5%/10%/50% pairs 对 |b_k| 的贡献比例），回答 γ=−2.46 是"很多 pair coherent 同方向推动"还是"极少数近奇异 covariance direction 放大"。**禁止删除贡献最大的 pair。**

### Stage 4 — bootstrap instability decomposition（report-only + diagnostic counterfactual）

WB78：50 次 bootstrap、17/50 rank 改变、最大方向角 158°。分解：rank 分布、rank cut 附近 eigenvalue 分布、mode persistence、哪个 pair type 控制不稳定 mode、不稳定是否由 covariance weight 驱动（用 diagonal-covariance diagnostic counterfactual 对比）。目的不是找更稳定的 W，而是判断 instability 来自 coverage/statistics 还是 covariance correlation model。

### Stage 5 — diagnostic-only covariance counterfactuals（A/B/C/D）

明确标注 `diagnostic_only=true, alignment_authorized=false`：

- **A** `full_frozen`：official baseline（= WB78）。
- **B** `diagonal_marginal`：相同 marginal variance，off-diagonal = 0。
- **C** `condition_capped`：eigenvalue floor 在 λ_max/1e3。
- **D** `unit_weight`：4 个 observable 等权。

只比较：candidate direction、magnitude、rank、bootstrap stability、χ² concentration。B/C/D 绝不提升为新 alignment model、不产生可部署 candidate、不因 γ 变小就宣布修复。若结论对 covariance treatment 极端敏感，这是模型失配证据，不是选"最好 W"的授权。

### Stage 6 — observable/Jacobian-transfer support audit（H3）

不新增真实数据 FD。检查 mean-J transfer 的误差结构：

- **(a) MC per-pair J dispersion（冻结 MC 属性，gated）**：每个 station pair 内 per-pair J 相对 mean-J 的 Frobenius 相对偏差。**预注册 gate：RMS 相对偏差 ≤ 0.35**（与 WB78 注入恢复 gate 同量级）。
- **(b) real-vs-MC kinematic support overlap（residual-blind，gated）**：在 track-slope 平面 (pred_tx, pred_ty) 上按 station pair 比较真实 calibration pair 与 MC bank 支持云。real pair 的 robust Mahalanobis² 落在 MC 99% 等值线（χ²₂(0.99)=9.210）内记为 within support。**预注册 gate：(0,1) 与 (0,2) 各要求 ≥90% real pair within support**；(0,3) 仅 2 pair，report-only。
- 不按 residual 挑轨迹；不用 nearest-neighbour J 替换 mean J。

### Stage 7 — calibration cross-run transportability（report-only solve，gated 比较）

用同一 frozen model 分别对 14973-only、14974-only 求 informed score / direction / one-shot diagnostic amplitude（**不是**新 correction，不开 held-out）。比较 rank、informed subspace projector Frobenius distance（report-only）、γ direction、γ magnitude、predicted observable direction。

- **预注册 gate（继承 WB68/77 15° 合同）**：两 run rank 相等；dominant informed eigenvector 夹角 ≤ 15°；γ-hat 方向夹角 ≤ 15°。

### 决策树（预注册，恰好一个终止字符串）

硬 gate（WB78 baseline 复现、数据有限性、bank/CSV 逐位复现）失败 → `real_data_model_adequacy_inconclusive`。

否则统计 {covariance_adequacy(H2)、transfer_support(H3)、cross_run_transportability} 中的软失败数：

- 0 个失败 → `real_data_model_adequacy_pass_nonlinear_response_preregistration_allowed`
- 1 个失败 → 对应专串：covariance → `real_data_covariance_model_not_adequate`；transfer → `real_data_jacobian_transfer_support_not_adequate`；cross-run → `real_data_calibration_information_not_cross_run_stable`
- ≥2 个失败 → `real_data_statistical_model_multiple_failures`

若任一关键项失败：不开 nonlinear geometry campaign，继续 `residual_dq_monitoring_only`。只有全部通过才允许另开 Workbook 80。

## 关于未来 nonlinear response（本战役禁止偷跑）

Workbook 79 首先只做 model adequacy。只有当本战役证明：(1) covariance/statistical model 没有足以解释超大 γ 的严重失配；(2) transfer support 可接受；(3) calibration cross-run direction 稳定；(4) 剩余证据更支持 genuine nonlinear response——才允许下一条独立 Workbook 80 预注册 `Calibration-Only Nonlinear / Trust-Region Reconstruction Response Feasibility V1`。Workbook 80 才能讨论 temporary non-official geometry probes，且必须：仍只用 calibration 14973/14974、held-out 继续关闭、temporary/non-official geometry only、不写 COOL/POOL、预冻结 trust radius/probe grid/stop rule、不以 held-out improvement 调 iteration、每步完整 reconstruction、先验证 response curve（如 γ = 0, ±0.05, ±0.10, ±0.15 的 full-chain residual response 是否单调、局部光滑、与线性 prediction 一致），probe 数值从已有 frozen support 合同继承，不得因 γ=2.46 就把 probe 扩到 ±2.5。

## 工程规范

- 新增：`configs/real_data_residual_covariance_model_adequacy_v1.yaml`、`alignment/real_data_covariance_model_adequacy.py`、`scripts/report_real_data_covariance_model_adequacy.py`、`tests/test_real_data_covariance_model_adequacy.py`、双语 docs、`outputs/real_data_residual_covariance_model_adequacy_v1/`。
- 无需新 reconstruction：读取既有 real-data outputs + MC reference propagations + 线性代数，交互完成。
- Driver 阶段结构性强制 freeze ordering：`validate-config → reproduce → covariance-audit → score-decomposition → bootstrap-decomposition → counterfactuals → transfer-support → cross-run → decide`；所有 audit 阶段要求 Stage-0 baseline 复现通过；`decide` 读取冻结的 audit artifact。
- 所有 diagnostic counterfactual 在 JSON 中带 `diagnostic_only=true, alignment_authorized=false`。
- 回归测试覆盖：held-out 路径不可访问、冻结 calibration population、no event dropping、counterfactual W 不能进入 production solver、empirical covariance 不能被自动提升为 W、no geometry write、no external prior、deterministic contribution decomposition、14973/14974 独立报告、WB78 baseline χ²/candidate score 精确复现、决策树全部分支。

## 结果

（执行后回填）
