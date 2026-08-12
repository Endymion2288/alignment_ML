# FASER Tracklet Alignment ML

本目录从可复现的多站 local tracklet 关联基线开始。MC 导出的 ROOT 输入契约、真实
refit/Acts 链路和纯几何 chi-square 基线完成验证后，才训练 Geometry-Aware Sparse
Transformer V1；其第一次封存 test 及未扩大 capture range 的负结果也已保留。

canonical 输入为名为 `tracklets` 的 flat ROOT tree，每行对应一条 local
tracklet。它要求显式的 station ID、global `(x, y, z, tx, ty)`、状态
`[x, y, tx, ty]` 的协方差、拟合质量、hit 摘要，以及监督训练所需的 MC
truth 标签。当前 FASER PHYS NTuple 可用于审计，但尚不满足这一契约。

## 快速开始

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/bootstrap_ml_environment.sh  # 每个 LCG Python 环境只需一次
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML
python -m scripts.make_synthetic_tracklets --output data/synthetic_tracklets.root --seed 7
python -m scripts.run_chi2_baseline \
  --input data/synthetic_tracklets.root \
  --config configs/baseline_chi2.yaml \
  --output-dir outputs/synthetic_chi2
pytest -q
```

baseline 会保存解析后的配置、metrics JSON、match CSV 和 candidate chi2
图。MC metrics 会区分全部 prediction 与可由非重复 truth 标签评分的子集。
Calypso 必须先编译，再在独立 shell 中执行
`source .../setup_environment.sh calypso`。bootstrap 会把本项目及其声明的 Python
依赖装入 LCG user site；之后的新 shell 只需执行上面的 `source` 命令。

对完整的 MC 导出、转换、内容审计和 baseline 运行，可执行：

```bash
cd /eos/home-x/xcheng/FASER
alignment_ML/scripts/export_mc_tracklets.sh \
  --input INPUT-xAOD.root \
  --output-dir alignment_ML/outputs/EXPERIMENT_NAME \
  --nevents 10 \
  --source-station SOURCE --target-station TARGET \
  --chi2-gate GATE_FROM_AUDIT
```

station ID 必须从 `content_audit.json` 选择，不能从 z 推断。省略后三个
baseline override 参数即可使用 YAML 配置。脚本会调用
`faser_ntuple_maker.py --export-tracklets`、写出 event-wise enhanced ROOT、
转换为 canonical 数据，并拒绝覆盖已有 artifact。只有直接对 MC 调用 converter
时才使用 `--include-truth`；具体契约和当前验证边界见下方链接。

## 当前四站控制样本

MC24 100 GeV FASERnu muon 是当前 IFT+1+2+3 控制样本。项目现已保留可选 q/p 字段、
转换 `FaserActsExtrapolationTool` 的 field-aware pair record、验证固定 truth 的
residual/pull/chi-square，并完成 station-level x/y closure。V4 truth 审计表明
reconstructed local q/p 不能作为 V1 local measurement；mode 1 因而只是 MC truth-q/p
propagation control，而非可部署输入。Geometry-Aware Sparse Transformer V1 现在只消费
既有 mode-0 physical candidate graph，并复用既有 route assignment 后端。它的
validation-selected、封存多方向 test 尚未证明 capture range 扩大；解释 alignment 结论前
必须先阅读 V1 专门文档。

仅 validation 的 Geometry-Aware Transformer V2 route-aware 研究同样没有通过固定 primary
point。当前 route context 回退到与 edge-only control 完全相同的解，而 raw physical truth chain
仍然可用。V1 test source 继续封存，也没有生成新的 final test bank；机制诊断和可复现 validation
contract 见 V2 文档。

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_tracklet_qoverp.py \
  outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/qoverp_audit_rerun.json

python scripts/evaluate_field_propagation.py \
  --tracklets outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --propagations outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/propagations.root \
  --q-over-p-mode 0 --min-truth-match-fraction 0.99 \
  --output-dir outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/field_propagation_mode0_rerun
```

MC22 electron 继续作为 1--3 station 的 exporter/covariance/loader smoke test。对于 MC24
四站控制样本，真实 Calypso `/Tracker/Align` payload 现在驱动从持久化
`SCT_ClusterContainer` 经由 `SegmentFitRefit`、`SegmentsRefit`、tracklet exporter 与
`FaserActsExtrapolationTool` 的 refit。station 3 `+1 mm` physical closure 在 27 条
truth-matched pair 上以 `4.9e-13 mm` 的最大误差恢复注入偏移。已完成的 25-point、三方向
physical capture scan 会为每个 payload point 重跑该链路。在严格 `0.01 mm` recovery criterion
下，全部 `0.1 mm` trials capture，而全部 `1 mm` trials 失败；这是特定配置下的 control 结果，
不是通用的探测器 tolerance。解释历史 coordinate-level scan 前必须阅读 physical-capture 文档和
保存的 point-level diagnostics。

## 文档

- [输入 schema 与 exporter 契约（英文）](docs/tracklet_export_contract.md)
- [输入 schema 与 exporter 契约（中文）](docs/tracklet_export_contract_cn.md)
- [当前数据审计（英文）](docs/data_audit.md)
- [当前数据审计（中文）](docs/data_audit_cn.md)
- [baseline 验证（英文）](docs/baseline_validation.md)
- [baseline 验证（中文）](docs/baseline_validation_cn.md)
- [含磁场 propagation 验证（英文）](docs/field_aware_propagation.md)
- [含磁场 propagation 验证（中文）](docs/field_aware_propagation_cn.md)
- [固定 truth alignment closure（英文）](docs/alignment_closure.md)
- [固定 truth alignment closure（中文）](docs/alignment_closure_cn.md)
- [conditions payload 与 coordinate-level closure（英文）](docs/condition_payload_alignment.md)
- [conditions payload 与 coordinate-level closure（中文）](docs/condition_payload_alignment_cn.md)
- [偏移几何 Segment refit（英文）](docs/displaced_geometry_refit.md)
- [偏移几何 Segment refit（中文）](docs/displaced_geometry_refit_cn.md)
- [物理 refit capture-range scan（英文）](docs/physical_capture_scan.md)
- [物理 refit capture-range scan（中文）](docs/physical_capture_scan_cn.md)
- [synthetic multi-track overlay（英文）](docs/synthetic_multitrack.md)
- [synthetic multi-track overlay（中文）](docs/synthetic_multitrack_cn.md)
- [真实 payload synthetic unknown-association baseline（英文）](docs/synthetic_unknown_association.md)
- [真实 payload synthetic unknown-association baseline（中文）](docs/synthetic_unknown_association_cn.md)
- [物理错位增强 curriculum MLP baseline（英文）](docs/curriculum_mlp_baseline.md)
- [物理错位增强 curriculum MLP baseline（中文）](docs/curriculum_mlp_baseline_cn.md)
- [pairwise MLP 与全局指派 baseline（英文）](docs/global_assignment_mlp_baseline.md)
- [pairwise MLP 与全局指派 baseline（中文）](docs/global_assignment_mlp_baseline_cn.md)
- [Geometry-Aware Sparse Transformer V1（英文）](docs/geometry_aware_transformer_v1.md)
- [Geometry-Aware Sparse Transformer V1（中文）](docs/geometry_aware_transformer_v1_cn.md)
- [Geometry-Aware Transformer V2 机制诊断（英文）](docs/geometry_aware_transformer_v2_diagnostics.md)
- [Geometry-Aware Transformer V2 机制诊断（中文）](docs/geometry_aware_transformer_v2_diagnostics_cn.md)
- [Geometry-Aware Transformer V2 route-aware 验证研究（英文）](docs/geometry_aware_transformer_v2.md)
- [Geometry-Aware Transformer V2 route-aware 验证研究（中文）](docs/geometry_aware_transformer_v2_cn.md)
