# 多自由度全局 Alignment 闭环

## 范围

这是 FASER 全局 association 与 alignment 的物理控制链，不新增 Transformer 架构。
association backbone 保持冻结，当前可用的严格封存控制为 V2 BCE route-query。
永久封存的 test bank 不会被读取。

第一批激活的是 IFT/station-0 的 `dx`、`dy`、`Ry`，下游 S1--S3 固定为参考坐标系。
Calypso payload 为 `[dx, dy, dz, Rx, Ry, Rz]`，单位为 mm/rad；报告中转动使用 mrad。

## 物理约束

每个点都独立执行：

```text
/Tracker/Align SQLite/POOL payload
  -> SCT_ClusterContainer
  -> SegmentFitRefit
  -> SegmentsRefit
  -> NtupleDumper
  -> FaserActsExtrapolationTool (mode 0)
```

不允许坐标平移替代、residual-level 注入、缓存 propagation 或 local `q/p` 替代。
local segment `q/p` 仍只是不可依赖的 seed，因此物理 V1 一律使用 mode 0。

## 可重复的 Iteration

生成 source-disjoint 的 anchor/probe bank：

```bash
source scripts/setup_environment.sh ml
python scripts/prepare_multisource_multidof_iteration.py \
  --source-config configs/physical_curriculum_v3_expanded_trainval.yaml \
  --iteration-template configs/physical_refit_multidof_smoke_mc24_100043.yaml \
  --output-root outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --iteration 0 \
  --current ift_dx_mm:2.0 --current ift_dy_mm:-1.5 --current ift_ry_mrad:35.0 \
  --nevents 100
```

该 bank 含 10 个 train 与 8 个 validation 原始 xAOD file。每个 source 有 8 个独立
refit 的真实物理点：reference、anchor，以及三个激活参数的正负 probe。每 source 一个 job：

```bash
python scripts/submit_multisource_multidof_iteration_condor.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --submit-dir outputs/condor_mc24_multidof_ift_iteration00_anchor_trainval_physical_v1 \
  --schedd-mode eossubmit --submit
```

全部点通过 completion check 后，聚合真实有限差分：

```bash
python scripts/run_multisource_refit_multidof_local_step.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --anchor-point iteration_00_anchor --target-point iteration_00_reference \
  --fit-split train --held-out-split validation \
  --capture-tolerance ift_dx_mm:0.1 \
  --capture-tolerance ift_dy_mm:0.1 \
  --capture-tolerance ift_ry_mrad:1.0 --require-full-rank \
  --output-dir outputs/mc24_multidof_ift_iteration00_anchor_trainval_closure_v1
```

train 只用于确定 update；validation 只评估该冻结 update，并输出独立 fit 诊断。
结果包括每个点的 raw candidate truth-chain retention、有限差分曲率、rank、无量纲
condition number、covariance/correlation，以及 source/station-pair 稳定性。只有
`capture_success=true` 才可推进下一轮。

## Association 闭环

完成的 bank 可以只读地交给既有 pooled synthetic 工具：

```bash
python scripts/assemble_multisource_multidof_iteration_manifest.py \
  --iteration-manifest outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/iteration_manifest.json \
  --materialization-config configs/physical_alignment_iteration_trainval.yaml \
  --output outputs/mc24_multidof_ift_iteration00_anchor_trainval_physical_v1/physical_corpus_manifest.json
```

`alignment_iteration_shared_across_payloads` 只共享确定性的 overlay 选择，以便跨真实
payload 求 selected-route provenance 的交集。真实 tracklet state、covariance 与 Acts 输出
仍各自对应 payload。冻结 V2 通过 `--payload-id` 对每个物理点单独推理，再把 selected 的
精确 mode-0 Acts edge 交给 `run_route_selected_multidof_update.py`。其中 straight-line fit
只可作诊断，不能作为磁场下的 alignment objective。

`audit_field_global_fit_contract.py` 已审计当前 propagation product：tree 具有全部
pairwise residual/covariance 字段，但没有导出的 transport Jacobian 或 source-state
transition representation。因此当前 field-aware update 必须准确称为“route consistency +
WLS”，不能称为独立全局 likelihood。真正的 field-aware global fitter 要等 Calypso 导出
该 Jacobian，或提供可验证的共同状态 Acts repropagation API 后再实现。

用已验证 update 的 `--update-json` 生成下一轮真实 payload bank。该选项默认拒绝未通过
closure 的 update，诊断性 override 必须显式指定。

## 自由度准入

payload/Jacobian 代码已支持完整 station rigid components，但 `dz`、`Rx`、`Rz` 仍未激活。
只有真实有限差分 bank 同时给出 full rank、可接受的 scaled condition/correlation、稳定的
source 与 station-pair response 和 held-out physical closure 时，才加入一个新分量。station、
layer、module 层级均已预留；不得把 layer/module conditions 静默映射成 station payload。

## 当前状态

此前 10-event joint anchor 测试虽然 rank=3，但 `dx` 不稳定，因此不用于更新几何。
iteration-0 多 source 物理 bank 已提交 Condor；在每个真实 refit 与 mode-0 Acts export
通过 completion gate 前，不做物理结论。
