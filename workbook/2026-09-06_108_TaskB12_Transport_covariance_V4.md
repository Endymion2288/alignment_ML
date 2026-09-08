# Workbook 108: Task B12 Transport Covariance V4 证伪

日期：2026-09-06
状态：**完成** —— 在 B10 / B11 正式判定之后做 V4，不是重跑旧 V3。预注册四臂，冻结 WB103 输入与 WB81/WB87/WB98 闸门。未删 100043/37，未调 C/Q，未做 PSD 投影，未编造 `Cov(pred,target)`，未改闸门，未进 Measurement Model V2。未改 WB96–WB107 结论。

**最终判定：`NOT_ESTABLISHED`**

- `decision = leave_target_out_prediction_contract_not_established`
- `primary_case = leave_target_out_arms_unavailable`
- `secondary_cases = official_cin_is_global_kf_refit_front_state, full_track_c0_and_c1_shape_still_fail`
- `measurement_model_v2_authorized = false`
- `lto_arms_evaluated = false`
- `shared_measurement_leakage_isolated = false`
- `cin_semantics_isolated = false`
- `material_on_isolated = false`

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

B10：LTO 契约未建立。  
B11：官方 Cin 是全局 KF refit 的 `front()` 状态，不是独立传播种子。

## 四臂

| 臂 | 状态 | 协方差 | 本 run |
| --- | --- | --- | --- |
| A | full-track CKF | C0 | 评估（冻结 WB104 指标，不重开 V3 战役） |
| B | full-track CKF | C1 | 评估（同上） |
| C | leave-target-out | C0 | **封锁** |
| D | leave-target-out | C1 | **封锁** |

construction (0,1)：

| 臂 | N | χ² 均值 | χ² 中位 | pencil | λ | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| A C0 | 370 | 7.38 | 0.048 | 22.40 | 0.036–0.197 | 失败 |
| B C1 | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | 失败 |

传播前 Cin（WB105）：λ = 0.027 / 0.054 / 0.153 / 11.35，pencil 6.80。
`100043/37` 仍占混合 χ² 的 80.3%，保留。

V4 的判别不是“哪条臂数字最好”。没有 C/D，就不能把共享测量泄漏与 Cin 语义拆开，也不能判断 material-on 在正确预测契约下是否物理合理。不得因此改闸门或进 V2。

## 分类

| token | 本 run |
| --- | --- |
| `transport_covariance_validated` | 否 |
| `ckf_covariance_contract_repair` | 否（无 LTO） |
| `acts_material_process_noise_diagnosis` | 否（无 LTO） |
| `mixed_or_inconclusive` | 不选用 |
| `leave_target_out_prediction_contract_not_established` | **是** |

## 工程记录

Config SHA：`42f4d89904be71ecd8fb7c9a9808769bcc6ebbe69b1e7d0f72f5443a88d52575`

WB107 decision SHA：`d65200270fab5a3c796b03c2b5c67b3280ad6afdea569518650d90ddee3aade6`  
WB106 decision SHA：`f8e2140f0c7f057ba8b8c616695bf003b7681625ee5daf75d1b0af23b0fedf67`  
WB105 decision SHA：`e29fd93b0dbc9f435934184b93ce550d7dc2e3236177ea47502129c6ec2890f2`  
WB104 decision SHA：`8e620b50a3b5509d39e66e6dd7e1fd7a1a7aed5f53fdf63f53ec2456da266031`  
WB103 contract SHA：`e8c1f927e1ba03d418c2ed9b03084b198d09ba15196297a1a13cb5044ec48e3d`

Run ID：`sbb12_transport_covariance_v4_20260906T212151Z_2112e1b4`

| 产物 | SHA256 |
| --- | --- |
| `transport_covariance_v4_arms.json` | `f0c92408c2a1fea061766bb28f499bc51a5add4853d0f0bb6bbaf38b4d271f9e` |
| `transport_covariance_v4_decision.json` | `666859d4a032a93c80f50b36dab8f67267e246b138efe333dc2aa9bb58445deb` |
| `inherited_stage.json` | `a3ae5980c140d80b2d9589ebce8435337d6eaccb6f7d1cbb14ee5092737305f1` |
| `COMPLETE.json` | `ccd30678dd457b38e54baf5e0db0439f86cd4e50a00d3f36a71d093bfa9f9cae` |

测试 5 passed。未重跑 dump，未开 HTCondor。

Provenance 与 WB98 冻结一致：

- geometry `4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`
- field `60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`
- material_map `0df37bd621e6caac2e223fda12fec32a6f8c49685c7b831181b956e59996433c`
- conditions `d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

## 下一步

主线仍是：

```
WB105 mixed
  → B10 LTO 契约未建立
  → B11 Cin = 全局 KF refit front()
  → B12 四臂证伪：C/D 封锁
  → 独立 leave-target-out helper（不得包装 KalmanFitterTool.fit）
  → 重做 V4
  → 只有 V4 PASS 才进入 Measurement Model V2
```

不得从 material map、Q scale 或 reconstruction selection 横向搜索。
