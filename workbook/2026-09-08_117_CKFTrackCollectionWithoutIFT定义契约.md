# Workbook 117: CKFTrackCollectionWithoutIFT 定义契约（Yasu 第 0 阶段）

日期：2026-09-08
状态：**完成 / PASS** —— 从 Calypso 源码、ROOT metadata 和 20 事件重建 dump 锁定官方 `CKFTrackCollectionWithoutIFT`。未新建 source campaign。未把 LTO 或四站 `CKFTrackCollection` 当作该 collection。未用 SegmentFit dummy `q/p`。未用 truth `q/p`。未进 3ST→IFT 传播、alignment、B14M、B15、Measurement Model V2。未覆盖 master 既有产物。

**最终判定：`PASS`**

- `decision = ckf_without_ift_definition_established`
- `mechanism = null`
- `three_st_to_ift_chain_authorized = true`（只授权下一步小样本链路，本 workbook **没有**做传播）
- `lto_is_not_this_collection = true`
- `four_station_ckf_is_not_this_collection = true`
- `measurement_model_v2_entered = false`
- `geometry_write_allowed = false`
- `held_out_accessed = false`
- `b14m_reopen_authorized = false`
- `b15_authorized = false`

## 问题

Yasu 机制研究的第一步必须先回答：官方 collection `CKFTrackCollectionWithoutIFT` 在 Calypso 里究竟是什么，它是否真的只用 S1/S2/S3，IFT measurement 有没有进 fit，以及落盘 native state / 5×5 的坐标、单位、电荷符号和交叉项是什么。不确定时禁止猜测。

## 输入

- 同一 WB87 construction / validation 源划分中的两个 source-disjoint 文件（小样本）
  - construction：`mc24_100043_00400_00499`（µ⁻ particle gun）
  - validation：`mc24_100048_00000_00049`（µ⁺ particle gun）
- 冻结 WB87 / WB95 / WB96 / WB109 决策（只读 master `alignment_ML/outputs`）
- Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`，Athena 24.0.41，ACTS 32.0.2
- 节点：`lxplus909.cern.ch`；测试用 `scripts/setup_environment.sh ml`（LCG_110_cuda）；dump 用 `calypso`

## 源码契约（钉死，不是猜 API 名）

`faser_reco.py` 在 forward 打开时**总会**写入三站 collection，与 `--noIFT` 是否关闭四站 CKF 无关：

```
CKF2Cfg(..., maskedLayers=[0, 1, 2, 3, 4, 5], name="CKF_woIFT",
        actsOutputTag="{filestem}_3station_forward",
        OutputCollection="CKFTrackCollectionWithoutIFT",
        BackwardPropagation=False)
```

`CircleFitTrackSeedTool` 的 wafer 编码是 `6 * station + 2 * layer + side`。  
`SiDetectorElement`：

| 方法 | 条件 |
| --- | --- |
| `isInterface()` | `station == 0`（IFT） |
| `isUpstream()` | `station == 1`（S1） |
| `isCentral()` | `station == 2`（S2） |
| `isDownstream()` | `station == 3`（S3） |

因此 `maskedLayers = [0..5]` 就是 IFT 3 层 × 2 side 的全部 wafer。这些 cluster 在进入 CKF 之前就不写入 sourceLinks / measurements。`CreateTrkTrackTool` 只把 CKF sourceLink 写成 `ClusterOnTrack`。随后的 `KalmanFitterTool.fit` 只读 `measurementsOnTrack()`，不能把 IFT 加回来。

`CircleFitTrackSeedTool.m_removeIFT` 默认是 `false`。本 collection 的 IFT 排除靠 `maskedLayers`，不靠该旗标。

这**不是** WB109 LTO：LTO 从四站 `CKFTrackCollection` 出发，轮流排除 station 1/2/3，**保留 IFT**。  
这**也不是**四站 `CKFTrackCollection`。WB107 已记录四站 refit 把 IFT（`z < -100 mm`）标成 outlier，因此 `measurementsOnTrack()` 会漏掉 station 0；机器检查必须同时看全部 TSOS。

`CreateTrkTrackTool::ConvertActsTrackParameterToATLAS` 把 ACTS `q/p`（`1/GeV`）乘 `1_MeV` 写成 Athena `Trk::CurvilinearParameters`（`1/MeV`）；电荷是转换后 `q/p` 的符号；5×5 的 q/p 行/列同样缩放。

## ROOT metadata

uproot 读 `CollectionTree` 键（不反序列化 EDM）：

| 源 | 事件数 | `CKFTrackCollectionWithoutIFT` | `CKFTrackCollection` | 反向 / cluster / SegmentFit / Truth |
| --- | ---: | --- | --- | --- |
| construction | 500000 | 有 | 有 | 均有 |
| validation | 250000 | 有 | 有 | 均有 |

`s0013-r0022` MC xAOD 同时持久化了四站和官方三站 collection。这与 `faser_reco.py` 的 `TrackCollection#CKFTrackCollection*` 输出通配符一致。

## 重建 dump（每源 20 事件，login Athena）

站号由 `detStore` 的 `FaserSCT_ID` 解码。`isInterface()` 与 `station == 0` 的不一致数为 0。

| 划分 | 事件 | 含 IFT cluster 的事件 | WithoutIFT 径迹 | MOT 上 IFT | TSOS 上 IFT | 完整 5×5 / SPD | dummy / truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| construction | 20 | **20** | 20 | **0** | **0** | 20 / 20 | 0 / 0 |
| validation | 20 | **20** | 19 | **0** | **0** | 19 / 19 | 0 / 0 |

同一事件的四站对照：

| 划分 | 四站径迹 | MOT 上 IFT | TSOS 上 IFT |
| --- | ---: | ---: | ---: |
| construction | 20 | 0 | **18** |
| validation | 20 | 2 | **17** |

四站 TSOS 上能看到 IFT，说明这些事件里 IFT hit 存在且被四站链使用（多为 outlier）。WithoutIFT 的 MOT **和** TSOS 都是零 IFT。排除检验是有信息的，不是“事件里根本没有 IFT cluster”。

WithoutIFT 的 `front()` 在 S1 附近（例：`z ≈ 21.4 mm` / `17.4 mm`）。四站 `front()` 在 IFT 附近（例：`z ≈ -1886 mm` / `-1890 mm`）。两个 collection 的参考面不同，后续 3ST→IFT 传播必须从三站 native state 出发，不能拿四站 `front()` 冒充。

原生类型：`CurvilinearParametersT<5,Trk::Charged,Trk::PlaneSurface>`。  
q/p 单位 `1/MeV`，带符号，电荷符号与 `q/p` 符号一致。全部 5×5 都有非零 q/p 交叉项。无一等于 SegmentFit dummy（`1e-5 /MeV`，`5e-6 /MeV²`）。

q/p（`1/MeV`）：

- construction（µ⁻）：min `-7.84e-5`，median `-3.43e-6`，max `-5.71e-7`；电荷全为 −1
- validation（µ⁺）：min `-2.09e-6`，median `+7.61e-7`，max `+3.95e-4`；电荷 +1 有 17，−1 有 2

validation 有 1 个事件没有写出 WithoutIFT 径迹（20 事件 / 19 径迹）；17/19 条有 station 3。这记录为 candidate/selection 差异，不是本阶段的删除理由。本阶段不声称 q/p 已校准、无偏或可作 alignment 先验。

## 禁止项（均保持）

不用 LTO 顶替官方 WithoutIFT；不用四站 CKF 顶替；不用 dummy / truth q/p；不 rescale 协方差；不新建 source campaign；不写 geometry；不打开 B14M / B15 / Measurement Model V2；不读 sealed test；不覆盖 master 产物。WB87 transport covariance 仍未验证。

## 结论

官方 `CKFTrackCollectionWithoutIFT` 是 `CKF_woIFT` 写出的 **S1+S2+S3 三站 CKF**。在本小样本上，IFT measurement 没有出现在该 collection 的 fit / MOT / TSOS 里。native 状态是带符号 `q/p`（`1/MeV`）的 5 维 CurvilinearParameters，并带完整 5×5 交叉项。

**尚未回答** Yasu 的科学问题：三站 `q/p` 是否有偏、是否与 IFT `R_y`/`d_x` 形成弱模。本阶段只锁定对象。

## 工程产物

- 配置：`configs/ckf_without_ift_definition_v1.yaml`
- 模块：`datasets/ckf_without_ift_definition.py`
- Athena dump：`scripts/dump_ckf_without_ift_definition.py`
- 审计：`scripts/audit_ckf_without_ift_definition.py`
- 冒烟：`scripts/run_ckf_without_ift_definition_smoke.sh`
- 测试：`tests/test_ckf_without_ift_definition.py`（6 passed）
- 文档：`docs/ckf_without_ift_definition.md`、`docs/ckf_without_ift_definition_cn.md`
- 预注册：`docs/yasu_3st_qp_ift_weak_coupling.md`、`docs/yasu_3st_qp_ift_weak_coupling_cn.md`

分支：`yasuCheck`  
HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

Config SHA：`952646a6e8642b1cfadb880e6309706fba14bbc8a96273f91fa90495f8a8c8cd`

Run ID：`yasu_s0_without_ift_definition_20260908T125042Z_8559fecc`

产物（EOS，不入 git）：`outputs/ckf_without_ift_definition_v1/`

| 产物 | SHA256 |
| --- | --- |
| `ckf_without_ift_definition_contract.json` | `6eb3d9afaa3cbfc0cc46af8fa75ba254e71dd5da0198f8363ab69804fdc47ec1` |
| `calypso_source_contract.json` | `ab697ddf9cf7d0696376de53f100ee42c2ba01230934905df58cf4ff04af5edc` |
| `source_inventory.json` | `d771a42f23de4384cc28829f703f91c46dc5f71f23aa8edb308f6e9c2a717c22` |
| `state_definition.json` | `89582faf9d42a4a3756ee2a1965e8623d7d1aab8ceb353f5f5f35655fded52be` |
| `pinned_calypso_sources.json` | `1c514bc331532c2814e2b326fd92dd688cbcca730da43e55e235b47d04706328` |
| `inherited_stage.json` | `420249beb53f50113aab7ba99993e3a4c45939417c93bb0e75d6067fbb5bdd14` |
| `COMPLETE.json` | `a86752657ebef2431201856e837b0cd0011820bbefa3d336f1706ce8072f9e93` |
| construction tracks jsonl | `ff64c1a8e8e931e18566ea20bdaf30dfd40c081184f9702754a283413c106a2f` |
| construction events jsonl | `1baaecdaa76a6625d62fa7ee54898287033a37d7daf2a0a51a748ebf3ec99b02` |
| validation tracks jsonl | `10ea3a68cfbe4adac7c3f68b7fdbafe3ec15f8b5670f144653148a2b82432b98` |
| validation events jsonl | `c817775d1f5f81ff27e7c3d705005c12f3b5d177a863572876f43e1032b9b169` |

geometry_hash：`4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`  
field_hash：`60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`  
conditions_hash：`d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`

## 下一步

```
WB117 / Yasu-S0:
  官方 WithoutIFT = S1+S2+S3 CKF
  IFT 未进该 collection 的 fit
  native 5D + 5×5 已物化
        ↓
Yasu-S1: 独立 3ST→IFT 预测链（小样本）
  WithoutIFT native (loc1,loc2,phi,theta,q/p)+5×5
  → ACTS field-aware transport 到 IFT
  → 独立 IFT measurement residual
  IFT measurement 不得泄漏进 prediction
        ↓
再做 source-disjoint MC 上的 q/p 符号/尺度/不确定度校准
```

不得用四站 Cin、WB109 LTO Cin 或 truth q/p 作为该预测的输入。不得一开始就提交大规模 HTCondor。
