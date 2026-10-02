# Workbook 106 — 四站 `SurfaceError` 导航机制前瞻合同

## 0. 问题与状态

WB104 多事件 fixture 前置验证在 index 12（source `100044_00300_00399`、ordinal `2268`、actual event `100044:2268`）的官方 common-seed propagation 处返回 `SurfaceError:1`。WB105 已确认日志身份正确、错误不是静默 fallback，并停止了 WB103。WB106 只回答一个问题：**index 12 的 SurfaceError 是由冻结 target-plane/ACTS surface 身份或导航交点条件触发，还是由 field/conditions/worker 执行路径触发？**

状态：前瞻合同，结果 UNKNOWN；不访问新 event，不运行 Jacobian，不修改生产 backend/geometry/solver。

## 1. 冻结输入

只读：

- `outputs/mc24_four_station_wb92_wb104_event_fixture_preflight_v2/events/12/fixture.json`；
- `events/12/athena.log`、worker stderr/command、`preflight_summary.json`；
- WB92 v3 freeze、source/build/library/field/sqlite hashes；
- WB92 `CommonSeedAudit.cxx` 的现有 target-plane、sensor lookup 和 official propagation 代码。

不读取其它 ROOT event，不替换 index 12 的 seed/q/p/target frame，不用 indices 1/8 的成功结果代替它。indices 4/16/20 的保存非线性结果只作为离线负向回归样本，不进入新物理执行。

## 2. 诊断实现边界

若后续授权实现，必须在新隔离 `wb106_*` component 中增加只读 diagnostic hooks，复用同一 FASERNU-04、OFLCOND-FASER-06、冻结 sqlite、field map/cache、ACTS/FaserActs libraries 和 `material=false/covariance=false` 配置。允许记录：

- seed surface z、每个 target frame 的 z/rotation、target station 与 geometry ID；
- target-plane local coordinates、propagation direction、start/target distance、navigator boundary 和 surface lookup status；
- field query count、nonfinite state、step/abort status 和 exact `Acts::Result` error；
- official target-plane propagation 与 actual sensitive-surface lookup 的身份对照。

禁止改变 propagation tolerance、step policy、direction, q/p unit、target frame、surface transform、field backend 或错误处理为“继续”。错误必须 fail-closed 并保存完整 receipt。

## 3. 预注册判据

`IDENTITY_OR_SURFACE_GEOMETRY`：target frame/geometry ID、z 或 local direction 在输入和 ACTS surface lookup 间不一致，或 local intersection 明确无效；

`OFFICIAL_NAVIGATION_FAILURE`：所有 identity/conditions/field checks 一致，target surface lookup 有效，但官方 propagation 仍返回 `SurfaceError`/empty parameters；

`FIELD_OR_CONDITIONS_FAILURE`：field map/cache/sqlite/global-tag/library hash 或 field query 状态不一致；

`WORKER_OR_IO_FAILURE`：Athena/worker/ROOT/IO 错误在 propagation 前发生；

`UNKNOWN`：证据不足或诊断 hook 自身完整性失败。

只有完整 identity/conditions receipt 和 independent scalar/log audit PASS 才能给出分类；不能从 `SurfaceError` 字符串单独推断原因。

## 4. 负对照

诊断必须离线验证：故意替换 target frame z、station geometry ID、source/header、sqlite 路径和 q/p unit 时，identity gate 能拒绝；故意将 target frame 复制到另一个 station 时，surface mismatch 能拒绝。负对照不触发真实 event 重跑。

WB105 indices 4/16/20 的 Hxi Taylor/step-halving 失败必须保持为已知负向样本；WB106 不得把它们改判为 SurfaceError，也不得借此放宽 WB92 contraction `0.7`。

## 5. 结果与后续门

WB106 `PASS` 只表示 index 12 失败机制分类证据完整。若分类为 `IDENTITY_OR_SURFACE_GEOMETRY` 或 `FIELD_OR_CONDITIONS_FAILURE`，先建立修复后的新物理合同和独立 event fixture，再评议 WB103；若分类为 `OFFICIAL_NAVIGATION_FAILURE`，WB103 多事件扩展仍停止，需新的导航/交点机制研究；若 `UNKNOWN`，不得扩大数据、调阈值或开启 ML。

WB106 不授权：重跑失败 event、改变 WB92 数值阈值、删掉失败 event、复制 WB102 fixture、进入 WB93–WB101、实现 WB103 Jacobian、访问 held-out/sealed、formal qualification 或 ML training。

## 6. 合同结论

**`FROZEN_FOR_REVIEW; RESULT UNKNOWN`。** 本合同将唯一剩余物理执行阻塞点限定为 index 12 的 SurfaceError 机制分类，保留全部 WB104/WB105 失败证据和事件身份。
