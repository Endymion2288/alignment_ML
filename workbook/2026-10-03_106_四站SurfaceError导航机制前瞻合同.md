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

## 7. 本轮授权与可观测性补充（执行前）

用户再次授权“git commit，并继续推进下一阶段”。当前 HEAD `9d46a2f`，只有七个既有未跟踪 core。本轮授权覆盖隔离的 index12 instrumentation replay，supersede 第5节旧“不授权重跑失败event”；不覆盖其它event、修复、阈值、WB103、qualification或ML。先提交补充合同，再实现/测试/提交/冻结，最后单Condor。

FACT FROM REPOSITORY：ACTS32.0.2 `include/Acts/Surfaces/SurfaceError.hpp` 明确定义 `GlobalPositionNotOnSurface=1`。Calypso `FaserActsExtrapolationTool.cxx` target overload 返回 `optional<BoundTrackParameters>`，打印 `result.error()` 后返回nullopt；该接口不暴露失败终态、navigator boundary或逐step field count。因此这些量保持UNKNOWN，不能从nullopt证明失配发生在target还是某个中间surface，也不能把lookup成功等同geometry有效。

本次仅重放原WB92调用顺序直到原fail-closed停止；其原有Hxi/Htheta perturbations不是新的WB103四lambda矩阵。隔离source只添加flush的逐调用input/output/exception记录和sensor lookup记录，记录station、axis、label、seed/frame/start/direction以及成功时target-local residual；不添加新的传播、field query、step observer或数学控制参数。synthetic target plane没有detector geometry ID，sensor lookup和target plane必须分开表述。

预注册证据等级：完整trace/source/library/payload/event身份且同一失败重现，可报告 `LOCALIZED_OFFICIAL_SURFACE_FAILURE`（描述性事实）；target/中间surface因果归属、field-query异常和navigation机制仍按第3节 UNKNOWN，除非实际记录直接支持。不以重放成功复现替代根因分类PASS。若未复现或instrumentation失败，保留全部产物，不改参数、不retry。

独占输出 `outputs/mc24_four_station_wb106_surface_trace_v1`；worker先计算唯一允许raw xAOD全文件SHA/size并绑定原fixture stat，再build/run；这不是声称WB104事前做过raw hash。WB104未计算full raw hash的协议偏差保留，WB105缺乏独立negative controls/逐step证据的限制也保留。worker单start，新的terminal receipt不得覆盖WB104 receipt。

## 8. 实现与执行前检查

`scripts/wb106_contract.py`从冻结WB92 v3源进行唯一marker转换：新WB106命名空间、逐call seed/frame/label、官方调用前后、成功target-local返回、sensor lookup及终止receipt。`m_tool->propagate`原调用和数学/step参数不变；official接口nullopt继续fail-closed。新的`scripts/audit_wb106_trace.py`独立重建WB92调用前缀、扰动/left-SE(3)目标面、源header、start方向/distance和成功local返回；核验生成源/诊断二进制、官方库/实际fieldmap哈希。synthetic plane geometry ID为0，不作真实detector sensor的替代。

LCG_110_cuda下`tests/test_wb106_trace.py`九项PASS，包含partial failure无未来sensor receipt的合法性及source/header/frame/qop/missingcall/geometryID/station mutations拒绝；源码检查确认官方propagate调用不改、无stepTolerance修改。Python编译、shell语法检查通过。测试是diagnostic gate证据，不是物理执行证据。

合同快照单独保存/hash，最终workbook追加不会冒称原freeze未变。worker优先一键Calypso环境；初始计划单job 1CPU/8000MB、workday，EOS包装的实际资源/transfer另记，不把请求视为实际。尚未产生任何新的物理结果。

## 9. 冻结与单任务提交

实现commit `ce3a7b5`；v1 freeze绑定108 identities及合同快照，fixture仍只index12。提交前用户queue0、idle slots4513，继续bigbird24，cluster `1202338`。首次submit helper未提交：空queue时`condor_q -json`输出空串，JSON解析失败；后用`condor_q -totals`核对0 jobs、同一冻结worker/参数手动等效提交，`submission.json`明确记录该接口偏差。不改冻结源码或input。

实际scheduler ad请求被EOS包装为3CPU/9000MB/ShouldTransferFilesYES，区别于sub的1CPU/8000MB/NO。初始JobStatus1、NumJobStarts0；不是terminal PASS。目前execution/scientific classification均UNKNOWN，未开始WB103或读新population。

## 10. 完成结果与独立终态审计

FACT FROM REPOSITORY：`outputs/mc24_four_station_wb106_surface_trace_v1/scheduler_history.json`记录bigbird24/1202338.0：JobStatus4、ExitCode0、ExitBySignalfalse、NumJobStarts1、RemoteWallClockTime603秒，worker为b9g34p3570。`worker_exit.json`为0，完成UTC2026-10-03T09:25:04；`athena_exit.json`为1，这是预期的原official传播fail-closed，不是worker基础设施失败。stderr为空，build/source/binary receipts齐全。

`summary.json`及独立`postrun_audit.json`：108冻结身份重验、离线summary重建一致；实际复制sqlite/catalog字节与冻结payload相同，catalog指向的历史POOL PFN哈希一致；末记录FAIL_CLOSED和原错误一致。execution/integrity gate PASS仅覆盖这些显式检查。sensor geometry ID没有独立从另一重建输出重建，field query/status和内部navigator状态未记录，不能宣称全部physical conditions正确或完整因果审计PASS。

逐调用trace457行，125inputs、24boundary、101official calls、100成功output、1失败、1terminal。首个失败为call124、station3、axis−1、nominal：seed `[81.40511863538732,-1.3131074015600313,0.040472236225215015,0.006055748391783332,1e-5]`，seed z=`-1860.1500000000005 mm`，target identity rotation、z=`2427.4 mm`、synthetic geometry ID0。actual header仍100044:2268，raw source/ordinal见fixture与`raw_identity.json`。不是rotation perturbation失败，也不是未开始的WB103 Jacobian。

**描述性结论 `LOCALIZED_OFFICIAL_SURFACE_FAILURE`；科学机制分类 `UNKNOWN`。** 官方tool日志SurfaceError:1且返回nullopt；没有证据定位到requested target的最终bound conversion或中间surface。历史WB104/WB105失败、WB86 FAIL、WB103停止均保持；statistical qualification / physics screening / final oracle均NOT_EVALUATED。

## 11. 唯一下一阶段建议

RECOMMENDATION：WB107只检验“失败是否来自ACTS最终target bound-state conversion”，增加隔离EigenStepper的被动boundState输入/返回观测，逐调用与未改official tool对照。允许仍只index12的原调用前缀；100个成功返回必须数值完全一致、同一nominal失败必须复现，否则机制UNKNOWN。不调整target、容差、field、q/p、step policy或geometry；不能由最终转换失败推断导航的更深根因。先提交本结果，再冻结WB107前瞻合同/实现，单Condor。不得扩展WB103、删掉index12、做formal qualification或ML。
