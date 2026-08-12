# Geometry-Aware Transformer V2：机制诊断与 route-aware 改造

## 本轮边界

- V1 test source 永久封存；本轮不读取 test manifest、test ROOT 或 test output。
- 诊断 loader 只允许 validation；V2 训练 loader 只允许 train/validation。
- 不生成新 test bank，不改 physical refit、Acts propagation、candidate graph 或 route solver。
- mode 0 保持为物理传播模式；local q/p 不作为可靠 measurement。

## 已完成的 V1 validation-only 机制诊断

输出目录：

    outputs/geometry_aware_transformer_v2_mechanism_diagnostics_validation_v1/

contract 记录：

    loaded_splits = [validation]
    forbidden_splits = [test]
    test_events_loaded = false
    test_artifacts_opened = false

validation source 仅为 `mc24_100116_00010_00019` 和
`mc24_100117_00010_00019`。物理图有 1,440 event、18,669 node、175,084 条有向 message
edge、44,921 条相邻输出 edge、10,511 条 truth-positive edge。

完成的 probe：depth 0/1/2/3/4，adjacent-only，forward-only，backward-only，以及四个
leave-one-station-out。每个 probe 复用冻结 V1 checkpoint、validation Platt calibration、
threshold（0->1=0.001、1->2=0.5、2->3=0.05）和 dustbin penalty=1.0，不进行重训、
calibration 或 threshold selection。

关键现象：

- depth-0 相对 full-depth 的 mean absolute score drift=0.0135，rank correlation=0.99937；
  说明 local residual decoder 几乎主导 edge 排名。
- 0->1/1->2/2->3 的 edge AP 从 depth-0 的
  0.8433/0.8657/0.8962 只变为 full-depth 的 0.8405/0.8666/0.8983。
- station 内平均 cosine similarity 从 0.319 升到 0.549，state norm 从 8.01 升到 21.25；
  有明显 representation mixing/over-smoothing 迹象。
- 5 mm threshold truth-chain recall 对 local/full 都约 0.825，故失败不是 raw candidate
  被 message 过滤。10 mm full-depth：eff=0.573、purity=0.913、fake=0.129；false route
  同时有 fake endpoint 和 mixed-truth chain。
- forward/backward/leave-station probe 会改变 efficiency，但没有同时满足 purity/fake
  约束。full multi-station message 未显示出可用的 route-level 竞争机制。

因此 V2 不再加深或放大普通 Transformer，而是把完整四站 route 作为显式训练对象。

## 已实现的 V2 代码

- `models/route_transformer.py`
  - 固定 V1 的 d_model=128、4 block、8 head、FFN 128->256->128 backbone；
  - 只对已有 0->1、1->2、2->3 physical edge 连成的完整 route 建 route query；
  - 聚合四个 node state、三条 physical edge feature、station-pair embedding 和 base edge logit；
  - route correction 回投到已有相邻 edge，route assignment backend 不变。
- `training/route_aware_transformer.py`
  - route truth-consistency、one-to-one endpoint competition、fake-route penalty；
  - truth/synthetic role 不进网络，只做 label 与 loss audit；
  - 每个 train/validation graph 的 route table 一次物理构造后缓存，避免每 epoch 重复 Python
    枚举；缓存只包含已加载 split 的 candidate 索引。
- `scripts/train_route_aware_transformer_v2.py`
  - 显式 `allowed_splits=(train, validation)`；
  - validation-only early stop、Platt calibration、route threshold/dustbin selection；
  - same-checkpoint base-edge control 复用 V2 calibration 与 threshold，不允许独立调参。
- `configs/geometry_aware_transformer_v2.yaml`
  - 独立 config，无 test path 或 test-output reference。
- `scripts/audit_route_aware_candidate_bank.py`
  - 审计完整 physical route candidate 占用；只加载 train/validation。

新增/扩展单元测试通过：

    13 passed

覆盖 split 访问边界、V1 trace、完整 route 枚举、route correction 的零初始化、loss 反传和
V2 checkpoint round-trip。

## Route candidate occupancy audit

输出：

    outputs/geometry_aware_transformer_v2_route_candidate_bank_train_validation_audit.json

审计只加载 train/validation，并记录 `test_events_loaded=false`。完整 physical route 数为：

| split | graph | route | 每图均值 | p95 | 最大 | truth-consistent | fake-endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7,200 | 800,228 | 111.14 | 240 | 630 | 14,203 | 416,825 |
| validation | 1,440 | 148,551 | 103.16 | 216 | 600 | 2,832 | 71,756 |

因此 route table 以物理 graph 为单位一次构造并缓存，训练 batch 固定为 32 个 graph；不会在
每个 epoch 重复枚举 route，也不会截断或以 score/chi2/truth 预筛真实 physical candidate。

## 正在进行

执行 V2 train/validation run；不会生成或打开新的 test。

## 后续判定

V2 validation 需要同时满足 nominal eff>=0.70、purity>=0.95、fake<=0.05，并在 5/10 mm
相对 MLP、V1 full-context、V1 no-context 有明确优势。还必须比较同 checkpoint 的
base-edge control，确认提升不是来自重新 calibration 或 threshold。

只有该预先冻结假设通过 validation，才允许创建新的 source-disjoint multidirection test bank。
