# T02 Bootstrap 重复计数与数值契约

修复 event bootstrap 的布尔 mask 丢重问题，以及欠定 SVD 丢失的 right-null。
Workbook 68/69/73 的历史 rank 保持冻结，不因此改写成部署通过。

## Bootstrap

事件键 = `(original_source_uid, run, event)`。有放回抽样保留行 multiplicity。
报告 `n_draws`、`n_unique`、`effective_multiplicity`、seed 和 invalid replicate
原因。cluster-local 已正确的 row-list 分支未回退。

## SVD

高/方阵仍用经济型右基。`m < n` 时补全 right-null，使 `dim(V_null)=n-rank`。

## 协方差

`require_spd`：有限、对称、Cholesky。`[[1,2],[2,1]]` 被拒绝，不能得到
`chi2=-2`。不做特征值裁剪。

## 冻结

`rank_tolerance=0.01`。尺度矩阵 `S` 不变。
