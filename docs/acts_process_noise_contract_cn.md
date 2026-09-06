# ACTS Process-Noise 契约（Stage B / Task B）

Workbook 97。WB96 已经确认持久化 CKF 5×5。本任务问：在冻结的 WB87
construction/validation 划分上，

    C_out = F C_in F^T + Q

（Q 来自 ACTS `MaterialInteractor`）能否描述目标残差。

不是 alignment，不进入 Measurement Model V2。

Model 0：无 process noise。Model 1：ACTS Q（唯一通过模型）。Model 2：Highland
仅作诊断，不能替代 ACTS，不能按 chi2 调 Q。

生产外推保持 `InteractionMultiScatering/Eloss/Record` 为 false。验证作业可以只
改 Gaudi 属性打开这些开关，不改 Calypso C++。Gate 保持 WB81/WB87/WB93 冻结值。
