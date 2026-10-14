# 同六资产100→200诊断续训

固定原4train/2internalholdout、view000、目标、原厂基座、rank8alpha8、lr1e-4、batch4split2及seed42；仅延长步数，新独立目录ss_same_cohort_resume200。真实trainerload100恢复参数/优化器/RNG，模型FP32adapter保持，保存完整150/200 groups。受管worker2478887为228.674秒/exit0/no900timeout；评价worker2481067为187.652秒/exit0/no600timeout。GPU6回528MiB/0%，两worker已不存在。

|step|train均IoU|train最小|holdout均IoU|holdout最小|external002|
|---|---:|---:|---:|---:|---|
|100(父已有)|.69770|.34730|.44410|.31921|rawdim0错轴|
|150|.85314|.65655|.46875|.37117|[64,15,12],89.883°错轴|
|200|.96703|.95354|.47188|.36748|[10,64,12],1.217° length1|

200各训练GTIoU .961759/.953536/.965395/.987438；holdout0390 .367476、1f7c .576284。150到200训练接近记忆，heldout0390微降、1f7c升。external001/003/004 length1偏角.943/1.457/.508°，外部无匹配GT，不报告保真IoU。002从150到200主轴改变是真实无纠轴输出，不等于正式GLB通过。

评价仅采150/200各10SScases，已缓存原厂base的SHA验证后复用，不重复采base。所有每输入条件、CPU采样RNG、噪声hash与缓存base一致。SSproxy不是正式shape/textureGLB验收。两步完整groupSHA/大小和300AdamstateFP32/step正确验证。

16worker及prefetch状态未完整恢复，不能声称精确连续轨迹；150/200是在同一续训run内比较。父在runtimefailed通知后接管也执行另一resume200目录；此处透明记录实际发生两次独立训练，均保留，父run仅复现对照，不自动当主结果。本子只评价ss_same_cohort_resume200。

初评价脚本生成因literal替换断言失败，未创建或启动评价GPUworker；修为按summary行定位生成后一次成功运行，未重复GPU评价。无生产源码/源资产/data/PERM/NJ/旧结果修改，没有其它训练或自动进一步步数。生产批准false。

建议停止200，优先正式002+完整四图验证，关注训练/留出差距和车头语义；不因训练IoU.967自动扩大或投产。证据：RESUME200_RESULT.json、resume200_execution.*、resume200_eval_execution.*、resume200_evaluation/metrics.json/COMPLETE及50/100历史结果均保留。
