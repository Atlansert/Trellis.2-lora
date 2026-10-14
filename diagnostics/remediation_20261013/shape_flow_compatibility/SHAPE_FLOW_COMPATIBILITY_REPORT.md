# 两资产共同坐标 Shape Flow 单变量兼容对照

## 执行与控制变量

不是训练，无权重更新，仅GPU6。新目录 diagnostics/remediation_20261013/shape_flow_compatibility，旧数据/权重/PERM/训练源码未修改。

外部600s保护，worker2362022，112.658s结束exit0、未超时；pipeline加载66.407s。GPU6最后恢复528MiB/0%，worker已不存在。CPU指标/绘图12.36s、退出0。

条件：0086bbab37559442 frame000、0fab4af3f9a73bdd frame006，从renders_cond_t现有原生alpha。实际pipeline.preprocess_image黑底保存，每case重置seed42，按get_cond512、get_cond1024、get_ss_cond顺序。使用1024条件、官方12步sampler/cfg7.5/rescale.5/interval[.6,1]/rescale_t3，sample_shape_slat内部反归一化一次，未重复归一化。

坐标：使用common_shape_pilot真实shapeencoder推得64支撑2033/2610（不是旧Open3D2618），不采SS；两tag的coords、token数、512/1024/SS条件、噪声与CPU RNG哈希全部相同。官方先完成两例，再150层rank8/alpha8/dropout0严格fullEMA加载fixed_shape20k；adapter FP32、eval，基础权重真实来源配置/权重hash与固定EMAhash保留。

输出：每case实际latentNPZ、raw几何NPZ/PLY和proper(x,z,-y)formal geometry-only PLY。没有纹理/GLB正式remesh，formal_acceptance_eligible=false。

## 数值

|对象|GT z/y|官方 z/y|fixed20k z/y|GT/官方/fixed准确surface PCA偏角|官方→fixed sampled Chamfer均值|
|---|---:|---:|---:|---|---|
|0086…|7.078747|6.893497|6.520739|0.74903° / 0.53112° / 0.65596°|0.00355965 → 0.00349770|
|0fab…|5.850830|5.873219|5.460607|1.18178° / 0.84611° / 1.12583°|0.00367798 → 0.00342017|

fixed车高比GT约+8.55%/+7.15%；同normalized长度。四生成几何finite、degeneratefaces0。0fab GT本身4退化面，不改变GT也不将其解释为生成坏面。

PCA采用exact triangle surface integral covariance，包括within-triangle项，避免大型source三角形面心抽样偏差。Chamfer严格为20000面积采样点→另20000采样点的双向KD近邻均值（seed4242），不是point-to-triangle精确距离，有采样误差底噪。没有ICP、重新居中、缩放、镜像或bestrotation；小均值差不是几何通过证据。

## 实际图像观察与门槛

固定formal ZY侧视与同一oblique投影、统一normalizedscale。父/子已用read_image检查：0086输出保大体鼻形与轮组，但fixed明显车高更大；0fab官方与fixed都有车鼻/底部细节与GT不同，fixed高度更大、wheelprofile没有GT清楚。稀疏facepainter点噪/缝不构成真实孔洞证据，需完整PLY/GLB进一步审查。

条件图车头左右与canonicalGT显示相反是视角/朝向定义差异，不能据此断言生成错误，也不能自动声称noseanchor已满足；尚无明确相机→统一车鼻语义契约。两例正确GT支撑驱动结果不证明SSflow预测支撑可用。

结论分层：

- **technical_support_compatibility = pass_limited_two_assets**：两case在给定正确GT支撑下，shape20k输出有限、保留正式长轴z，没有全局坐标崩溃。不是SSflow预测支撑通过，也不泛化548对象。
- **semantic_geometry_quality = not_passed**：父/子审查确认fixed车高+7–8.5%；0fab官方/fixed鼻及轮组明显不如GT。完整语义几何保真仍未满足，人工审查与数字距离不能互相替代。
- **training_approved = false**：总体几何不劣于官方/GT的strong gate未通过，不应因z-long/PCA/Chamfer好转启动短训。PBR共帧和完整四图正式成品验收也未完成。

本任务已收尾，不新增GPU任务；潜在autocast精度或roof/nose有符号关系诊断需另行委托。

## 证据

remote shape_flow_compatibility/manifest.json、metrics.json、fixed_profile.png、fixed_oblique.png；同级shape_flow_compatibility.log/pid/command/result/exit_code.txt及CPUlog/exit_code。

local remediation/shape_flow_runtime_manifest.json、shape_flow_metrics.json、shape_flow_fixed_profile.png、shape_flow_fixed_oblique.png及本报告。输入/输出和采样控制哈希已独立全量复核，已有成品未重跑或覆盖。

严格退出状态与任务范围已给父代理session-930f1bc5-a3cf-41c7-9a8a-6ddd77ba9dc9；无遗留worker或后台job。
