# 顺序整改第3轮：真实共同帧shape/SS重编码验证

目标仍active，未短训练，没有发布TRAINING_APPROVAL。

## 新实际完成证据

两真实资产从proper新mesh重新构建1024 dualgrid，真正shape encoder posterior(samplefalse)→decoder，而不是轴置换旧features。独立worker PID2360153/2360219均exit0，encode14.683s/decode39.014s，总53.702s，无超时/OOM，GPU5回528MiB0%。

0086：581866 dualgridcells→2033 shape64tokens，feats/mean float32finite，decode583431vertices/1171880faces，rawlengthY/formaldiagnosticlengthZ。0fab：720880→2610tokens，721711vertices/1449094faces，同raw/up正确。

同帧无拟合对齐的每方向512sample exacttriangle最近surface对称平均误差6.40864e-5/4.41149e-5（长归一约1）；不是所有局部细节无损保证。

父独立检查common_shape_pilot20files hashes+completion manifesthash+两worker不存在，全部通过。原config/source/weights/旧PERM不变。

## 必要目标支撑修正

0fab新shape encoder实际coords2610，不同于之前Open3DSSsupport2618，独立supportIoU.93128925。不能沿用旧CPU编码目标假定匹配。

父实际从shape64coords重新CPU SS encode/decode，new目录shared_ss_shape_encoded；0086/0fab为2033/2610tokens，两new SS mean8×16³fp32finite，decodedoccupancy IoU1.0，targethash验证。坐标沿共同raw轴直接index，没有旧cyclicperm。与shape支持相同是构建保证，不冒充独立IoU检验。

## 指标方法修正

面积加权面心PCA仍受大三角/细三角tessellation影响；补exactsurface secondmoment=trianglecentroidouter+withintriangle covariance sum(vertex-centroid)outer/12。

0086 source/decode精确表面主轴偏角.74903155/.74309994°；0fab1.18177968/1.18784975°。旧manifest0fab源2.118°只是facecenter代理，不是真实倾斜，也不能声称decoder修正到1.188。

旧manifest保留，追加surface_moments_review独立SHAcomplete。验收代理17d8正在把此精确矩加到evaluate_fixed_frame，保旧proxy并新增v4report，不覆盖v3。

## 进行中与下一步

30a99...独立shape_flow_compatibility对照：相同新shape64support、nativealpha条件、seed42、sampler/decoder，official与fixedshape20k仅权重单变量，GPU6受限前台管理，不训练/no纹理/no正式GLB验收。要记录相同noise/supporthash，fixedframe无bestfit与exactsurface距离，不凭bbox给compatpass。

真实屋顶/轮组/鼻端image-camera关系及fixedshape兼容仍需图/数值审查。PBR全链未编码，SS/geometrypilot范围可以保留此限界，不用它做伪通过或无限阻碍。若固定shape不兼容，应先受控改下游方案而非盲加SS训练。

所有可靠性源码123CPUtests此前已完整通过；evaluate精确矩增tests待本轮最终汇总。上轮TRAINING_GATE_ROUND2里的shapeVAE null现已完成但approval仍false，后续新版gate应更新true并保历史。

工作产物remote diagnostics/remediation_20261013/common_shape_pilot、shared_ss_shape_encoded；本地COMMON_SHAPE_PILOT_REPORT.md/common_shape_pilot_manifest.json。没有正在运行的训练，没有旧结果清理。兼容agent可续，未complete/blocked目标。
