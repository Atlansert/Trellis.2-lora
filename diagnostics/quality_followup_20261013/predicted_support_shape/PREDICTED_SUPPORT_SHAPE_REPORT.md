# 预测SS支撑传播到shape：两输入2×2诊断

## 实际执行

新目录 diagnostics/quality_followup_20261013/predicted_support_shape。仅GPU6；worker2391406在161.980秒结束exit0，无600秒外部超时。CPU指标+完整三角轮廓绘制62.782秒exit0。所有worker已结束，GPU6回528MiB/0%；没有训练/权重更新、纹理采样或正式GLB remesh，没有使用GPU0–3，没有处理已隔离0fab。

8cases：输入train0086与heldout004；SS官方或proper live50；shape官方或fixed20k。SS每模型/输入仅采一次，两个shape分支使用该同一坐标与条件，并恢复采SS后的CPU/allCUDA RNG。创建LoRA的随机消耗发生在采样seed reset之前，不污染配对噪声。SS官方先采，随后只注入一次proper50。shape官方对象与深复制后的fixed对象各自独立，不重复注入或误用已适配权重作base。

SS两模型之间条件512/1024/SS和采样前RNG哈希一致；shape两模型噪声一致**仅限每个同SS/同输入pair**，不同SS产生token数不同，不声称跨SS噪声张量相同。所有8cases的32outputs、4SScoords以及COMPLETEmanifesthash独立复核，校验通过。

训练输入采用已缓存生产RGB512，不再次preprocess；heldout004按pipeline标准preprocess。训练与004的SS条件hash逐一与上轮SS50v3评估完全相等。训练图1024条件从该RGB512上采样，和更早shapeGT支撑对照的原始高分辨率native条件不是同一输入细节口径，不能跨实验直接把8.55%与本次1.48%视为单变量效果。

## 预测SS

| 输入 | SS | token数 | 对训练properGT支撑IoU |
|---|---|---:|---:|
| train0086 | 官方 | 2650 | .444924406 |
| train0086 | proper live50 | 2057 | **.866727522** |
| heldout004 | 官方 | 1582 | 无GT |
| heldout004 | proper live50 | 1416 | 无GT |

trainIoU与上轮.866727522精确吻合，说明使用了相同SS条件/采样口径。

## 形状传播：训练资产

GT共同raw归一geometry正式诊断坐标高度.141266525、宽度.120115861、长度.99999。

| SS / shape | 高度 | 宽度 | 精确表面主轴到z偏角 | sampleKD对称表面距离 |
|---|---:|---:|---:|---:|
| 官方 / 官方 | .149256229 | .141477734 | .774606° | .008056672 |
| 官方 / fixed20k | .147731811 | .138938814 | .578676° | .008149870 |
| proper50 / 官方 | .147839159 | .123329997 | .821787° | .003925365 |
| proper50 / fixed20k | **.143364280** | **.118010730** | .688118° | **.003665013** |

在本次给定输入口径下，proper预测SS带来的改善传播到了shape：双向采样近邻距离约降一半以上，宽度更接近GT；proper50+fixed20k高度比GT约+1.48%，宽度约-1.75%。这只是训练过的单资产诊断，不证明泛化或所有局部结构保真。

parent/子read_image完整三角samecamera轮廓：训练资产GT、官方支撑及proper支撑输出均鼻方向相容，proper支撑更接近车体；轮组/底部仍有差异。没有进行ICP、缩放、镜像、最佳旋转或纠轴。

指标：PCA使用精确三角表面积积分协方差；Chamfer为每面面积均匀20000点、seed4242的双向sampleKD近邻均值，不是精确最近三角面距离，具有采样误差。不用数值自动批准完整几何。

## 留出004：无GT，仅必要指标

| SS / shape | 高度 | 宽度 | 精确表面主轴到z偏角 |
|---|---:|---:|---:|
| 官方 / 官方 | .114258826 | .132004052 | .082089° |
| 官方 / fixed20k | .115102798 | .129226178 | .235320° |
| proper50 / 官方 | .117080986 | .102620006 | .582959° |
| proper50 / fixed20k | .119307607 | .101508737 | .258263° |

本例未出现整体错轴或此前axis10@20k约10°倾斜，但proper50使横向宽度明显缩小。未有004资产身份/真实几何，不计算GT IoU或保真度，不声称截面变化是正确改善。完整固定相机轮廓只是形体比较；同图输入鼻左右与固定相机显示差异不能单独视为镜像错误。

## 决策与边界

- 训练0086的proper支撑提升已传播到shape，结果是有意义的局部可学习性证据。
- 单资产过拟合50步不构成生产权重候选；不继续到100，不更新任何权重。
- 所有输出是无纹理/未正式remesh的raw或proper旋转PLY，不是GLB最终验收。
- 不支持因训练例改善就推广全548，亦不支持把heldout轴正确当作几何不劣于基线。
- source污染资产0fab完全排除。本任务未开展新的数据清理或长训练。

## 文件

remote predicted_support_shape/{manifest,COMPLETE,metrics,METRICS_COMPLETE,INDEPENDENT_VALIDATION}.json，same_camera_full_silhouettes.png，各8case raw_geometry.npz/PLY、shape_latent.npz、formal_geometry_only.ply；父目录run/evaluate_predicted_support_shape.py、bounded...py、command/PID/result/log/exitcode。

local remediation/predicted_support_shape_{manifest,metrics}.json、predicted_support_shape_silhouettes.png及本报告。四SS条件hash和priorv3匹配已记录INDEPENDENT_VALIDATION，所有旧产物与数据保留。
