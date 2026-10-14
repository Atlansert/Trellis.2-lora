# 0fab清理候选剩余Cube独立审查

只读CPU，未使用GPU/模型/训练、未修改原资产或候选、未读取训练权重。实际检查候选Scene的两个剩余实例并绘制三向原生正交图。主命令13.655秒退出0，没有后台工作。

## 精确几何事实

- node `1435` → geometry `Cube.001`，24顶点/12面，无纹理或material；局部顶点[-1,+1]，世界变换为原点中心、每轴scale0.717500031。
- 因此世界包围盒每轴[-.717500031,+.717500031]，**三条边均1.435000062**原生单位，世界体积约2.955平方米立方单位。不能单凭node名字判断，但尺寸与1435毫米标准轨距高度吻合（假设native单位米；资产未单独认证units）。
- 主车辆node `material` → `material.006`，870175顶点/1496874面，世界长度30.898388、高度4.560257、宽度3.590908，native X长度/Y向上。
- Cube约75.26%的包围盒体积不在车辆整体包围盒内（这里只是bbox相交，不是mesh实交体积）；向车体前方突出.724925、低于车辆最小Y .717500。Cube高度相当于车高31.47%。
- 候选全体长度31.623313、高5.277757；车辆本体长度30.898388、高4.560257。保留此cube会影响归一化scale/中心/低位支撑，故不是数值可忽略的小点。
- source和candidateindexed `is_watertight=False`不作为物理缺陷判定：GLB的UV/seam重复顶点可能造成索引拓扑假不闭合。

## 图像审查

已read_image剩余组件三向图：cyan车辆，orangeCube。侧视在车鼻前/下方形成独立规则方块；顶视位于车头中心附近；端视位于车辆中线低位并明显低于主体。图侧/顶投影使用所有三角形轮廓，端视为点采样明确标注，不是真实材质渲染或正面照片。

这是一块**简单等边、轴对齐、无材质的1.435尺寸立方体**，不像具有建模细节的实际耦合器。几何+尺寸+名字共同支持轨距/尺度参考辅助块假设，比仅名字更强；仍缺作者说明或可信部件语义标注，不能把假设升级成已确认用途。

## 决策

当前Fromt/Side-only候选仍不批准扩展训练或作为纯车体GT。必须先作剩余Cube的明确asset级包含/排除审核；若决定排除，另建新白名单版本并重新归一化、SS/shape编码及条件几何关系验证，不能静默删现有candidate。

本审查不创建新的vehicle-only文件，不自动删除Cube、不修改已生成候选、不声称coupler绝对不存在。可作为下一个**隔离的有/无Cube数据诊断**设计依据，不能直接作生产训练集。

## 身份与证据

原sourceSHA `0fab4af3f9a73bddb8dfd5e744cb92dba7efffa82e6d4a5b46d51e31049988b1`；candidateSHA `0ec2e64afbea940050622e77a94ea5407f40e8194e383f37a0b5b0dfad88e630`，现场校验均保持。

远程新证据目录 `/data/lzx/TRELLIS.2/diagnostics/quality_followup_20261013/remaining_cube_review`：review.json（节点世界变换、bounds、尺寸、material及decision）、remaining_cube_orthographic.png、COMPLETE.json绑定报告与图hash。候选仍保留原1435/material两节点。

训练批准false；未涉及父GPU6预测SS→shape8case工作，不重复推理。
