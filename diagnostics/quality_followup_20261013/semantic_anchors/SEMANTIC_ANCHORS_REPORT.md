# 两资产条件图与共同帧语义锚点诊断

仅CPU，不训练、不推理、不使用GPU，不修改原数据/PERM/权重/源码。新目录diagnostics/quality_followup_20261013/semantic_anchors。

## 相机约定：此前左右相反不构成镜像证据

阅读现renderer：相机Blender世界位置(cos(yaw),sin(yaw),0)，TRACK_NEG_Z/UP_Y约束看原点，世界高轴+Z。proper common mesh采用native→Blender(x,-z,y)再row Rz(yaw)。同yaw相机在共同raw帧位置为+X，朝-X，上+Z，screen-right=forward×up=+Y。

正式GLB坐标(x,z,-y)下，screen-right=-formalZ，而原固定profile画的是+formalZ向右。因此以前条件图车头左右与GT相反，是两个固定视图从不同侧看车的合法显示差异；不能仅据此宣称模型镜像或车头反转。

新图所有GT/official/fixed都用共同同一相机screen-right=rawY，screen-up=rawZ；没有拟合、挑best视角、镜像或更改mesh，完整所有三角投影轮廓。

真实历史相机外参仍无法恢复：custom_prepare_data把transforms矩阵写固定占位，只保yaw。这里验证的是当前renderer约定与现几何/条件的兼容解释，不是重新认证历史外参/相机标定。

## 程序候选与图像观察

以GT长轴rawY两端8%区间截面的高度跨度×宽度跨度较小端标为taper/nose候选；这不是人工标注的测量真值。top+Z线为roof候选，bottom+Z低位线为wheel候选；面积加权下20%高度band沿长度20bins分布保存在anchors.json，不等于识别每个真实轮组。

- 0086bbab37559442：负Y端截面height.082386×width.086680，正Y端.136293×.119161，nose候选-Y。已read_image，条件鼻左；同相机GT/official/fixed鼻均左，车顶在上、轮/底在下，up符号与实图形态相容。fixed高度仍.153343 vsGT.141267，与先前质量问题一致，语义方向一致不证明几何通过。
- 0fab4af3f9a73bdd：正Y端截面height.170914×width.062395，负Y端.148225×.119369，nose候选+Y。条件鼻右，与同相机GT的taper候选相容，不能再将原profile左右差解释为必然镜像。屋顶/低轮组方向未见up/down翻转。但完整轮廓很大矩形遮蔽细节，不能仅用silhouette批准。

## 新发现：0fab源几何含大平面，稀疏painter会漏掉

完整三角silhouette中0fab呈大矩形。独立最大三角检查：源最大三角面积.072361894，span[~7.4e-8,.97637594,.14822555]，几乎跨整辆车的侧面平面。0086最大三角面积仅8.547e-6、跨度约.004。

这是实际源mesh里存在大平面的证据，不是渲染随机错误；其材质是否透明/用于贴图背景/实际车身平板不能仅由geometry断定。GT中保留它的真实原因未知。此前稀疏facepainter只取部分面，会漏掉这个大三角，造成轮廓看起来更细；不能把稀疏图当完整mask。

源Scene节点进一步核验：Fromt节点geometry名CJ5-0506-Side Front，仅2三角，native world bounds约[0..30.8766,0..4.6874,z≈0]；Side节点同类2三角。材质alphaMode=BLEND，但共享6080×1080图片的实际alpha全255（nonzero/strong coverage1、仅一个alpha值），不是已证的透明alpha裁车card。已read_image嵌入贴图，它是白底车头正面和长侧面参考照片拼图，带大块白背景。这些矩形贴图参考面作为实际几何进入dump和voxelization，会制造超大整幅侧向支撑。另node1435/Cube.001为12faces小盒，无材质；不把所有盒状部分自动判无用。

建议0fab作为源参考card几何风险先隔离，而不是拿它的全部mesh当纯车GT质量门槛。可靠native alpha条件只证明条件图透明来源，不能证明资产只有真实三维车体。没有删除源平面或更改材质；语义清理需资产审核、明确原作者建模用途与新数据版本。official/fixed由同GT支撑驱动，保大支撑平面不自动说明flow学坏，之前两tag鼻/轮组比较应限定为与含参考card的原GT比较，不能全归因模型。

## 限界和决策

两资产鼻候选符号并不一致（0086-Y，0fab+Y），因此当前common_frame仅统一长轴和up，**没有跨资产统一车鼻符号**。canonical target可定义nose方向，但必须资产级语义标注/可信taper检查，不能浮点PCA min或mask最大长宽比自动认定。

旧SS的negative-up反射问题和新common frame的nose左右显示问题是不同概念：本次合法camera变化解释左右相反，不否定旧SS框架反射/错up的已验问题。没有证据证明新common frame需要左右镜像。

training_approved=false；两资产观察不批准全548，更不启动shapeFlow或训练。对semantic图实际read_image完成，程序候选与人工观察分开。

## 证据

semantic_camera_consistent.png（条件/GT/official/fixed，同相机完整三角轮廓+候选线），anchors.json（上下bounds、截面、下band面积分布、sourceSHA），largest_source_triangles.json（源大平面证据），semantic_anchor_diagnostic.py（可复现脚本）。脚本前台exit0、51.024秒，无后台worker。新metadata hash索引与相机矩阵在COMPLETE/semantic_manifest中。
