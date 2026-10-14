# 六资产数据重建实际进度

## 已完成

源审核：六完整GLB、有效嵌入贴图与UV，单meshnode，没有已确认参考照片板/独立辅助节点；四yaw0、两yaw90原生X长轴。仅批准渲染，不认证整个543候选。

Blender4.5.1实际CPU512RGBA、samples16、threads8，每资产八相机角0..315、固定共同rawgeometry(depthX,lengthY,upZ)。首0086渲染38.42秒，其余37.07/38.57/39.32/35.86/34.27秒，全部exit0无timeout/worker结束，源文件未变。

全部48PNG真实alpha未背景差分，actualcameraWorld/Inverse/K保存，geom/source/review/options/script/PNGhash绑定。父独立全48hash、alpha非空/nonfull/no裁边、cameraCWinverseproper/float32tol1e-6/K投影检查通过。首独立fulltriangle0/180 silhouette对strongalpha IoU.96345/.96365，alphaoutsidegeometry0，投影点maxerr5.68e-14pixels。最初1e-7相机容差过紧首failedvalidation保留，按float32修正后passed；不是相机数据故障。

父read六主图contact白body/鼻/窗/轮组完整，光照偏平不承诺与历史像素同。原source/sourceimages完整保留。

真实shape1024dualgrid→posterior(samplefalse)/decode，GPU6两阶段31.76/97.10秒exit0；sharedSSshape64support→SSCPUmean8x16³/decode20.79秒exit0；finalize13.23秒exit0。actual6newmeansFP32/finite，SSroundtrip全部IoU1.0。sharedsupport一致是构建，不冒独立体素算法验证。

|对象|shape64tokens|shape对称平均surface距离|SSroundtripIoU|
|---|---:|---:|---:|
|0086|2033|.0000631094|1|
|008ae|2848|.0000436051|1|
|015de|2476|.0000511631|1|
|0390|2882|.0000523973|1|
|1208|2218|.0000436939|1|
|1f7c|2167|.0000544097|1|

sameasset512sample/方向到精确三角表面，小误差不能代替局部细节/semantic全质量。父全部stagefilesSHA/最终COMPLETE独立检查passed，GPU6回528MiB/0%。编码没有更新训练权重。

## 进行中

- 独立fb3ac595 target_validation读取全部6arrays/hash、SSCPU可独立decode核对、shape几何/PCA及条件语义检查；只数据前置范围，不批准生产模型。
- ea3652a0 preparing新shorttraining，建议固定train4(0086/008ae/015de/1208)，heldout2(0390/1f7c)，仅view000条件。其余7view是几何审查，不盲共享canonical目标。
- 首50stepbounded先停，四训练/两留出单一proper新data配方，同原厂baseline，不重复训练已知错误targetnegativecontrol。审批尚未发布，未开始新批训练。

## 工作路径与安全范围

服务器 /data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1：asset_review/renderer_prepare/renders/first_render_validation_v2/batch_render_validation.json/encoding_prepare/encoded_targets_v1。

六物体源scene全部保留，不source节点自动删除、不修改oldcache/weights/PERM/NJ、没用GPU0–3、没stopComfyUI。supervisor首次commandrecordprefix与commandfile同名导致拒执行，换独立executionprefix后actual渲正常，保初spec不覆盖。

目标active，接下来按gate独立target通过→训练50→内部holdout/外部四图正式成品验收→决定增规模。不得因小batch数据数值重建通过就直接543全训。
