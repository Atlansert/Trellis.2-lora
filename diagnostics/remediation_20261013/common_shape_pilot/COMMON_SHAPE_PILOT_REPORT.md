# 两真实资产共帧1024 shape VAE验证

## Completed：真实编码与解码，不是旧latent轴置换

新目录 `/data/lzx/TRELLIS.2/diagnostics/remediation_20261013/common_shape_pilot` 在不存在时创建。仅0086bbab37559442和0fab4af3f9a73bdd，用上一propergeometry（native glTF Y-up→proper raw lengthY/upZ，det+1）重建1024 flexible dual grid，再执行本地官方shapeencoder posterior mean（sample_posterior=False）与decoder.eval()。没有LoRA、纹理、训练、正式GLB导出，未改变既有数据、datasetcache、权重、PERM、NJ或父basic/train文件。

Encoder配置 `shape_enc_next_dc_f16c32_fp16`，权重SHA256 f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41。Decoder `shape_dec_next_dc_f16c32_fp16`，权重SHA256 e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581。Decoder原JSON resolution256，但运行显式set_resolution(1024)，不是低分辨率替代。完整架构及JSON hash在manifest。

| 指标 | 0086bbab37559442 | 0fab4af3f9a73bdd |
|---|---:|---:|
| 1024 dualgrid voxels | 581866 | 720880 |
| grid64 shape encoder tokens | 2033 | 2610 |
| posterior mean shape | 2033×32 | 2610×32 |
| output coords spans | 8,64,10 | 8,64,12 |
| dualgrid//16 vs ENCsupport IoU | 1 | 1 |
| ENCsupport vs独立Open3D64支持IoU | 1 | 0.93128925 |
| decoded vertices / faces | 583431 / 1171880 | 721711 / 1449094 |
| rawdecoded extents | .1201593,.9998767,.1412804 | .1198500,.9998601,.1709513 |
| source/decode长度轴PCA偏角（精确面积矩） | .74903155° / .74309994° | 1.18177968° / 1.18784975° |
| 对称平均surface距离，归一化长约1 | 6.40864e-5 | 4.41149e-5 |

所有posterior feats/mean为float32、有限，保存coords为uint8（内部计算int32），grid64；均与return_raw mean逐值相等。采用与原encode_shape_latent一致的dualvertex uint8量化输入，不对已学特征/旧latent做空间置换。

surface误差是同资产、同坐标、不做拟合对齐，每方向512个面积均匀采样点到对侧精确最近三角表面，非KD近似。本次rtree已有，不安装依赖。采样平均不能保证所有局部细节无损。

## PCA方法限定与补充修正

初manifest记录面积加权面心PCA，但输入有长大三角、解码网格有细三角，面心二阶矩有tessellation偏置。追加 `surface_moments_review.json`（独立SHA completion）使用每三角表面精确积分二阶矩，包含within-triangle项；上表以此为主。尤其0fab初源2.118°只是面心proxy，精确表面值1.18178°，不能声称decoder将真实倾角2.118°修正到1.188°。初manifest保留，不静默覆盖。

解码 raw 长轴Y；仅数学正式proper rotation(x,z,-y)后诊断extent长轴Z、高轴Y。PLY是未remesh原始geometry，不能替代formalGLB验收。

## Shared SS support与remaining evidence

`ss_support64_from_shape_encoder.npz` 来自真正编码后的grid64 coords，和shape ENCsupport逐点共享，属于构造一致性；独立Open3D支持IoU另外报告，不捏造成独立1。

0fab上一CPU common_frame_pilot SS有2618 tokens，本次真实shape有2610，IoU.93128925。旧SS mean不能假定匹配本次共同shape支持。后续应由本次共享SS支持重编码SS mean，或固定支持形状流诊断直接使用本次coords。这一额外SS编码不是本任务执行范围。

当前geometry/SS pilot的剩余前置是固定shape EMA20k在新commonframe条件下的兼容性；该对照由父代理另行执行，本代理不重复。PBR/纹理共帧编码属于完整纹理链的未验证边界，不作为本阶段geometry/SS pilot额外必需项。真实车顶/车轮/车鼻语义标记通过VAE后对应关系、图片/相机与target朝向关系也未由本任务证明。输入up坐标marker不能自动当解码semantic标签。`training_approved:false`，不发布训练审批，不宣称总训练目标完成。

## 运行与清理证据

GPU5运行前528MiB/0%；独立processgroup encoder PID2360153 exit0，14.683s；decoder PID2360219 exit0，39.014s；总53.702s。每阶段outer300秒、总600秒，没有timeout/OOM、不重试全量。stdout/err独立日志、命令/workerhash/pid/exit保存。两个worker /proc已不存在，GPU5回528MiB/0%；GPU0–3未使用、常驻服务未停止。

manifest+COMPLETE原子任务完成语义：`completed:true`只指这两资产1024编码解码程序完成，不是训练gate通过。20个原任务artifact SHA与COMPLETE绑定manifest hash已独立全复核；追加面积矩补充有自己的SHA completion。

Remote证据：`common_shape_pilot/manifest.json`、`COMPLETE.json`、`encode.log`、`decode.log`、`surface_moments_review.json`，每资产`new_dual_grid1024.npz`、`new_shape_mean.npz`、`ss_support64_from_shape_encoder.npz`、`decoded_raw_geometry.{npz,ply}`。工作脚本独立在诊断父目录，不触父正在修改的训练源码。

本地主要证据：common_shape_pilot_manifest.json与本报告。仅移交验证结果，不启动shape-flow兼容对照或训练。
判定：**两资产1024共帧geometry shape VAE encode/decode完成，数值重建证据良好；训练gate仍未通过。**
终结后用户/父可根据上述剩余门槛决定后续小范围兼容验证。
