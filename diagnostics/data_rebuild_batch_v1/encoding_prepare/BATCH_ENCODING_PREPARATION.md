# 六资产共同帧 Shape / SS 编码准备

仅准备，不启动任何模型、GPU、编码或训练。代码新建在服务器 diagnostics/data_rebuild_batch_v1/encoding_prepare，原数据/权重/源脚本/PERM/NJ未修改。

## 验证结果

15项CPU测试通过，-W error::ResourceWarning；源码py_compile通过。包括显式批准、精确六对象范围、完整渲染manifest/COMPLETE绑定、源资产变化、图片及几何变化、重新哈希但缺失视图、NaN和错误长轴、路径越界、旧输出拒绝、部分输出拒绝、完成产物篡改、模型依赖变化、父objects/render_directory接口与GPU6/CPU阶段限制。

测试仅说明输入契约和准备代码行为，不说明真实编码/解码已通过。

## 父代理门槛接口

必须先存在父代理 batch_render_validation.json：

```json
{"schema":"trellis.batch-render-validation.v1","passed":true,"approved_for_encoding":true,"training_approved":false,"assets":[{"object_id":"0086bbab37559442","render_dir":"ABS/actual_render_dir","render_manifest_sha256":"64hex"}]}
```

assets实际必须六对象完整清单：0086、008ae、015de、0390、1208、1f7c的完整16位IDs。也支持父的objects列表、render_directory字段。未知schema/未批准/范围不一致拒绝。默认validate阶段不importtorch/bpy/o_voxel。

父校验实际48图片RGBA、相机矩阵/inverse/K/投影、真实geom和源审核。编码入口再次核对八视图SHA、COMPLETE/manifestSHA、源资产/审核SHA、geomSHA及有限性/[-.5,.5]规范化/长轴Y。批量source/input改变即失败，不靠statussuccess单独授权。

## 精确CLI

程序：/home/test/miniconda3/envs/trellis2/bin/python -u
入口：/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/encoding_prepare/encode_render_batch.py
门槛：/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/batch_render_validation.json
输出建议：/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1/encoded_targets_v1（首次应不存在）。

依次执行：

```
CUDA_VISIBLE_DEVICES='' python encode_render_batch.py --validation ABS/gate.json --stage validate
CUDA_VISIBLE_DEVICES=6 python encode_render_batch.py --validation ABS/gate.json --output ABS/encoded_targets_v1 --stage encode
CUDA_VISIBLE_DEVICES=6 python encode_render_batch.py --validation ABS/gate.json --output ABS/encoded_targets_v1 --stage decode
CUDA_VISIBLE_DEVICES='' python encode_render_batch.py --validation ABS/gate.json --output ABS/encoded_targets_v1 --stage ss
CUDA_VISIBLE_DEVICES='' python encode_render_batch.py --validation ABS/gate.json --output ABS/encoded_targets_v1 --stage finalize
```

每阶段使用父现有run_stage_bounded.py、独立JSON argv及新recordprefix，600秒外部保护。禁止GPU0–3，GPU6单设备；CPU SS另进程，避免o_voxelGPU初始化限制。此报告中的python应替换上面的实际conda绝对路径。

## 真正计算链路

- 输入同renderer canonical_geometry.npz，已proper共同帧depthX/lengthY/upZ；不再次旋转，不旧循环PERM。
- actual1024 flexible dualgrid，localdualverticesuint8量化同原先工具，再官方shapeencoder posterior sample_posterior=False/return_raw=True。
- shape32feats为float32有限，grid64coords内部int32/磁盘uint8；ss_support64保int32。实际coords与dualgrid//16 support精确相等才commit。
- 官方shape decoder显式set_resolution(1024)，保存raw decodedgeometry NPZ/PLY。精确三角表面积积分矩与每侧512面积样本到最近三角表面距离，不ICP/缩放/旋转拟合。既有rtree缺失则报错，不装依赖或伪装精确距离。
- SS直接从同shape64coords输入编码，不置换已学mean。CPU encoder119MB使用真实local路径/data/lzx/TRELLIS/checkpoints/ckpts/ss_enc_conv3d_16l8_fp16；decoder使用已确认HF本地snapshot。显式fp32/samplefalse，保存8×16³mean和decoded64support、IoU。
- 主条件仅渲染view000；其余七图用于几何相机审查，不盲目将全部任意视角共享目标。

## 安全与续跑

新输出INPUT.json绑定父gate哈希；每资产encode/decode/ss各提交 *_COMPLETE.json，包含geometry输入SHA、输出SHA、checkpoint配置/权重SHA与结果。重跑仅已完整且输入/输出/模型依赖哈希仍匹配者跳过。任何部分文件不覆写，需选择新输出版本。finalize已有完整组拒覆盖。

最终manifest/COMPLETE仅表示六资产计算完成，training_approved=false、PBR未编码。质量阈值与真实roof/wheel/nose条件对应仍需父代理审查；坐标正确不自动证明decoder语义细节。

## 来源与预算

复用并阅读上一实际common_shape_pilot_worker.py编码/解码逻辑，未执行本批模型构建。预计六资产GPU encode+decode约160–240秒、SS CPU额外数十秒，仅估计，不是本批实测。

入口SHA 5e3a2fc02831f38857c98c1f17c7252386ea1b35b32928024b31e90bf424f59f；契约SHA 71f4aea1dbf04df00bf1be2b99808d0bb81bbdca15f3ed879a7acc01075af781。完整准备结果在PREPARATION_RESULT.json。
