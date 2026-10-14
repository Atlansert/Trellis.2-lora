# 顺序整改第2轮：可靠性修复完成，训练前置仍不通过

## 已核实结果

1. 全量alpha覆盖与四图验收的commit/hash父代理独立复核。1182对象/28368帧；所有633derived继续隔离。风险85对象不等于全部损坏判定。
2. 新共同帧两资产geometry+SS VAE实际CPU encode/decode完成，两对象occupancy IoU1.0，newmeans finitefp32×8×16³。properdet+1、rawlengthY/upZ，formal lengthZ/upY。共同shape support相同是几何构建保证，不是shapeVAE编码证明。
3. Stage3最终12tests，四图axis10正式GLB齐。003运行v2manifestverified，历史11artifactprovenance未完全验证；全部geometry-not-worse unverified，004offaxis10.17356°（仅10°诊断参考非生产新门槛）。新003固定侧视预览亦有形体/车头方向与细节问题，z/y4.066并非验收通过。
4. Stage4完整落地：trainer29distincttests+trainlaunch25distincttests；与其余69一起全套**123CPUtests通过**（-W errorResourceWarning，约1.4sec，实际父整套命令运行4sec）。没有真实训练。

## 精度与保存行为

- actualBasicTrainer.load/save/finetune_from方法经AST提取绑定tiny假trainer（不是只测helper）；fresh/save/resume下一步loss与adapter更新bitexact。
- 精确保LoRAfp32原数据而非bf16.float，保Parameter身份和complexbuffers；Adam浮点moments与参数dtype相匹配。
- 同步独立CPU整组snapshot→逐件fsync+atomicrename→complete_stepXXXXXXX.json最后提交，模型/EMA/misc完整组size/SHA验证。拒完整/部分旧step覆写，失败无complete，留partial证据不清理旧文件。
- latest仅complete组，对badmarkerfailclosed；旧misc-only不可抢latest。旧数字步可显式选择，模型/EMA/misc完整非空检查并warn缺事务/hash证据，不迁移旧marker。
- freshnone/无complete+历史输出拒绝，在保存config/command之前；记录run_attempts uniquejson保历史配置/命令，重试只有本outputcomplete可继续，否则raise不从0再训。失败耗尽非零。
- rank0 Python/NumPy/TorchCPU/CUDA RNG保存恢复，其他rank不接收rank0state；workerprefetch/DDP全rank RNG未覆盖，不保证轨迹级exact。
- skip_snapshot阻止最终昂贵采样后丢图，enabled采样error仍传播；max_steps非i_savealigned最终commit补存、已提交不duplicate覆盖。

## 修复限制

旧checkpoint若自身已经保存bf16adapter，float32不能逆转历史量化；现EMA文件fp32不证明当时livefp32。shapev2真实resume_run.log证明ckpt8000→20k恢复过，但缺当时immutable源码版本，不能据此宣称精度bug是已确认退化原因。当前tinybitexact不代表可复原历史大模型训练。

整组同步CPUclone+全hash增加RAM/IO，各DDPrankloadhash重复。大模型/GPU/DDP训练回归尚未执行，只有CPU可靠性与真实003推理。原trainedweights/data/cache/旧PERM未变。

## 正在继续的训练前置

独立子代理f36a...做两资产真实1024dualgrid→shapeencoder posterior→shape decoder小验证。此前CPUimport o_voxel要求CUDAbackend初始化，不是缺依赖；仅GPU5任务受外部预算/进程组保护，不install/不训练。

共同数据的shape/PBR VAE、真实roof/wheels/nose语义、固定shape20k与新共同帧兼容仍未证，**TRAINING_APPROVAL=false，不启动短训练**。下一步先actualshape VAE，再同support/seed条件下官方vsfixedshapeflow单变量推理诊断（不更新权重），满足数据/坐标契约之后才短训练。

## 证据位置

服务器：diagnostics/remediation_20261013/all_tests_final_round2.log/exit_code、basic_checkpoint_contract_result.json、train_launch_result.json、implementation_hashes_round2.json、common_frame_pilot/manifest.json、fixed_frame_acceptance/v3/report.json。

本地：同remediation目录对应记录及TRAINING_GATE_ROUND2.json。目标仍active，本轮未complete或blocked。所有历史产物/未通过证据保留。
