# 六资产训练改进：时长阶段结果

## 实际完成

同四训练/两留出、同输入与target/recipe，100步训练与SS评价完成。100run前50与原独立50run的RNG/sampler相同、640冻结base精确相同，但adapter/Adam非bitexact；不能声称同轨迹精确复制。原与重复50采样指标几乎相同，数值分歧已保存。

主续训100→200训练worker2478887 exit0/228.674秒，主评价2481067exit0/187.652秒。保存150/200完整组与300FP32adapter/640baseeq/无非finite。另父worker2479644在子代理被runtime报告failed后接管，但子恢复独立运行，造成一次非必要重复，两个不同新目录都保留，已duplicate_resume_execution_disclosure.json公开，不冒充两treatmentarms；只评价主ss_same_cohort_resume200。

|step|train mean IoU|internalholdoutmean|
|---|---:|---:|
|repeat50|.484115|.413991|
|100|.697701|.444102|
|150|.853145|.468747|
|200|.967032|.471880|

训练拟合接近记忆，留出150→200平台且0390略降，不继续更多步或全543训练。002SS base/50在seed42/0/1均错raw0，旧axis10EMA20k长度轴可选对但seed0倾斜约18.96°且旧targetup/handedness不同不作为生产修复。主200 SSseed42的002长度轴正确[10,64,12]，off1.217°，其他外部也长1。全部同条件/噪声控制已验证。

## 正式GLB验证

固定shape20k/stockdecoder/seed42/pipeline1024/正式remeshtexture/no纠轴。

- 002运行232.23秒exit0，artifact来源verified，longz/off1.287986°，z/y5.478854。父read全tri原坐标侧顶图，方向从50错x恢复，但几何仍未批准不劣于旧基线。
- 001运行212.07秒exit0，longz/off.589452°，z/y6.21886。
- 003运行267.74秒exit0，longz/off2.063107°，z/y5.13856。
- 004尚无GLB：三个case在一个outerSSH620秒批处理里，004仍导出时outer超时中断监督，worker2487047成为遗留自建processgroup。父发现后用pidfd有限等待420秒，仍停export/remesh，核对cmd及PGID后只终止该组；worker已结束/GPU6恢复528MiB0%。原退出码不能认证，runtime_manifest更新terminated_by_recovery_supervisor，保全部失败日志，不重跑/不以fastGLB代替正式。其余三case有效不冒四图成功。今后每正式case独立tool调用预算，不再以总批SSH超时包三case。

正式报告formal_step200_validation_v1/report.json：004missing→全组不能通过，其他geometrynotworse unverified。

## 目标定义下一独立实验

新nose_canonical_candidate仅对1208train/1f7cheldout properrawZ180几何副本，统一source鼻-Y，四其它资产identity引用，condition6完全不变。sourceGT语义选角，不根据heldoutpredictions选；不是旋转预测验收。父实际read候选图、验证source/conditionhash、Rproperdet+1/up/vertex/faces，数据尚须真正重编码而不置换旧latent。

准备的encode_nose_targets脚本9CPUtests通过，实际GPU编码尚未开始（子代理异常结束后已查无运行worker/output，明确要求恢复，不重复接管）。另一agent a5dff1只CPU准备同100步原厂起点目标定义对照，不执行训练。只有两目标重编码和roundtrip验证后才能审新实验审批；不混改时长、lr/rank/目标。

## 产物与状态

远程diagnostics/cohort_training_followup_v1：DURATION_STAGE_RESULT.json、RESUME200_REPORT.md、prefix50_comparison.json、prefix50_difference_scale.json、duplicate_resume_execution_disclosure.json、axis002_v1、formal_step200_validation_v1、formal002_50_vs200.png、004 recovery监督记录、nose_canonical_candidate。

目标active，训练停200，生产批准false，GPU0–3未用，source/cacheddata/历史权重/NJ未改。新实验未训练，下一round继续实际两目标编码及是否有必要的新target100对照。
