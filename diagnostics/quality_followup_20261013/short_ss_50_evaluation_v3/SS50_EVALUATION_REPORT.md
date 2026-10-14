# SS 两目标帧50步评价

只评价现存 checkpoint，不续训、不改权重。3输入：train0086精确cached生产RGB512，不重复裁剪；001/004 actual pipeline preprocess。seed42、生产getcond512→1024→SS顺序、12步sampler、无canonicalize/swap。所有指标为SSproxy，不能完整GLB验收。

## 独立检查

两live各940 tensors/150pairs/300 adapters，全部FP32、0非有限；640冻结参数逐tensor等于原厂base cast到对应精度。Adam所有step50、exp_avg/exp_avg_sqFP32，misc step50，两arm训练后的Python/NumPy/TorchCPU/全部CUDA RNG状态完全相等。目标不同未扰动训练随机序列。权重SHA：legacy19164b8c6610c122eefe5cfb1ecee64d39ade78a3a1bbfbb4d2ed3f560e7caff；proper8b9acf7fe7dbcbc21d368e8062697c5905627da7f6a1c05964c91b22b19a6276。

## 采样结果

|model|train0086 properGT IoU|train legacyGT IoU|count|depth,length,up spans|
|---|---:|---:|---:|---|
|base|.4449244|.4173729|2650|10,64,10|
|legacy live50|.6378421|.5669829|2096|8,64,10|
|proper live50|**.8667275**|.5428140|2057|8,64,10|
|legacy EMA50|.4497224|.4203687|2667|10,64,10|
|proper EMA50|.4574599|.4275362|2695|10,64,10|

正确共享GT2033tokens。proper live precision/recall与详细PCA/cov在metrics.json。base/legacy/proper的train主长度轴偏角 .46493/.40962/.66788度，不能单凭较小角度排名质量；proper更高IoU支持此单资产目标可学习，但不说明方向/几何普遍通过。EMAd=.9999在50步初始参数残留约99.50%，应只作记录，不主质量排名。

|input|base count/spans|legacy live|proper live|
|---|---|---|---|
|001|4818 /16,64,11|3020 /16,64,10|3108 /16,64,11|
|004|1582 /10,64,8|1449 /10,64,8|1416 /8,64,8|

两个holdout都仍lengthindex1；三tag偏角都<.81°，未见主轴collapse。004 proper的depth跨度10→8、tokens下降，说明截面收缩变化，**不能称泛化无退化**。holdout无已知匹配训练资产，不提供GT IoU或几何通过。

固定投影图已read_image：三行length/up、length/depth、depth/up，无旋转/对齐/bestfit；图中proper轮廓更靠正确目标。角度和cov使用全部占据体素等权而不是mesh triangle指标。mass_distribution.json另有各轴centroid/5,50,95分位及固定高低高度band计数；band不是轮组检测器，未把它当物理up语义通过。

训练loss父独立查first10→last10：legacy.03622→.02934，proper.02203→.01690，目标不同不可直接质量排序。没有本次固定噪声flow-loss附加实验，不伪造训练BF16 vs推理FP32比较。

## 执行与失败证据

v1 worker2386426 102.994s exit1：low_vram decoder CPU与targetz GPU不一致，诊断脚本设备错误，尚未采样。保留log/integrity，不改模型。
v2 worker2387408 126.649s exit1：采完base+legacy live6cases，第二次applyLoRA误包装已有adapter子层变300，诊断注入错误；父已独立修生产helper幂等并测试。保留v2结果，不重复这些已采样case。
v3先拷贝v2六case，仅补proper live3及两个EMA6cases，正确仅注入一次150layers。worker2388712 145.604s exit0，无600秒超时，全部15cases完成；GPU6恢复528MiB/0%，无遗留worker。目标decodeCPU FP32，预测decoder同一FP32，low_vram自动搬设备；训练/checkpoint不改变，阈值>0对两target同口径。

新目录：diagnostics/quality_followup_20261013/short_ss_50_evaluation_v3。metrics.json/integrity.json/mass_distribution.json/各target+predcoordsNPZ/train_fixed_projections.png/COMPLETE/artifact_hashes.json；父目录每版本script/command/log/pid/result保留。

结论：单资产proper目标50步的采样适合度明显优于legacy反例；heldout主轴未崩但截面变化需警惕。尚不足批准推广或自动续训。父应明确下一步停止或有限100步诊断；本子任务没有续训/正式GLB/shape采样。
