# LoRA adapter 分支精度单变量诊断

## 范围与执行

仅新诊断目录 diagnostics/quality_followup_20261013/adapter_precision；没有训练、参数更新、旧数据/模型/源码/权重/PERM/NJ改动。临时monkeypatch LoRALinear.forward只在独立worker生效，finally恢复；base F.linear保持原先输入/权重dtype行为，只有lora_down/up矩阵运算包CUDA bf16 autocast。DINO条件、其他pipeline运算、decoder均不包全局autocast。

同fixed_shape20k EMA、rank8/alpha8/dropout0/eval，官方1024decoder、官方12步sampler；同GTshape64coords2033/2610、nativealpha frame0/6、seed42、getcond512→1024→ss顺序。两mode为normal与adapter_autocast_bf16。

worker PID2379332，600秒外部processgroup保护，163.388秒exit0/no timeout；四case全部成功、finite，不sanitize NaN。GPU6回528MiB/0%，worker不在/proc，未停止ComfyUI。CPU指标第一次新脚本缩进错误已定位，保留失败log；只修新诊断脚本，第二次exit0/12.5sec，无GPU重复。

## 控制验证

同一资产两mode的coords、512/1024/SS条件、noise及CPU RNG哈希全部一致。DINO1024条件实际float32；300adapter参数fp32；所有model参数的对象id、dtype与数值hash前后不变。输出20文件hash独立复核。

normal输出与前一fixedshape20k对照指标一致，支持复用对照口径。此为推理算子精度比较，不是训练整pipeline autocast复现。

|指标|0086（真实几何可用）|0fab（GT已由父发现参考平面污染，隔离）|
|---|---:|---:|
|latent relative L2变化|4.9891%|4.3014%|
|latent RMS差|.27818|.22939|
|latent最大绝对差|6.6106|9.2001|
|不同latent元素|65056/65056|83519/83520|
|normal→bf16车高变化|+0.02616%|−0.06365%|
|相对当前参考GT高度 normal→bf16|+8.5487%→+8.5771%|+7.1490%→+7.0808%（不可作为真实车体误差）|
|20k sampled KD Chamfer normal→bf16|.00349770→.00350812|.00342017→.00341866（污染GT，仅技术记录）|

PCA采用精确三角表面积矩。Chamfer为20000面积采样点间KD近邻，不是精确三角距离；无ICP、旋转拟合、重新居中、缩放。raw/数学正式轴旋转PLY不是remesh正式GLB验收。

## 图像审查与结论

两固定profile/oblique图已实际read_image：两mode宏观高度及轮廓变化很小，局部细节可变化；面片稀疏painter噪点不是几何孔洞证据。图原标签GT propergeometry不意味着0fab是可信纯车体GT；其已被父semantic调查识别参考平面混入，必须隔离，不能拿它批准训练或归因真实车体质量。

**0086中adapter分支fp32 vs bf16仅引起约0.026%的高度差，远不足解释当前约8.55%的高度偏差；这一个小变量不是该样本宏观车高问题的主要解释。** 0fab同输入数值比较也只有约0.064%高度变化，但因GT污染不得拿7.15%作真实几何归因。

latent变化约4–5%并非微不足道，不能推断所有局部几何不受精度影响。此次不排除DINO条件bf16/fp32、其他训练autocast算子或历史量化对质量的作用，不泛化到整个548对象。没有发布训练批准，也没有假设bf16更好。

## 证据

远程新目录的manifest.json、metrics.json、fixed_profile.png、fixed_oblique.png与四case latent/raw NPZ/PLY/formal诊断PLY。父目录run_adapter_precision.py/evaluate_adapter_precision.py、adapter_precision.log/pid/command/result/exit，以及CPU失败/重试日志。

本地 adapter_precision_manifest.json、adapter_precision_metrics.json、adapter_precision_profile.png、adapter_precision_oblique.png与本报告。源脚本和LoRA源码哈希、所有实际模型来源配置/权重哈希在runtime manifest中。

training_approved=false，formal_acceptance_eligible=false。所有本任务进程已结束，无后台job。
