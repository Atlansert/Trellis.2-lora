# 质量续查第1轮

目标 active；生产训练未批准，尚未执行任何训练。

## 新事实与纠正

1. 相机语义核实：renderer 同yaw相机在共同raw+X，朝-X/up+Z，screen右=+rawY=-formalZ。此前固定profile的+formalZ右与输入鼻方向相反，是合法视角差异，不是镜像证据。完整同相机三角轮廓图中0086条件/GT/官方/fixed鼻左相容、roof上/wheel低相容。两资产nose符号不同，不批准跨548nose统一。
2. 0fab原资产实际包含建模参考照片节点Fromt/Side，各2大三角，opaque6080x1080白底火车照片atlas。material标BLEND但图alpha全255，不是透明车形card。referencecards混入material-free mesh dump/shape监督。父已read嵌入图片与samecamera完整轮廓，0fab不再视为纯真实车体GT，显式隔离，不删节点/改原文件。
3. 全548native候选几何risk扫描完成，96.55sec CPU/worker2381217exit0，readerrors0；18datachunks。5风险对象f50d6a143bdfe293、0fab4af3f9a73bdd、164d20b761652694、3d80f1c0d8bebe55、5d4a5e8e46991642。只有0fab深入材质/texture已证card，其余风险待review；不自动剔大tri/BLEND，也不认定543无flag都干净。0086maxtriangle8.547e-6无flag。
4. adapter计算精度严格单变量4caseGPU6诊断完成163.388sec/worker2379332exit0、GPU恢复。normal vs仅adapter BF16autocast：全参数id/dtype/hash保持、条件/noise/supporthash相同。latentrelativeL2~4.99%/4.30%，宏观height仅+.02616%/-.06365%；可信0086原+8.55%高偏差没有因此消失。不能归因此小变量为主要高度原因，但不排除局部细节、DINO精度、其它训练算子与历史量化。0fab仅技术同输入对照，不再质量GT结论。父逐artifacthash与controlhash复核passed。

## 下一步已准备

短SS目标帧A/B诊断候选manifest与standalone runner/10CPUtests，未torchCUDAconstruct/未训练。只准父审批0086，0fab永不准入此plan。cached生产预处理RGB512相同，原厂SS1.3B rank8alpha8/AdamW1e-4/batch2split1/seed42同，仅目标legacy反例 vsproper共帧改变。单assetdataset两个重复drawslot明确不伪第二asset。约10GB两arm50step live+EMA（fullmodel文件），预留32GiB。正在把首阶段改50steps停点而非100持续，父评审后才能生成hashbound诊断scope批准。不得批准全548或把短weights投产。

目前已具备0086数据/坐标/图像语义的小范围依据，模型现成形状不必完美才可测试目标定义learnability；但原资产污染必须排除。下一轮先检查phase50接口与CPUtests，审查绑定证据，执行严格预算单asset两arm50步，之后SS-only评估并决定是否继续，不盲开20k。

新增输出：diagnostics/quality_followup_20261013/{semantic_anchors,adapter_precision,geometry_asset_scan,short_ss_pilot_plan}。原源码hash与上轮最终manifest一致，未修改旧data/cache/weights/PERM/NJ；所有已完成worker结束，短pilot尚未开始。详细报告与图在本地remediation目录，目标未complete/blocked。
