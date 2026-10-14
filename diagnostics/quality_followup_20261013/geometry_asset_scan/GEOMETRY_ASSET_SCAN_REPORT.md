# 原生alpha候选548对象：源几何风险筛查

## 范围与执行

CPU readonly，CUDA_VISIBLE_DEVICES空，无模型/GPU/训练/安装依赖。集合来自canonical_contract_v2_native_staged manifest中的548 admitted对象，不等于已获训练批准。读取本地trusted项目mesh_dumps pickle，每对象dump完整SHA256+bytes保存，按4CPU线程、32对象分块，18个数据chunk及18个SHA sidecar原子保存。500秒自然预算，实际一段96.55秒完成；worker2381217已结束，退出0，没有子进程遗留。最多同时4对象，face计算65536批次，不把548资产一起载入内存。

custom_prepare_data.py:164–192已下载/read确认dump仅objects.vertices/faces，无材质、UV或alpha语义；源trimesh几何将原资产归一化不改变轴。该筛查只几何风险，不判断真实渲染可见性。没有修改旧data/cache/nju/weights/source或创建训练批准。

## 完整结果

548对象，读取/计算错误0，5对象触发风险；最大三角面积归一化阈值>.005有3对象、最大face各轴跨度/full该轴跨度>.5有4对象，重合2；整体顶点采样PCA薄度sqrt(lambdaMin/lambdaMax)<.02对象0。后者不能排除真实列车内部存在薄参考卡片。

最大三角面积（按最长轴归一1）分位min/Q25/Q50/Q75/P90/P95/P99/max：3.7043e-6 / 5.7071e-6 / 6.6849e-6 / 8.0548e-6 / 9.9132e-6 / 1.1240e-5 / .00044994 / .07360315。

最大face跨度比例分位min/Q25/Q50/Q75/P90/P95/P99/max：.02505 / .03304 / .03616 / .04056 / .04470 / .04856 / .22849 / 1.0。

## 需材质/节点审查的5对象

|对象|最大tri面积归一值|占总面积|最大face跨度比例|轻量GLB JSON线索|
|---|---:|---:|---:|---|
|f50d6a143bdfe293|.07360315|12.1029%|1.0|CR400BF/CR400BF01；mesh CJ5-0506-Side Front与.001；BLEND+OPAQUE|
|0fab4af3f9a73bdd|.07236333|11.7045%|1.0|Fromt/Side；同CJ5-0506-Side Front；BLEND+OPAQUE|
|164d20b761652694|.01315625|2.1210%|.48673|CRH2C、Mesh.002；多材质含BLEND，不足判断参考卡片|
|3d80f1c0d8bebe55|.00340778|.5971%|.58326|晶格/Bogie/1435mm，材质OPAQUE；只几何跨度proxy|
|5d4a5e8e46991642|.00105326|.2076%|.64592|Front/Side、Plane/Plane.001；BLEND值得参考图card审查|

0086bbab37559442对照：878866vertices/1495340faces，最大tri8.547217e-6，占总面积.0018468%，最大face跨度.053366，不触风险。0fab已知最大面积与父semantic调查相吻合，扫描没有重复读其大纹理或材质深入细节。

仅5 flagged GLB读取JSON chunk（节点/mesh/material名称与alphaMode），不读BIN/图片/UV缓冲。f50与0fab共享reference名强线索，5d4有Plane/Front/Side候选，仍必须结合实际primitive/材质/texture才可判referencecard。大三角也可能真实结构板，BLEND也可能车窗，node名字不是地面真值，禁止自动全部删除或认为548-5全干净。

## 协议与限制

记录最大tri顶点、几何面积、退化面数、绝对face法向30bin、顶点采样PCA仅proxy；未算连通片，按任务范围保留为后续topflags审查。全对象薄度比不能发现隐藏在整车中的几何照片卡；face跨度比例按每个全资产轴归一，横向短轴跨度>.5也可由真实端盖引发。

resume校验immutablecontract/metadata/scriptmanifest与chunk哈希，但不重读先前完成chunk的源inputs；每源dump完整hash记录当前读取bytes，长期续跑须再次源hash核对。本次一段完整扫描，原资产未修改。

COMPLETE仅证明覆盖与文件一致性，不代表质量通过。training_approved=false。records.json/risk_candidates.json/summary.json/manifest.json的完整SHA在COMPLETE，已逐一现场独立验。PID及exit/log同级diag父目录保存。

## 交付

Remote：diagnostics/quality_followup_20261013/geometry_asset_scan/{records,risk_candidates,summary,manifest,COMPLETE}.json；同目录本报告；脚本diag父geometry_asset_scan.py，segment1.pid/log/exit_code位于父目录。Local主要报告与summary/risk与脚本在trellis_context_20261013/remediation。

建议下一步仅审查5风险资产的primitive/material mapping及是否建模参考照片card，使用新版本净化/排除规则保原资产与材质。扫描本身不创建训练批准，也不能解释全部shape退化因果。
