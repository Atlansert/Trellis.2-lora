# 顺序整改：数据阶段进度

当前仅执行第1阶段；生产训练配置未改，未启动训练。数据候选尚未获准训练。

## 已完成

- 基线hash复核，原tracked变更未新增修改；保留所有历史数据、权重、结果与nju。
- 新增 data_toolkit/build_ss_canonical_contract.py 与2个tests文件，不替换旧builder。
- 新独立版本 canonical_contract_v2_native_staged：548原生alpha对象、1096输入配对；633derived+1无候选隔离，共634。
- 目标最大PCA偏角3.037°，全longindex1，目标源view属于accepted输入集合。
- 旧131错配：46native目标已修正，85derived隔离；整体64目标源view变化。
- 所有目标与源mean数组严格相等，输入图/vox/latent/transforms/metadata/cache哈希全部独立验证。
- 9单元/集成测试通过，包括覆盖拒绝、路径穿越、错误坐标、NaN角、latent dtype/shape/finite、负缓存、错误阶段无COMPLETE标记。
- 选帧姿态差≤.05°时按原frame最小index，避免相反侧视浮点误差导致180°随机翻向；这不证明跨资产车头方向统一。

## 进行中

- alpha全1182/28368帧扫描，由独立子代理执行8CPU processes、32对象原子chunk，500秒预算自动停止可续跑，覆盖633derived全部15192帧。
- 指标是算法风险信号，不把低coverage或RGB接近背景自动视作真实mask错误。
- 原生alpha也仅是较可信来源标记，不能保证语义mask正确。

## 后续必须验证的坐标线索

同资产0086bbab37559442：SS候选spans[10,64,8]，旧shape1024训练coords spans[8,10,64]，原mesh dump extents[.1201,.1413,.99999]。当前候选只是修输入/目标pose契约，未解决下游训练帧一致。

shape decoder实际SparseChannel2Spatial/SparseUpsample采用newcoords=2*parent+child(0/1)各轴局部细分，当前flex_gemm是submanifold sparse conv，不能任意全局交换support轴。应在stage2用真实资产与显式标记锚点核对所有帧，不能拿longindex1通过当整链正确。

## 版本与审批状态

manifest SHA256 c1c401d7d2ce0c000d32720b9838e2ed64b8243b09242b7a7c2474d0e52e201a。
COMPLETE仅表示数据候选无读写错误，不是训练审批，training_approved=false。
候选构建时源脚本已归档在候选目录builder_source.py，hash对应manifest，后续修正错误阶段COMPLETE行为与关闭句柄没有改候选输出。
不得用此548子集实验声称单变量修复原1181实验（对象集合与target都变了）。后续短对照必须同一集合/其它变量固定，先验证坐标与加载可靠性。

服务器目录：/data/lzx/TRELLIS.2/diagnostics/remediation_20261013。
当前原数据、缓存、checkpoint、历史run配置全部未被替换。
先完成alpha风险覆盖与隔离，再进入坐标阶段；其后依次进行加载/失败状态/验收、resume/原子保存/监控，最后满足前置条件才短训练。
有用代码与证据已落盘，用户的长运行目标仍active，未完成。
