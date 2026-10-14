# 顺序整改第1轮执行结果

## 状态

用户已授权依次执行。目标仍active，未完成。先前审计只读承诺已转为用户授权修复，但每处既有源码修改前已读/备份。没有gitreset/checkout/commit，没有修改nju/原数据/cache/权重，没有训练。

### 第一阶段：数据风险覆盖与隔离完成

全1182×24=28368图片；633derived15192+549native13176，37原子chunk可恢复，readerror/empty/full/存储RGB不一致/derived算法重算不一致皆0。85derived风险对象、207帧最大连通块丢>10%；风险不是损坏真值。新增明确可见绿车切掉前部与白车丢body对象；所有633derived仍隔离，不批准unflagged直接训练。

staged v2：548native1096pairs，目标PCA≤3.037°、全longindex1。旧131中46native修目标、85derived隔离。所有target源属于接受输入集合，哈希与mean数组独立验证。

新增dataset canonical_manifest契约：启用后默认拒绝无训练审批，诊断需显式allow_staged；审批需绑定passed坐标/掩膜证据hash。初始化验证依赖文件与exactobjects/views，getitem检测帧/图/target后续变更。17CPU测试通过（非20，曾因fixture class重复计数已修正），旧1181/2362与3getitem抽样回归通过。

### 第二阶段：完整轴链诊断完成，训练门槛未通过

548sameasset shapeTRAINsupport long2=507,long0=41,long1=0；SStarget全long1。identitysupportIoUmed.033996；48signedperm诊断maxIoUmed1/min.9324，519exact1，524uniquebest，所有best存在det-1。没有按bestfit修改生成结果。

真实SparseChannel2Spatial4levels8192leaves父轴保持（两个probe错误已定位修正：scriptPYTHONPATH、dummyfeatures每level须divisible8；不是项目module错误）。实际Blender4.5.1导入nativeglTF标记验证(Dx,Dy,Dz)->(Dx,-Dz,Dy)det+1。现SSyaw0=(-Dy,-Dz,-Dx)：up是negativeSS0且formalGLB落-x，不是+y。两真实asset原GLB/dump顶点匹配、固定support图落盘。

新增canonical_frame.py只为NEW数据定义共同proper raw长度y/upz；5CPUtests。未接入旧encoder/pipeline、不改PERM、不转置cachedmeans。必须共享SS/shape/PBR帧并重新encode小pilot验证；不能声称只修SS与固定shape20k必兼容。TRAINING_APPROVAL没有发布。

### 第三阶段：可靠性与成品验收进行中

base pretrained加载strict校验（允许计算rope_phases重建，禁止其他missing/unexpected），缺本地component不fallback到其他权重；明确跨repoSSdecoder不再错误checkpoints/microsoft先尝试。7tinytests通过。

infer严格显式路径、LoRA150/rank/key/eval，失败nonzero、batchpartial失败，v2原子runtime manifest记录input/output/weightsha、sampler/flags/阶段timing，rawGLB/PLY同propercoord但notformal。28CPUtests通过；deadline需外部supervisor兜底。

新003真实正式推理：results_infer/axis10_20k_003_remediation_20261013，seed42/pipeline1024/shape20k/decoder/SSaxis10EMA20k，无swap/canonicalize。外部600秒界限，worker2351894于206.894s结束exit0，GLB62472288B，sha1a3792653f40a8af92fc80a07ad7bc533b611c69f0d8d616bab48a62087d3c0c。输入/输出hash独立复核，pipeline_load89.004s，SS1.954/shape2.595/texture1.863/pipeline_run9.467/export81.606s。GPU4回528MiB/0%，未改ComfyUI。

003CPUfixedframe：extents[.2466765,.2463259,1.0015928] z/y4.06613，PCAoffz.889797°、695935vertices/991098faces，degenerate0。这是必要指标，不是几何保真自动通过。

全部既有GLB11份20秒CPU筛查；004axis10PCAoffz10.173559°（参考10deg不是擅改生产门槛），AABBlongz依然有明显倾斜。验收agent正在增强provenance failclosed：历史无v2runtime manifest不能靠manualreview给总pass；新003验证schema/flags/seed/input/outputsha后输出v3含四图。一个子代理执行意外中断已sendmessage恢复，未重复GPU运行。

## 待执行

- 完成stage3新版证据验收tests/v3report（001002004历史provenance明确unverified，003新runtimeverified，几何review未完成）。
- 依次修resume保adapterfp32/complexrope、原子整组checkpoint与完成marker、retry非零+resume语义、snapshot按skip，不启动全训。
- 小jointframegeometry/encodingpilot是后续前置；未通过不得短训练，当前不应做548staged的20k。

## 当前源码责任分工

父：data_toolkit/build_ss_canonical_contract.py，data_toolkit/canonical_frame.py；trellis2/datasets/canonical_contract.py与sparse_structure_latent_views.py（有beforebackup），models/__init__.py与pipelines/base.py（有beforebackup）；相应CPUtests。

子86fdd：scripts/infer_finetuned.py（beforebackup）、scripts/utils/inference_contract.py、tests/test_inference_contract.py，完成无GPU任务残留。

子17d8：scripts/utils/evaluate_fixed_frame.py、tests/test_fixed_frame_acceptance.py；v2report已在fixed_frame_acceptance/v2，补v3进行中。

子57872：alpha_scan全量chunk/report与独立stage2，完成无残留。

工作服务器目录diagnostics/remediation_20261013，所有原历史artifact保留。日志/证据以工具实际读取为准，当前目标不应complete或blocked。
