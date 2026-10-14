# 六资产小批量数据重建、短训练与正式验收结果

## 结论

已按顺序完成：源资产审核 → 新共同帧几何 → Blender真实透明重渲 → shape/SS同源重编码与独立往返验证 → 四训练/两留出50步训练 → SS评价 → 外部四图正式GLB验收。

**数据重建流程通过小批量验证；模型未通过生产验收，002仍错轴。决定不将此次live50投产、不直接扩大到543资产长训练，不自动续100或20k。** 原文件、旧缓存、历史权重与结果保留，没有使用GPU0–3或停止ComfyUI。

## 一、审核与重渲

六个源GLB实际CPU场景加载、世界几何、UV/嵌入贴图和依赖检查通过，均单mesh节点、没有已确认参考照片板或独立辅助节点。四个yaw0、两个原生X长轴资产yaw90：0086/NZJ03、008ae/CR400BF21、015de/CR200J1-04、0390/CR200J-B05、1208/CRH6A0439、1f7c/CRH5EFront。只批准本批审核范围，不代表543候选全干净。

Blender4.5.1实际CPU512×512/samples16/threads8，每资产8相机，共48真实RGBA。源几何按proper共同raw(depthX,lengthY,upZ)归一化，所有camera保持同canonicalgeometry；没有旧SScyclicperm或每view旋转target。

首渲38.42秒，其余37.07/38.57/39.32/35.86/34.27秒，每stageexit0、无timeout。保存真实camera-to-world、inverse、intrinsicK、right/up/forward、投影检查点、源和审核hash、几何及每PNGhash。

父全48hash/alpha/nonfull/no裁边/camera矩阵与K重算通过。首独立完整1495340三角形投影0/180与strongalphaIoU.96345/.96365，alphaoutsidegeometry0；投影点最大误差5.68e-14pixel。最初相机Float32容差1e-7过紧的失败记录保留，采用相应1e-6容差后通过，不是相机错误。

父实际read_image检查六主侧视，白body、鼻、窗、roof、轮组保留，没有旧颜色差分删部件。透明alpha直接来自渲染，不再使用最大连通域背景差分。光照偏平且历史RGB像素等价未认证，此域差异是后续实验边界。

## 二、同源shape/SS重编码

新渲染器导出的共同帧canonical_geometry真正重建1024双网格，官方shape encoder posterior(samplefalse)→decoder，随后从shape64支撑编码SSmean。未转置旧latent假装重编码。

|对象|shape64支撑数|shape对称平均surface距离|独立SS往返IoU|
|---|---:|---:|---:|
|0086|2033|6.31×10⁻⁵|1.0|
|008ae|2848|4.36×10⁻⁵|1.0|
|015de|2476|5.12×10⁻⁵|1.0|
|0390|2882|5.24×10⁻⁵|1.0|
|1208|2218|4.37×10⁻⁵|1.0|
|1f7c|2167|5.44×10⁻⁵|1.0|

GPU6实际shapeencode31.76秒/decode97.10秒，SSCPU20.79秒/finalize13.23秒，全exit0。独立CPU23.8秒重新解码全部六SSmean、完整hash和dtype/finite/coords集合检查通过；独立精确三角面积矩与source/decoded一致，最大相对尺寸误差.09054%。surface距离为每方向512样本到精确最近三角，独立审查绑定hash但未重复最近三角距离计算。共享shape/SS支撑相同是构建保证，不冒独立体素算法检验。

此数据门槛只允许本六资产view000小训练，其余七视图audit；没有生产批准、没有全543targetready。PBR纹理目标没有重编码，完整纹理训练链尚不在本批范围。

## 三、受控训练

固定train4：0086、008ae、015de、1208；internalholdout2：0390、1f7c。新渲染RGBA按实际pipeline裁剪/alpha乘RGB/量化/LANCZOS缓存RGB512，AST像素一致测试通过。仅view000，留出不进入dataset采样。

原厂SS1.3B起点、LoRArank8/alpha8/dropout0、AdamWlr1e-4/wd0、batch4/split2、seed42、CFG.1、logitNormal(mean1/std1)、AMPbf16保adapterFP32，配方固定。独立审批绑定targetgate、视觉、split、source/manifesthash，限制首50steps，不自动续训。

实际GPU6worker2458885，168.85秒exit0、无900秒timeout，50步停止；live/EMA/misc完整组SHA和size校验通过，300Adamadapterstates为FP32/step50，trainable4669440。GPU6回528MiB/0%。快照按skip不采后丢图，最终完整marker提交；训练不是配置准备。

## 四、SS评价结果

20次实际SS采样：原厂和live50各六新资产及外部001–004。输入条件、采样RNG与noise哈希对比一致，无shape/tex/纠轴。全checkpoint300adapterFP32、冻结基座精确等官方，无非finite。评价162.71秒exit0。

|范围|原厂平均IoU|live50平均IoU|原厂最小|live50最小|
|---|---:|---:|---:|---:|
|训练4|.471929|.484014|.284793|.308475|
|留出2|.387565|.414222|.259843|.328092|

单对象并非一致改善：015de训练IoU .70559→.52461下降；1f7c留出.51529→.50035下降。不能只选平均增益或最优资产。外部002仍rawdim0最长，PCA约88.61°，没有matchedGT所以其它外部只方向/截面代理。

固定SS投影显示部分车鼻方向和目标不一致。另只做CPU hypotheticalyaw180/reflection诊断，几个对象假设180°后IoU更高；此数值不能当验收或用于旋转成品，所有正式指标仍直接原坐标。共同长/up规范不等于跨资产车鼻正负语义已统一，需要后续明确策略。

EMA50尚未热身，不主排名或擅自降低EMA。

## 五、四图正式GLB验收

固定pipeline1024/seed42/stockshape decoder/fixedshapev2EMA20k/新SSlive50，纹理及正式remesh全部实际运行，无ss_canonicalize/ss_swap_yz、无快捷GLB。

|图|运行秒数|成品bytes|最长轴|精确surfacePCA到z偏角|z/y|x/y|
|---|---:|---:|---|---:|---:|---:|
|001|338.26|69020004|z|.17027°|6.17896|1.43227|
|002|362.89|61937076|**x**|**88.77125°**|1.28078|5.40379|
|003|243.94|65252560|z|.97170°|5.15554|1.09690|
|004|246.75|60562720|z|.34958°|8.69179|1.12052|

四workerexit0/no600timeout/ended。父和独立CPU工具复核run_manifestv2实际参数、源模型路径、input/outputSHA，新的四图provenance全部verified；旧33k四图仍保provenance_unverified。全部三角形固定侧ZY/顶ZX轮廓图未对齐/挑角/镜像。

**002不满足原始最长轴z要求，整组acceptance=fail。** 001上视更宽、003鼻背/轮廓有差异，004变化小，不能凭z/y或表面平滑给几何不劣通过。几何notworse仍未批准，未把snapshot/SSIoU当GLB验收。新几何degeneratefaces0也不能代替质量。

## 六、决策与下一轮方案

1. 不将此次live50替换生产权重；保持独立diagnostic标签，停50。
2. 不直接对543候选开启长训，也不依据002错轴马上全局交换坐标。
3. 数据重建流程可继续复用：资产审核、真alpha/真实camera、同源shapeSS编码和gate已实际验证。
4. 扩数据前先明确车鼻正负策略/姿态条件，检查当前低IoU对象方向是否是主要失配；不把假设180°比较变为验收纠轴。
5. 在不混入新对象变量的前提下，下一受控实验可分阶段增加训练时长/同语义侧视覆盖，并保internalheldout与外部002停止条件。只有有实际泛化及四图正式验收证据后才扩大训练规模。
6. 中性环境图与实拍照片域差、少量样本覆盖与旧shapecheckpoint混native训练帧兼容仍需实验，不能将任何一项未经对照宣称唯一根因。

## 七、交付与安全

所有新文件在/data/lzx/TRELLIS.2/diagnostics/data_rebuild_batch_v1。源资产、oldtrainingdata/cache/checkpoints、历史结果、nju未改；未使用GPU0–3，ComfyUI未停止。新batchsixrender48+encodedtargets6+scopedtrainingcheckpoint+评价20case+formalGLB4和全部sourcehash/log/pid/exit完整保留。

生产代码全套131CPU回归通过。本批新增准备测试分别renderer18、encoding15、pilot18、SS评价10，正式验收工具15；不把不同目录重复测试加成虚构统一suite总数。

这轮按顺序的小批量执行与验收已完成；**模型质量未通过，是否扩大训练的决定为暂不扩大**。数据flow成功不是模型达标，新batch权重不是生产发布。
