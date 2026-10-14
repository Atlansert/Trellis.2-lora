# TRELLIS.2 质量续查与受控短训练结果

## 总结

已完成本轮继续工作的具体目标：核对语义与相机关系、定位源几何污染、做计算精度对照、在通过单资产数据前置后执行两组50步短训练，并验证实际SS预测对shape几何的影响。

**正确共同帧目标有效，但项目尚未达到生产质量验收。** 单资产proper50权重只供诊断，停止在50步，不继续100，不替换原生产checkpoint。不把留出图无主轴崩溃当泛化通过。

## 一、真实问题的进一步定位

### 合法视角差异不等于镜像

同yaw渲染相机在共同raw+X、朝-X、up+Z，screen-right=raw+Y=-formalZ。原固定profile用formal+Z为屏幕右，和条件图从另一侧看车，所以鼻端左右相反不能认定模型镜像。

同相机完整三角投影中，0086条件/GT/官方/fixed鼻向一致，车顶向上、轮组低位相容；历史transforms仍为占位，未认证历史真实外参。跨资产nose符号尚未统一，不做任意镜像或车头规范化。

### 参考照片板混入真实几何监督

0fab、f50d、5d4a三个资产的两个大三角节点及实际贴图已审查。贴图为白底火车参考照片atlas，alpha全255，尽管材质标BLEND。导出只留vertices/faces，参考平面进入体素/shape监督。

全548native候选CPU几何扫描发现5个风险对象。其余两个只为几何代理风险，不能直接判污染；543未触发也不是全部干净。原资产未删改。

仅给0fab另建移除明确Fromt/Side节点的候选，世界几何/UV/贴图重载验证通过。保留的1435/Cube.001规则块边长1.435原生单位，实质改变包围盒及归一化，强支持尺度/轨距辅助块假设，但作者语义未确认，所以新候选也不批准训练。

### Adapter分支精度不是宏观高度偏差主因

四case单变量normal vs仅adapter bf16autocast，权重值/Parameter身份/dtype、条件/噪声/支撑均相同。latent相对L2约4–5%，但可信0086高度只变+.026%，原约+8.55%高度偏差未消失。不能据此排除局部细节、DINO精度、其它训练autocast或旧训练量化；0fab只保技术对照，不继续把污染GT作真实车体真值。

## 二、受控短训练：仅第一资产

独立审批绑定manifest与几何/语义证据，只允许0086bbab37559442单资产、两组、首阶段50步。旧目标组是明确的反例，不是旧frame生产批准。

两组均原厂SS1.3B、rank8/alpha8、AdamW1e-4/wd0、batch2/split1、seed42、CFG.1/logitNormal、AMPbf16且adapterfp32。cached生产RGB512条件完全相同，仅目标mean坐标帧不同。单资产两个重复draw slots有明确声明，未伪称第二资产。

legacy/proper各149.2/148.1秒exit0，无超时；GPU6恢复原占用。各完整live/EMA/misc组提交SHA/size全校验，300adapter全fp32、640冻结tensor精确等于原厂基座、0非finite。训练后Python/NumPy/TorchCPU/CUDA RNG逐项相等，Adamstep50/expavgfp32。

|对训练正确共同帧目标的占据IoU|值|
|---|---:|
|原厂SS|.444924|
|旧目标训练live50|.637842|
|正确目标训练live50|**.866728**|

proper输出2057体素、spans[8,64,10]，GT2033体素。没有旋转或拟合对齐。EMA50仍接近初始值（.9999约99.5%初始参数残留），不作为短期主排名，也没有降低EMA。

001/004留出SS主轴仍index1，但004深度跨度从10变8；没有匹配GT，不能认定保真改善或无退化。已停止50，不自动续100。

## 三、真实预测SS进入shape：2×2对照

使用训练资产0086及留出004，各做SS官方/properlive50 × shape官方/fixed20k，共8case。不是正确GT支撑替代实际预测。SS同条件/采样前RNG；同一个SS/input的两shape分支coords/noise/CPU+CUDA RNG完全相同。不同SS输出token数不同，不声称跨SS noise数组相同。

GPU6 worker161.98秒exit0，CPU指标及全三角samecamera图62.78秒exit0，32几何产物SHA父代理独立复核。无纹理或正式GLB remesh，不把geometryPLY当生产验收。

训练GT高度.1412665、宽.1201159（长度约1）：

| SS / shape | 高度 | 宽度 | sampled surface KD距离 |
|---|---:|---:|---:|
|官方 / 官方|.149256|.141478|.00805667|
|官方 / fixed20k|.147732|.138939|.00814987|
|proper50 / 官方|.147839|.123330|.00392536|
|proper50 / fixed20k|**.143364**|**.118011**|**.00366501**|

proper50/fixed组合高度误差约+1.48%、宽度约−1.75%，说明SS目标拟合改善可以传播shape。轮组局部仍有差异，单资产不等于正式质量通过。KD距离是各20000表面样本近邻，非精确三角距离，noICP/no重新居中/no缩放/no最佳旋转。

此前GT支撑fixed8.55%高度偏差与本次预测支撑不是相同输入，且本次训练cached512条件再生成1024不同于原始高分辨率条件；不可跨实验宣称单变量解释。确有SS支撑、shape权重与条件口径交互，不是fixed权重输出固定高度偏差常数。

004四case高度.114259/.115103/.117081/.119308、宽.132004/.129226/.102620/.101509。proper50更窄、稍高，主轴偏角均<.6°，没有匹配GT仍不能判泛化。不得以主轴z或低Chamfer将单资产权重推广。

## 四、本轮额外工程修复

评价第一次错误为low-vram decoder留CPU而targetz放GPU；第二次是再次LoRA注入导致adapter内部linear被重复包装。失败日志保留，已完成case不重复采样。

对apply_lora_to_linear_layers做最小幂等修复：排除已有LoRALinear整个子树（包括根）。首次架构、forward、state键不变；不更改既有权重。5针对测试+全套**131CPUtests通过**。

## 五、后续数据计划与未通过门槛

共同帧cohort计划已建立：548源GLB全部实际SHA、1096条件frame/imagehash验证，5风险隔离，543仅provisional_source_candidate。**ready_targets=0，training_approved=false。** 没将旧shape features或旧SS mean链接为新数据，不置换已学latent假装重新编码。

下一阶段应按源场景语义白名单审核，统一proper共同mesh，再重建shape/SS支持和VAE目标。新target鼻方向规则需要资产级语义，不由PCA浮点或掩膜宽高比自动认定。PBR完整纹理链还未按新帧重编码。

本轮短训练证明方向有效，尚未解决整个项目质量。后续可扩大少量已审核实体资产做同数据集受控训练和正式四图seed42验收，不能直接对543启动长训练。

## 六、交付与安全状态

远程全部新增证据在diagnostics/quality_followup_20261013，旧data/cache/checkpoints/sourceassets/NJ保留，GPU0–3未用，ComfyUI未动。唯一生产代码新增修改为LoRA幂等修复（有before备份），此前工程整改保留。

单资产proper50live位于short_ss_proper_50/ckpts/denoiser_step0000050.pt，**DIAGNOSTIC_ONLY_NOT_PRODUCTION**。两组50完成标记、SSv3评价、预测SS→shape八case、semantic/cards/Cube审查、精度/几何扫描及cohort计划均有独立manifest/hash/log/退出状态。

结论：本轮继续工作的具体诊断与受控短训练已完成；生产质量/全数据训练批准仍为否。没有继续100或20k，无自动清理旧产物。
