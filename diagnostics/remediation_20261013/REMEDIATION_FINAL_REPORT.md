# TRELLIS.2 顺序整改最终报告

## 结论

已按用户授权顺序完成本轮工程修复、数据隔离与坐标验证，并完成两资产单变量推理兼容对照。**短训练未执行：技术兼容仅在两例给定正确支撑下通过，语义几何门槛未通过，训练批准仍为 false。** 这不是项目模型质量已经达标的报告，也不是把未通过门槛标作成功。

原数据、缓存、历史配置、checkpoint及旧结果保留；未修改 nju、未 reset/checkout/commit，未使用 GPU0–3，未停止 ComfyUI，未用纠轴掩盖成品问题。修改已有源码前均已阅读与备份。

## 一、阶段执行结果

| 阶段 | 实际结果 | 范围与限制 |
|---|---|---|
| 数据检查与隔离 | 完成1182对象/28368帧扫描；633derived隔离；548native/1096配对候选 | 85风险对象不等于全部坏；未把未触发风险的derived自动批准 |
| 规范帧契约 | 新目标与接受输入一致，最大偏角3.037°；哈希/数组核验 | pose-correct候选不等于完整物理帧正确，默认拒绝用于训练 |
| 坐标链 | 确认旧SS与shape监督不同帧；高度符号和手性真实标记验证 | 未修改旧PERM，不用旋转成品修复验收 |
| 共同帧VAE pilot | 两资产真实1024shape重编码/解码，匹配SS目标重编码成功 | 两例，不是全548shape/PBR新数据完成 |
| 加载与推理可靠性 | 严格路径/键/LoRA/eval，失败非零，实际运行manifest+阶段计时 | 历史产物仍缺完整来源证据 |
| 成品评测 | 四图正式GLB齐；精确表面积矩/PCA、来源哈希与人工几何三态 | 四图几何未通过；004倾斜 |
| 训练恢复与保存 | FP32 LoRA/complex buffer保护、完整组提交、latest/retry/skip/finalsave | CPU回归，未进行大模型/DDP训练；worker预取不保证精确轨迹 |
| 单变量对照 | 同条件/支撑/噪声官方 vs fixedshape20k两资产推理 | 有限技术兼容通过，语义几何质量未通过；不批准短训练 |

## 二、数据整改

全量633derived的15192帧、549native的13176帧覆盖，37原子chunk与完整标记。读错误、空/整幅alpha、存储RGB不一致及合成算法重算不一致均为0。85derived对象触发风险；207帧最大连通域丢弃超过10%的颜色差分掩膜。补alpha丢白车身、绿车前部的具体实例已视觉复核；风险是候选，不是掩膜真值。

保守隔离全部633derived，加1无合法输入native对象。548native/1096pair候选目标最大偏角3.037°。旧131目标超10°训练对象中，46native目标已纠正，85derived隔离。目标选帧以独立PCA姿态约束为准，近似等角候选稳定选原序号，不使用最大掩膜宽高比或任意000回退。

启用canonical_manifest时，缺训练批准默认报错；批准需绑定坐标及掩膜审查证据。诊断显式opt-in才能读取未批准候选。初始化验证源图/体素/latent/metadata/transforms/target哈希与完整对象/视图集合；采样时检测帧、图、目标后续变化。旧配置1181对象/2362pair与getitem回归保持。

## 三、旧坐标问题与共同帧验证

全部548候选的SS目标lengthdim1，旧shape训练支撑lengthdim2=507、dim0=41，dim1=0。直接支撑IoU中位仅.033996；48种带符号轴置换诊断后519对象IoU精确1，证明主要是帧差而非单纯体素化误差。该诊断没有用于旋转生成结果。

实际Blender glTF轴标记导入为 native(Dx,Dy,Dz)->raw(Dx,-Dz,Dy)，det+1。旧SS yaw0=(-Dy,-Dz,-Dx)，native向上的Dy进入negativeSS0，正式GLB仍落-x，不是+y。真实稀疏C2S四级8192子节点均继承同轴父cell，不能学会把整辆车从一轴搬到另一轴。

新增共同帧 helper 只供新数据：raw lengthY/upZ、正式GLB lengthZ/upY，properdet+1。两非对称资产从原几何重新编码，不转置旧SS mean/shape feature：

| 指标 | 0086… | 0fab… |
|---|---:|---:|
| 1024dualgrid体素数 | 581866 | 720880 |
| shape64编码支撑数 | 2033 | 2610 |
| shape VAE对称平均表面误差（长度约1） | 0.00006409 | 0.00004411 |
| 源/解码精确表面主轴偏角 | .74903°/.74310° | 1.18178°/1.18785° |
| 匹配新shape支撑的SS解码IoU | 1.0 | 1.0 |

0fab的独立Open3D支撑有2618，不能沿用其旧编码mean假定匹配2610；本轮已从真实shape编码支撑重编码对应SS mean。共享支撑一致属于构建保证，独立体素化IoU另行记录。

## 四、成品与兼容对照

axis10 EMA20k、seed42、pipeline1024、固定shape20k、不纠轴：四图正式成品已齐。新增003在严格加载+外部600秒保护下206.894秒完成，输入与输出SHA已独立核验，SS/shape/texture采样与导出分阶段日志齐全。003成品62,472,288B，SHA 1a3792653f40a8af92fc80a07ad7bc533b611c69f0d8d616bab48a62087d3c0c。

| 图 | 成品z/y | 精确三角表面积矩主轴到z偏角 | 来源证据 | 几何不劣于旧基线 |
|---|---:|---:|---|---|
|001|5.42290|2.83130°|历史证据未完全验证|未验证通过|
|002|3.91122|1.09703°|历史证据未完全验证|未验证通过|
|003|4.06613|.88982°|新v2运行记录验证通过|未验证通过|
|004|4.38010|10.17357°|历史证据未完全验证|未验证通过、明显倾斜|

10°仅是诊断参考，不是擅自增加用户生产门槛。主轴z/较接近照片的AABB比例不能证明几何质量。面积加权面心PCA存在三角细分偏置，现保留旧代理并新增精确三角面二阶矩作为主诊断，相关解析与细分不变性测试通过。

两资产的shapeFlow对照没有训练，正确GT支撑、条件、noise、seed全部hash一致，唯一变量official vs fixed20k权重。GPU6 worker112.658秒exit0，恢复原占用。固定shape20k车高相对GT分别+8.55%/+7.15%；第二例官方与固定权重的鼻部/轮组均不如GT。采样点KD双向距离略下降不能代替几何审查。结论：有限技术兼容pass、语义几何not_passed、training_approved=false。

## 五、实际代码修复与测试

本轮既有源码修改：
- [数据集](</data/lzx/TRELLIS.2/trellis2/datasets/sparse_structure_latent_views.py>)：可选不可变canonical契约、未批准默认拒绝、采样变更检测。
- [基础模型加载](</data/lzx/TRELLIS.2/trellis2/models/__init__.py>)、[pipeline加载](</data/lzx/TRELLIS.2/trellis2/pipelines/base.py>)：严格键与明确来源，不广泛fallback掩盖损坏本地文件。
- [推理入口](</data/lzx/TRELLIS.2/scripts/infer_finetuned.py>)：严格显式路径与LoRA150目标/rank、eval、原子运行记录、失败非零、阶段计时、快捷产物坐标统一但不计正式验收。
- [trainer](</data/lzx/TRELLIS.2/trellis2/trainers/basic.py>)：恢复精度/complex buffers、同步整组保存与完整标记、RNG范围、skip snapshot、最终step保存。
- [训练入口](</data/lzx/TRELLIS.2/train.py>)：保护旧输出，完整组latest、重试只续完整提交点、耗尽非零、独立attempt记录。

新helper包括规范帧构建与坐标定义、数据加载契约、推理契约、固定坐标评测、checkpoint契约和训练启动契约。已有源码before副本、最终after SHA、精确diff、每次测试和运行日志保留。

**126项CPU测试通过**，包括真实BasicTrainer方法AST执行的tiny保存/恢复、下一步loss与adapter更新bitexact；实际推理003验证大模型加载与导出路径。没有伪称大模型训练/DDP已回归。

## 六、尚未解决与运行限制

1. 633derived真实掩膜尚未重新生成，只隔离；全548共同帧shape/PBR数据未重建，仅两资产pilot。
2. 当前shape20k在新帧下两例有限技术可用，但局部几何不佳；四图模型未通过完整验收。
3. 条件图到统一车鼻/侧向的语义规则仍需明确；图像左右相反不能单凭预览判为模型翻转。
4. 训练/推理DINO与adapter算子精度差异未量化为质量因果，不用它作为未经验证的退化解释。
5. 恢复修复不能逆转历史bf16更新；EMA fp32不证明旧live一直fp32。DDP全rank RNG及workerprefetch未保存，不能保证轨迹级exact。
6. 同步CPU克隆与全哈希增加内存和I/O；大模型/GPU/DDP保存恢复需以后在批准的短训练中验证。
7. PBR完整纹理链尚未按共同帧重编码；当前geometry/SS pilot不把此项伪装为已经完成。

## 七、下一步决策

本轮不盲目开启新的20k，也不以纠轴产物作为验收。后续建议先确定统一车头语义与新共同帧数据规范，扩展少量真实资产与条件质量审查；再明确固定shape20k的保留、重新适配或官方prior对照方案。受控短训练需要新的数据版本、相同对象/条件集和明确阶段停止条件，而不是把548子集与原1181run当单变量因果比较。

当前gate记录为false，未发布TRAINING_APPROVAL，未进行权重更新。工程整改与验证任务完成，但**模型质量问题仍未解决**。

## 八、交付

[源码变更与证据索引](</data/lzx/TRELLIS.2/diagnostics/remediation_20261013/REMEDIATION_FINAL_MANIFEST.json>)、[本轮精确补丁](</data/lzx/TRELLIS.2/diagnostics/remediation_20261013/REMEDIATION_FINAL_SOURCE.patch>)；各阶段报告、测试、manifest、计时、四图评测以及共同帧和兼容对照图均在新的诊断目录。

所有本轮worker结束，GPU4–7恢复528MiB/0%查询状态，常驻服务未变。原结果与旧证据全部保留。
