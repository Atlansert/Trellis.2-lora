# TRELLIS.2 当前项目完整审计

日期：2026-10-13（Asia/Shanghai）。远程：192.168.65.39，/data/lzx/TRELLIS.2（与/home/lyx/lzx/TRELLIS.2为同一目录）。

## 0. 总结与决策

当前系统能训练、能加载并生成GLB，但**还没有通过本阶段完整验收的SS新权重**。问题不能归结为“LoRA没有加载”“EMA随机基座污染”或“只有一个全局yz交换”：当前view-side、axis10及固定shape20k检查点的冻结基座和适配器已全量校验正常。真实问题集中于：

1. 条件掩膜质量与canonical目标选帧存在已复核的污染链；
2. 正侧视输入过滤不约束canonical目标，实际有131/1181对象目标不符合≤10°；
3. 只对齐长度轴未证明高/深轴和下游坐标约定一致；
4. SS主轴筛查、loss与表面平滑不足以反映成品姿态/比例；axis10的004已出现明显斜向；
5. resume精度、保存/自动重试、推理失败状态与权重校验存在明确实现缺陷；
6. 发布包、历史报告、当前研究目录和在线目录并非同一运行架构。

推荐先修复验收/数据定义与失败可观察性，再做受控小实验；不直接开启又一轮20k/50k训练、不大规模重构、不换rank/lr/EMA来碰碰运气。

## 1. 审计范围与可信度

- 刷新Git、进程、GPU、磁盘、检查点与推理目录；阅读交接、报告、发布和服务文档。
- 核心训练、LoRA、数据、DINO、SS/shape模型、采样与推理/导出源码已读；引用源码快照与远程sha256核对一致。
- 全量1182对象的canonical数组来源定位、帧表一致性、目标姿态核对；28368逐视角latent参与匹配，未逐一decode。
- 三个EMA20k全量CPU tensor检查，不实例化大模型。
- CPU tiny LoRA模拟resume精度问题；31对象744帧alpha/aspect抽样、31对象38帧预处理对比；alpha并非全量检查。
- 新axis10 SS筛查：2k/10k/20k各四图、seed42，精确生产预处理/条件调用顺序，无纠轴。
- 新正式GLB：axis10 EMA20k的002和004，固定shape20k、decoder、pipeline1024、seed42。001为上一轮已生成产物；003本次仅SS筛查，没有新完整GLB。
- 不修改已有源码/配置/训练数据/权重；不训练、不删除、不reset/checkout/commit、不启动在线服务或第三方付费调用。仅新增审计脚本、报告和推理输出。全部启动进程受管前台；一个全量alpha尝试超过前台超时后终止，不把它报告为已完成。

可信度分类：A=实际文件/代码/重算事实；B=支持机制但未因果证明；C=历史报告，不作本次重跑结果。

## 2. 当前目标、架构和资源

目标：data_test/001–004，seed42，pipeline1024，不开ss_canonicalize/ss_swap_yz，成品GLB长轴z，几何与比例不劣于旧33k基线。多种子仅参考。该目标未达到完整验收。

研究链路：RGB -> BiRefNet或已有alpha裁剪 -> DINOv3@512 -> SS原厂1.3B/rank8 alpha8 LoRA，SS16³×8 -> SS decoder64³ -> shape flow v2 EMA20k（DINOv3@1024）-> shape decoder -> tex flow12步 -> remesh/UV/4096纹理 -> GLB。纹理虽不是当前评估重点仍实际运行；--vae_only并不关闭纹理采样。

两当前SS run都使用单图条件训练：每对象随机选一视角，不是24张图联合输入。view-side：1151对象、15502对、每对象2..20合法视角；axis10：1181对象、2362对、每对象恰好2输入视角共享固定canonical latent。

Git：main...origin/main，HEAD75fbf01，18 tracked文件变化（310插入41删除），大量untracked核心LoRA/配置/脚本/结果。最近commit是2026-06-05上游CodeQL，不代表本地研究版本。没有可复现版本标签/不可变run源码manifest。

磁盘：/data7.0T，已用4.3T，可用2.4T，65%；根879G，可用274G，68%。未发现磁盘耗尽，未进行目录清理。GPU0–3各约134GB由其它任务占据、不使用；GPU4–7约528MiB常驻ComfyUI、查询时0%但不是完全无人使用。新推理使用4、SS筛查使用5，没有改常驻服务。

环境：Python3.10.20；torch2.11.0+cu128；torchvision0.26.0+cu128；transformers5.14.1；timm1.0.28；flash-attn2.7.3；numpy2.2.6；trimesh4.12.2；open3d0.19.0。当前成功推理证明本环境可用，不证明发布包在别处同样可用。

## 3. 当前实验证据

### 3.1 训练/产物

view-side EMA2k..20k共10个，20k训练完成；axis10 EMA2k..20k共10个，训练完成。当前源config与各run保存config相等。canon_side只有配置，结果目录不存在；A′现存EMA到13k，B′到24k。旧33k与shape v2现存各只有相应关键EMA一份，不是旧文档写的完整50档。tex EMA22份到130k。

view-side001已有GLB复测：2k/10k长轴z、z/y6.438/6.467；12k x/y降0.937；14k跨度y=.585,z=.890，虽argmax仍z但显著斜向；16k–20k长轴y。loss下降并未阻止形状/方向崩溃。不能把001结果写为四图都端到端测过。

### 3.2 axis10四图SS筛查（新）

比值E1/max(E0,E2)，>1.5只是主轴筛查明确，不是所有旋转角/形状合格。

| step | 001 | 002 | 003 | 004 |
|---|---:|---:|---:|---:|
| 2k | 4.000 | .250（dim0长） | 4.000 | 6.400 |
| 10k | 4.000 | 4.571 | 4.000 | 6.400 |
| 20k | 4.571 | 3.556 | 3.765 | 4.000 |

20k spans分别[14,64,12]、[17,64,18]、[16,64,17]、[10,64,16]。所有20k长轴dim1，但004仍可能斜向，下面成品证明主轴指标不足。

### 3.3 axis10正式GLB（001沿用上一轮，002/004新生成）

| 图 | 输入RGB前景W/H | 旧33k z/y | axis10 z/y | axis10 x/y | axis10 AABB(x,y,z) |
|---|---:|---:|---:|---:|---|
| 001 | 4.233 | 8.574 | 5.423 | 1.110 | .205164,.184759,1.001930 |
| 002 | 3.222 | 6.372 | 3.911 | .991 | .253864,.256166,1.001924 |
| 003 | 3.928 | 6.052 | 未生成 | 未生成 | SS-only |
| 004 | 5.688 | 8.839 | 4.380 | .641 | .144685,.225851,.989248 |

001、002比旧基线全局瘦长有所缓和，但不能直接判几何改善；004在固定侧视原坐标CPU预览明显倾斜，尽管最长轴z。其z/y缩小可能由倾斜增大y跨度，不是“比例更好”的证据。002还有突起/端部几何问题。完整预览three_acceptance.png固定坐标、同长度尺度、没有纠轴/挑角；近似三角深度排序、不是真实光照渲染或最终评分。

新002运行257.476秒/退出0，004230.855秒/退出0；日志核对三权重路径与输出，PID均结束。001240.495秒。003曾有100分钟remesh历史，因此本次未盲目重复该昂贵环节；后续需要有时间限额/阶段诊断再补。

## 4. 分级问题清单

### P0：当前验收或下一次训练前必须明确

**D01（A，真实数据污染）合成alpha可能抹掉白色车身；已证一对象。**

build_renders_cond_t.py:28–44用边缘RGB中位背景、颜色差>3、closing5×5、只留最大连通块、填洞。在对象3e988013c8136c9b，原始views0/6 alpha全255，补alpha前后RGB完全相等，现存alpha与该算法独立复算完全相等。view0只保留1150像素/bbox144×8；view6只保留12196像素，白色车身与独立车轮在训练实际RGB中消失，主要剩窗和蓝线。extreme_alpha_comparison.png可复查。633个derived对象潜在受此方法影响，但本次不宣称633全部坏。

**D02（A，目标帧不符合过滤定义）131/1181 axis10训练对象canonical源角>10°。**

全1182 canonical mean严格唯一匹配其24 view mean，无失败/重复数组。131个入训不一致对象中130约15°，另一个3e988013c8136c9b为89.966°，source000跨度[10,10,64]，长轴index2；其输入views6/18却各.76°。唯一未入训e2c8cb5a2feac747为18.466°且无≤10输入。

原因链明确：build_canon_canonical_latents.py:23–38按128BOX alpha>32的bbox宽高比argmax，未用PCA姿态约束；极端端视alpha仅蓝线导致aspect18高于正常侧视11.625，错误选端视作为canonical。dataset过滤只作用输入，canonical target原样读取。不能再声称“固定目标恒index1”“axis10所有输入目标都正侧视”。131不是全部错长轴，只有1目标longindex2；其余130角度语义仍错。不能直接用这一比例断言造成整个collapse。

**G01（A/B，轴定义高优先疑点）长度轴对齐未证明高/深对齐。**

生成器声称camera vox0水平1视深2竖直；encoder默认perm(2,0,1)→SS0竖直1水平2视深。实际pipeline:288仅去channel、:308–311原coords给shape，shape decoder fdg_vae:103–108原coords给mesh转换；o_voxel flexible_dual_grid_to_mesh坐标逐轴(coords+dual_vertices)*voxel_size+aabb无swap；GLB postprocess:313变为(x,z,-y)。这条链下SS0映到GLBx，SS2映到GLBy，与标准GLBy高的语义不同。真实shape网络可在upblock生成细节/改变support，不能仅凭链条宣布全部mesh固定roll90；但以前用长轴index1/GLBz证明全部帧正确明显不充分。需要同一非对称资产的车顶/车底/宽度锚点对照，不直接改PERM或开启swap。

**E01（A，验收缺口）axis10尚未四图GLB几何通过；004已明显斜向。**

SS ratio>1.5、AABB argmaxz都允许斜向。AABB z/y在斜向时不是真实长度/高度。不要只看新比值比旧更接近输入就排名。003缺完整GLB；001/002/004需要顶/侧/端固定视图以及截面/车头/底部核验。轮廓IoU选最佳角+缩放、二面角、loss不可代替。

### P1：明确实现缺陷和可靠性风险

**T01（A，已tiny复现）resume把fp32 LoRA与Adam状态降为bf16。** basic.py:344–351全模型to(bfloat16)，AMP master直接引用model参数(:221–240)，Adam load(:364–365)状态随参数dtype。fresh adapter fp32；resume后adapter/master/exp_avg bf16、EMA仍fp32。当前两个run启动ckpt none，不把它硬说成当前失败原因；恢复/继续训练前应修，并回归比较fresh vs resume。

**T02（A）snapshot配置并不实际监控质量。** views dataset默认skip_snapshot=True，当前configs未覆盖；basic.py:850–854/:886–887跳过中间采样；:920却无条件最终snapshot，采完后:586因skip不保存。当前samples只有dataset.jpg切片代理，不是decoded SS或GLB。即使启用generic snapshot也未自动得到朝向/比例验收，宜独立阶段指标。

**T03（A）checkpoint保存非原子整组、latest选择与重试状态不可靠。** basic.py:388–441多线程分别写model/EMA/misc，没有整组完成标记、原子rename/join；misc持optimizer活动引用存在竞态风险。train.py:23–30按misc最高步选latest；当前--ckpt none --auto_retry3每次从0重开并可能覆盖同输出，不等于自动续训。重试耗尽:187–199只打印，不非零退出。未证明当前run发生过覆盖/损坏。

**I01（A）推理显式权重缺失/错误键仍继续，单图失败仍打印完成/退出0。** infer_finetuned.py:70–94/:130–202路径不存在跳过；strictFalse只警告数量继续append loaded；:452–463catch单图error后继续最终完成。models/__init__.py:65–66对基础加载strictFalse返回完全忽略。当前三个关键EMA检查正常不受此实证错配，但自动评测不能只看exit0。

**I02（A）fast_glb/geometry_only与正式GLB坐标/处理不同。** infer_finetuned.py:225–243直接导出raw mesh，不做正式postprocess:313的(x,z,-y)转换，也不remesh/UV；即使文件同为GLB，轴指标不可直接与正式GLB表比较。正式texture采样仍发生在这些快捷导出前，不能当完全shape-only性能。

**D03（A）真实相机外参字段是占位。** 全1182 transforms前24 matrices存在重复，样本全identity+[0,0,2]；custom_prepare_data.py:315–327明确固定matrix，只有yaw真实变化。当前SS仅读yaw/filename，所以不自动破坏当前索引，但不能用matrix进行重投影、相机帧证明或未来多视图几何loss。metadata num_views=0的677对象实际有24+帧，当前SS不用该字段，其他工具可能受影响。

**D04（A/B）训练/推理预处理和条件精度不完全一致。** 训练crop alpha!=0，推理alpha>204；取整、RGBA resize后乘alpha vs先乘后RGB量化resize也不同。31obj38frame CPU像素MAE中位.370/255、最大2.264/255，19/38 bbox有变化。训练DINO明确bf16/autocast、推理默认fp32无autocast；同名DINOv3不代表数值完全一致。尚未证明这是崩溃根因，别为差异直接重训。

**R01（A）研究/发布/在线架构分叉。** 发布README声称开箱包含data_test/assets，但目录实际缺这两项；发布pipeline使用合并旧SS33k/shape20k，不是新axis10；核心pipeline/dataset/infer与研究目录不同，train/lora相同。release infer.sh默认GPU0不适合当前机器资源约束；例子在已合并ft33k基座再注入同LoRA有重复适配风险，merge脚本example还用ft33k当base而arghelp说需未微调base。不能用发布README成功声明替代当前研究验收。

### P2：维护与再现风险

- frame/path全量一致并非所有图几何正确；verify_canon_full实际前400voxel代理，没有验证真实canonical源数组，漏掉上面端视目标。
- angle/aspect caches没有单位/版本/资产hash/transformhash；v4 npz只存coords，存在文件则跳过，无法判断实际生成CLI/混版本。
- generator axis_map(0,2,1)det−1×signs(−1,1,−1)乘积+1×rz+1=−1，“纯旋转/保持手性”数学注释错。但跨trimesh/Blender/native坐标可能需要反射；缺真实历史CLI/外参，不能直接宣称现存资产镜像。
- checkpoint保存全量base+adapter约GB级，不是轻量adapter-only；源码/配置/数据/权重缺不可变manifest与hash关系，易混产物。
- 当前data_for_train1323目录：1182既有metadata对象+137新GLB有transforms+4无mesh；137未进当前1182 canonical/metadata，不等于当前训练已扩充。
- 518真实照片在frames前24之后，当前SS24候选没有用；是可用数据未纳入，不擅自把它们接进需要姿态配对的任务。
- 没有标准tests目录，主要是硬编码路径/人工utility；部分工具会自动安装Blender或默认占GPU0，不能直接运行旧脚本。
- Pipeline.from_pretrained相对path+远程model path先尝试错误HF组合再fallback；本次产生HEAD到checkpoints/microsoft错误路径、network unreachable后依cache成功。当前根pipeline SS decoder依赖HF cache，不完全离线自包含。
- online目录当前是Seed3D外部API，不是TRELLIS权重服务；8555未见监听，不能说生产TRELLIS服务已运行。源码无API认证/请求大小或zip输出上限，健康接口不检查外部SDK/key（未读取或测试任何密钥）；不启动收费生成。仅作为附属服务风险，不混为SS根因。
- resume未保存Python/NumPy/Torch/CUDA/worker RNG，sampler预取使位置可能先于已训练batch；不能严格轨迹复现。

## 5. 已排除的误判与已验证正常项

全量CPU检查view-side EMA20k、axis10 EMA20k、shape v2 EMA20k：每份940tensors、150完整rank8 pairs、300 adapter tensors全fp32，无NaN/Inf/shape错/缺base/多冻结key；640冻结tensor逐一与官方base cast到该dtype后精确相等，rope_phases不在文件。不能再把当前失败简单归因底座载错、adapter丢失、非finite权重。

预训练加载->冻结->LoRA注入->optimizer/EMA顺序正确；只有adapter训练。零B标准初始化保持base、第一步down梯度0/up非0正常。EMA.9999正确递推，20k初始adapter参数残留13.53%，不能称随机巨大基座污染，也不能把输出简化为live的86.47%（EMA分别平均A/B，E[B]E[A]不等E[BA]）。

Flow时间方向、v目标、t×1000、CFG zero neg、SS rawlatent无归一化等源码链一致；DINO训练推理token提取算法相同。fresh与当前推理RoPE保护已有实现，不把旧buffer故障当当前发生。rank8/alpha8/dropout0/5attention targets与当前loader一致；dropout未来>0插入后eval风险不影响当前0。

真实帧序全量：renders_cond/t同frames、sorted PNG前24=transforms前24；000..023npz齐、yaw0..345，无重复frame名、无real混前24。canonical1182数组来源唯一。没观察通用列表错位导致当前错误；真正canonical错选是掩膜/宽高比指标而非文件排序错位。

## 6. 文档差异与具体缺失

- HANDOFF_NEXT漏记axis10已完成；REPORTcanon_side未对应运行目录；A′实际到13k；旧SS/shape现存档位数量大幅少于旧HANDOFF。
- archive MANIFEST仍称B′24k最好、轮廓IoU代表保真、近base解释成立，与后续REPORT修正冲突；应保留历史而明确标记过时，不照着重跑。
- canonical构建注释“所有目标index1”被极端对象实测证伪；axis10输入≤10不能推导target≤10。
- 数据camera matrix占位与“真实外参”潜在误读；verify名字的full不等全量实际目标核查。
- releaseREADME自包含示例数据与资产实际缺失；线上README是Seed3D不是TRELLIS。

缺失：原offaxis/aspect cache生成脚本及生成manifest、v4真实CLI及真实渲染外参来源、测试001–004到训练资产映射、历史不可变采样集合和源码hash、axis10的003正式GLB及四图完整目视验收、alpha全633derived质量统计、bf16条件差异量化、精确车顶/车底高深轴对照。

## 7. 建议执行顺序与验收

1. **先完成数据契约，不训。** 对131目标错配列出对象/源view/输入view；对3e...明确标异常；审查633derived alpha，使用真实透明render或可信mask比较，不改原缓存，所有候选写新目录/version/hash。canonical选择用姿态/几何约束加mask质量，而非只按最大bbox aspect；找不到合格目标则显式排除，禁止fallback任意000。
2. **独立确认完整轴语义。** 选非对称且宽高明显不同资产，以车顶/底/左右锚点追踪dump->vox->SS->shape->rawmesh->formalGLB。分别检查length/up/depth与det；结论前不改PERM、不canonicalize验收。
3. **补可失败的验收与加载检查。** 输出实际组件路径/hash/rank/alpha/flags/seed/sampler；显式缺权重或非允许键失败；batch有失败返回非零；区分快捷raw与formalGLB坐标。以阶段时间预算补003，固定视图检查004倾斜，并报告形状比而不单独用AABB。
4. **最小修resume/保存/snapshot。** 保留adapterfp32与RoPE，原子整组checkpoint完成标记，保存RNG并明确restart/resume语义，重试耗尽非零。用CPU tiny+中断恢复测试，不先全训验证。
5. **上述定义通过后才新短对照。** 只改一变量、另命名、从原厂，留原结果；阶段性SS/GLB朝向、宽高与截面监控，001不改善就不跑完整20k；四图seed42为最终门槛，多seed参考。

本审计不推荐现在直接升级rank/修改lr/EMA，也不推荐把全局yz或canonicalize当质量修复。

## 8. 证据产物索引

同目录：checkpoint_integrity.json、tiny_resume_result.json、audit_cpu.py；data_summary.json/data_records.json/data_issues.json；ss_screen.json/audit_ss_axis10_four.py；002_metrics.json/004_metrics.json；three_acceptance.png/preview_three_acceptance.py；alpha极端和预处理抽样证据另复制入同目录。上一轮audit.json与15502集合/PCA仍有效，不重复该实验证据。

远程正式产物：results_infer/axis10_20k_001_recovery_20261013/001.glb；results_infer/axis10_20k_002_fullaudit_20261013/002.glb；results_infer/axis10_20k_004_fullaudit_20261013/004.glb。各infer.log、command.txt与metrics.json。不要把fast_glb/PLY视为这些正式GLB的同坐标替代品。
