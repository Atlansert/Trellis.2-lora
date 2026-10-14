# TRELLIS.2 上下文恢复与数据诊断（2026-10-13）

## 核对边界

本次先读交接与历史报告，再核对 Git、代码、配置、日志和现存产物，向用户输出状态确认后才继续新诊断。没有改已有项目源码、配置、训练数据或检查点，没有 reset/checkout/commit，没有删除，没有重跑已定论的训练。

服务器：test@192.168.65.39。`/home/lyx/lzx/TRELLIS.2` 与 `/data/lzx/TRELLIS.2` 经 readlink 与设备/inode 核对为同一目录。results -> /data/TRELLIS.2/results；training_data -> /data/TRELLIS.2/training_data。

Git：main，HEAD 75fbf01，git status 为 main...origin/main；18 个 tracked 文件有修改，大量 untracked 文件。最近提交是上游 2026-06-05 的 CodeQL 修复合并；本地 LoRA 工作尚未提交，不能用 HEAD 代表全部当前实现。

记忆服务 disconnected；没有以旧记忆替代磁盘证据。恢复时 GPU0–3 各约134GB已用，不使用。GPU4–7约528MiB、利用率0%；有ComfyUI常驻服务，不能声称完全无进程，也不会终止它。

## 目标与当前架构

单图条件 -> DINOv3@512/crop_scale=1.0 -> SS flow原厂底座+rank8/alpha8 LoRA -> SS decoder -> 固定shape flow v2 EMA20k -> shape decoder -> remesh -> GLB。

当前目标：data_test/001–004，seed42，pipeline_type=1024，不用ss_canonicalize或ss_swap_yz，成品长度轴z，且比例/几何不劣于旧基线。轴正确不是充分条件。多种子、loss、二面角和最佳视角轮廓IoU都不是最终验收门槛。

## 当前训练状态：日志/磁盘核实

| run | 实际目标/过滤 | 样本 | 状态 |
|---|---|---|---|
| multiview_1186_ss_flow_lora_64_views_side | view，aspect>=3.5，无offaxis | 1151对象/15502对 | 完成20k；10个EMA从2k到20k |
| multiview_1186_ss_flow_lora_64_canon_axis10 | canonical，aspect=0，offaxis<=10度 | 1181对象/2362对 | 完成20k；10个EMA；console明确Training finished |
| multiview_1186_ss_flow_lora_64_views_canon_side | 仅配置存在 | — | 结果目录不存在 |
| multiview_1186_ss_flow_lora_64_views_canonv2（A′） | 历史逐视角/全视角 | — | 现存EMA到13k；不是报告旧写法24k/28k |
| multiview_1186_ss_flow_lora_64_views_canon_canonical（B′） | 历史规范帧/全视角 | — | 现存EMA到24k；历史比例过瘦，不能当最佳 |

view-side与axis10当前源配置均与各自run/config.json解析后完全相等。两轮都从原厂权重、ckpt none开始；条件DINOv3@512。axis10与view-side同时改变目标帧与过滤方式，因此不能用两者对比单独证明偏角的因果作用。

view-side末步MSE=0.0306973、elapsed13412.715秒；axis10末步MSE=0.0226956、elapsed13403.509秒。不能根据这两个loss排名成品。

## 现存view-side GLB复测（001，固定shape20k）

本次CPU trimesh加载GLB为scene重新计算AABB，与最新交接一致；没有重做这些已有端到端实验。

| 权重 | AABB(x,y,z) | 最长轴 | z/y | x/y |
|---|---|---|---|---|
| base SS | 0.236474,0.159663,1.001461 | z | 6.2723 | 1.4811 |
| view-side2k | 0.236737,0.155523,1.001293 | z | 6.4382 | 1.5222 |
| 10k | 0.192694,0.154847,1.001438 | z | 6.4673 | 1.2444 |
| 12k | 0.139330,0.148636,1.002026 | z | 6.7415 | 0.9374 |
| 14k | 0.132459,0.585394,0.889630 | z | 1.5197 | 0.2263 |
| 16k | 0.130292,0.887650,0.725590 | y | 0.8174 | 0.1468 |
| 18k | 0.148760,0.849879,0.732725 | y | 0.8622 | 0.1750 |
| 20k | 0.149245,0.812029,0.694869 | y | 0.8557 | 0.1838 |

12k已变窄，14k不能因argmax仍是z就宣称朝向正常。view-side002–004未见对应产物，不能声称四图皆测试失败。

## 新数据诊断：已验证事实

使用实际数据集类，按run/config.json和command.txt恢复当前有效集合，得到1151对象/15502对，与原启动日志一致。不存在当时不可变样本清单，因此这是“当前集合复现日志数量”，不是历史每次随机抽样的逐步重放。

对全部15502对，读取对应v4 coords并施加(2,0,1)，独立计算协方差最大特征向量与index1的无向夹角：degrees(acos(abs(v[1])))。与角缓存差值最大0.0049968度，中位0.0024997度，没有缺角或>0.011度差异。角缓存在该集合上的数值与此定义一致，但原生成脚本仍未找到，不声称找到了其原始实现。

cache SHA256：db82f002ef4f8fa9fe6bdbab83e248f961c1941e8b1e9e4d646dd2e3421fb23c。

| 偏角大于 | 对数 | 按配对占比 | 按对象均匀采样概率 |
|---|---|---|---|
| 10度 | 13204 | 85.1761% | 83.9138% |
| 15度 | 9925 | 64.0240% | 62.6059% |
| 20度 | 8667 | 55.9089% | 53.9278% |
| 30度 | 4992 | 32.2023% | 30.4928% |
| 45度 | 2914 | 18.7976% | 17.5319% |
| 60度 | 374 | 2.4126% | 2.0194% |

角度中位29.93度；最大/次大特征值比全量最小5.313、中位31.427，长轴不是近圆截面里的不稳定次轴。角<=10度的2298对，置换体素AABB长轴全部index1。

从当前训练集合中明确抽3个哈希对象，各取近0/30/45度视角，共9对。CPU加载同一SS decoder（转fp32，不加载整个pipeline），decoded occupancy与置换后的对应v4体素IoU均1.0，三轴跨度一致。例：0086bbab37559442为[10,64,8]、[10,59,37]、[10,49,49]。这支持“正侧视轴对，斜侧视占比高”；不证明全量latent逐一正确，也不证明三维手性或透视配对完全正确。

训练目录没有对象名001。data_test/001.png是测试编号，目前没有它到训练资产哈希ID的映射；没有伪造“抽查001物体”。

诊断产物：diagnostics/recovery_20261013/audit.json、training_subset.json、pairing_samples.png及只读脚本audit_viewside_recovery.py。

## 代码/文档差异与风险

1. HANDOFF_NEXT_20261012遗漏axis1020k完整训练；其“只有view-side完成”不再成立。
2. REPORT11.22中的canon_side正在训练没有对应结果目录；不能与axis10混称。
3. generate_ss_view_voxels_canon.py默认axis_map=(0,2,1)是det=-1，signs=(-1,+1,-1)乘积+1，rz det=+1，合成det=-1。其“纯旋转/保持手性”注释的数学理由错误。编码循环置换det=+1并不能消除该反射。但是dump->render可能本身需反射，且现存npz没有完整生成参数；不能仅凭这条就宣布全量数据镜像，也没有修改数据。
4. 偏角缓存无版本/来源hash/单位验证；体素npz只存coords，生成器会按文件存在跳过。旧缓存或修改后混版本是风险，不是已证明本次发生。
5. verify_canon_full并非全量图/latent几何对应验证；verify_canon_data用分别排序的图/latent列表比较宽高比相关，不足以证明手性和精确配对。本次抽查按transforms顺序绑定文件。
6. run_ss_lora_canon_axis10脚本注释把“斜视导致圆管/翻轴”写成原因，但本次只能确认斜视目标广泛存在，尚不能把原因当作事实。

## 已有结论，不重复实验

全局yz交换不能消除混合目标；DINOv2与原厂DINOv3条件错配不可重新作为产线方案；旧绕dump错误轴的生成器不运行。B′全视角规范帧历史上比例过瘦，不再只凭轴稳定宣布成功。canonicalize、loss、二面角与最佳视角IoU不作为验收捷径。

## 新axis10效果验证

状态：单次001端到端验证已成功完成。使用GPU4，启动前/结束后528MiB、0%（ComfyUI常驻未动），受管前台运行240.495秒、退出0；推理进程2323671已结束。不是重训，不修改已有配置。命令固定seed42、pipeline1024、shape v2 EMA20k、官方shape decoder，禁止canonicalize/swap_yz。

产物：results_infer/axis10_20k_001_recovery_20261013/001.glb（60,494,652 bytes）。同目录command.txt、infer.log、metrics.json记录复现与测量。日志确认加载axis10 SS EMA20k、指定decoder、shape v2 EMA20k；SS missing=1/unexpected=0符合pop rope_phases实现，无错误。

CPU scene AABB(x,y,z)=[0.2051643580,0.1847592369,1.0019302964]，最长轴z，z/y=5.4228969，x/y=1.1104417；688142顶点/954334面。相对输入001前景W/H约4.23，仍偏长/瘦约28.2%；base的z/y=6.2723。这个全局比值优于base与历史B′24k的8.91，但不代表细节/几何已经通过验收。

生成CPU固定尺度、原始GLB坐标的侧视/固定斜视几何预览mesh_compare.png（近似三角深度排序，不是最终照片级渲染）；没有纠轴或挑最佳视角。目视axis10不呈现view-side20k的大幅斜向，但车头/底部仍有几何突起，不能仅据AABB宣称质量达标。view-side20k在固定短画布中被裁切，图只说明错向，不供该行全形状验收。002–004尚未对axis10测试，不能上产线。

## 后续决策边界

先评估axis10现有权重，不再盲目重开规范帧训练。即使单图轴对也不能直接上产线，须比例、mesh目视及002–004验收。若axis10失败，要区分这不是“view+仅偏角过滤”的单变量实验：不可因此宣布偏角过滤已证伪。任何新的短训应另命名、从原厂权重、固定其他条件，先对001分阶段检查形状，再决定是否扩四图。

具体缺失：原偏角生成脚本/来源元数据、001测试图到训练资产的映射、历史不可变训练样本清单，以及新增权重在四图上的完整mesh验收。未验证根因：斜视监督导致平均化/训练崩溃的因果链、实际数据三维手性、rank8能力限制、EMA底座能力百分比。
