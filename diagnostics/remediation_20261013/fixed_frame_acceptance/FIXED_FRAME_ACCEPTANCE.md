# Stage3 CPU fixed-frame acceptance diagnostics

## 当前v4：精确三角表面积积分（v3数值原样保留）

facecenter-area covariance仍对长三角的不同细分敏感；v4补每个uniform triangle内部协方差Σ_i(v_i-center)(v_i-center)^T/12，加到facecenter-areaCov，实现coplanar三角细分下精确表面moments不变。

旧pca_*键仍是历史facecenter代理。新增surface_pca含mean/covariance/eigenvalues/major_axis_world_xyz/major_to_z_offangle_degrees/major_to_second_eigenvalue_ratio/axis_ambiguous；primary_orientation_method明确exact_uniform_triangle_surface_integrated_covariance，主orientation_reference_status与ambiguity用surface字段。不可从旧pca_*读当前primary，不是改用户验收10deg强gate。

15tests通过（新单三角两次细分、解析矩形+细分、float32旋转box细分3tests）；所有原scene/provenance/缺失/raw/manualreview回归保留。只新v4完整12GLB报告、耗时23秒；无重绘profile，无旧v3覆盖。

v4 axis10角0012.8313044536°、0021.0970299554°、003.8898225144°、00410.1735668063°，004相对旧proxy仅7.44e-6°变化，倾斜结论保留。source粗长面差异可能大，父compatpilot应改用surface字段。四图geometry仍unverified、新003provenanceverified、历史其余unverified。

报告remote diagnostics/remediation_20261013/fixed_frame_acceptance/v4/report.json SHA7872ff56ced49ca8fbeb9df7ac507fbb0bf44192cfb9c41dea495fe8b466b456，schema trellis.fixed-frame-report.v2。工具SHA1aeff5ae678a118765cfb659999d85a9deb873a66a036eaf84ae74b67ecdd13b，tests83fe5f4795855b8f0f4c0d0303efde192f8922ef74a34a639fac5032c8fbadf0。

## 最终v3更新（下面v2表格保留历史缺003证据）

父补正式003后，最终报告为fixed_frame_acceptance/v3/report.json（SHA256 fb553333f85dacc20b93b591642f6486ac65bd5982638be5c2708a83521bad61），12正式GLB现已全覆盖。003面面积加权PCA到z角.8897971436°、extent[.2466765419,.2463259324,1.0015927553]、z/y4.0661279362、x/y1.0014233557、degenerate0、longz。

运行证据独立于几何：历史11均provenance_unverified，即使manual geometry pass也不能生产pass。新003 run_manifest_path对接trellis.inference-run.v2，实际input_artifact与output的sha/path/size验证，manifest文件全hash记录，status/image success、flags seed42/pipeline1024/actual coordinate false、formal_remesh_textured、loaded overrides与actualbase config/weight来源哈希记录完整，provenance verified。巨大checkpoint只核对记录结构/path/size并保留SHA，不重新读取几十GB或加载tensor。

最终12tests通过，新增nativev2、input/outputhash错、raw/纠轴contract、manual review缺provenance仍unverified。生产通过必须formal export、verified运行证据、方向参考筛查与explicit人工geometryreview共同成立；不能只看manualreview/PCA。四图axis10 geometry仍全部unverified，004倾斜10.173559°仍存在，诊断参考10°不是用户新生产强gate。

v3没有重绘profile；下面图片仍只001/002/004、旧missing证据未覆盖。最后显式raw/纠轴manifest fail状态分支新增并通过12tests，不改变v3现有数值，未再次重复eval。

新增远程 scripts/utils/evaluate_fixed_frame.py 与 tests/test_fixed_frame_acceptance.py；未编辑infer_finetuned.py或任何已有实现，不GPU、不推理、不训练、不渲染、不修改输入mesh。

## 指标与方法

- trimesh.load(force=scene,process=False)，遍历scene.graph.nodes_geometry，将各实例全部world transform应用到copy后合并；重复实例分别处理。
- finite、原坐标AABB/extents、最长AABB轴、degenerate faces计数（area tolerance=maxspan²*1e-14）。
- 面面积加权面心covariance/PCA，而非vertexPCA；避免UV seam/重网格vertex数量直接偏置。三角面心仍是surface代理而非体积PCA，粗网格细分会有积分误差。
- 主PCA轴与固定z夹角，以abs(z)消除符号不定；world x/y横向尺度与固定ZY、ZX投影bbox。没有PCA校正旋转，没有最佳角选择。
- 未进行最大连通面网面积统计；这次范围是主PCA与degenerate数量，报告明确缺少拓扑完整度。

## 11既有正式GLB结果，12记录保持缺失003

| record | PCA到z角° | AABB最长轴 | degenerate faces | z/y | x/y |
|---|---:|---|---:|---:|---:|
| base001 | 1.0894 | z | 2 | 6.2723 | 1.4811 |
| base002 | 89.6166 | x | 0 | 1.4754 | 5.8264 |
| base003 | 1.6027 | z | 5 | 5.1263 | 1.2093 |
| base004 | .1714 | z | 0 | 8.5646 | 1.1211 |
| old33k001 | .4466 | z | 0 | 8.5737 | .9517 |
| old33k002 | .8936 | z | 0 | 6.3721 | .6971 |
| old33k003 | 1.4134 | z | 3 | 6.0518 | .9953 |
| old33k004 | .2888 | z | 0 | 8.8388 | .9138 |
| axis10_001 | 2.8313 | z | 0 | 5.4229 | 1.1104 |
| axis10_002 | 1.0970 | z | 0 | 3.9112 | .9910 |
| axis10_003 | **missing** | — | — | — | — |
| axis10_004 | **10.1736** | z | 1 | 4.3801 | .6406 |

axis10_004 PCA方向[-.005342672,-.176549718,.984277224]，定量支持固定侧视可见的倾斜。AABB longestz不能保证平直，且倾斜会增大y extent让z/y看似改善。不要把PCA轴当车顶/车底帧或保真度。

## 状态三态与边界

- artifact_status: pass/fail/missing。003不存在返回missing，整表不能漏它并报告四图通过。
- orientation_reference_status: pass/fail/unverified。CLI --reference-angle10是**诊断参考线**，不是用户新批准的强制生产角度门槛；004仅略超10，精确数值与范围一并保留。PCA谱比<1.5标unverified避免不定主轴误通过。
- geometry_not_worse_status: pass/fail/unverified，必须由人工review JSON明确提供reviewer/evidence/rationale；缺省11成品全部unverified。
- acceptance_status组合报告，所有记录只有方向和人工几何皆pass才pass。当前总表fail（base002方向、axis004参考方向、axis003缺失），但不声称所有其它成品几何已通过。
- 实际GLB/hash、输入图/hash、formal/raw导出contract、seed/pipeline/纠轴声明、SS/shape/decoder路径、现存load日志行与command内容/hash记录。历史无command保留missing，不发明运行参数证据。声明与实测日志分开；该CPU工具不加载检查点或encoder，不替代父加载验证。
- raw_fast_glb等contract拒绝，命令记录若canonicalize/swap/fast_glb/geometry_only也拒绝。

## 输出/使用

当前最终报告：diagnostics/remediation_20261013/fixed_frame_acceptance/v2/report.json，输入manifest同目录input_manifest.json。root目录初始报告保留，v2加入了输入hash与明确checkpoint路径，不覆盖旧证据。

CPU运行约22秒，全11成品20.4秒首跑。10tests通过，覆盖aligned、25°tilted长轴仍z、缺mesh、empty、degenerate、nonfinite、重复scene instance和translation、无输入mutation、未review不可pass、拒绝raw/纠轴command、missing第四图不能通过、拒绝输出覆盖。

运行示例：

```bash
CUDA_VISIBLE_DEVICES='' /home/test/miniconda3/envs/trellis2/bin/python scripts/utils/evaluate_fixed_frame.py --manifest path/to/new_input_manifest.json --output new_output_dir --reference-angle 10
CUDA_VISIBLE_DEVICES='' /home/test/miniconda3/envs/trellis2/bin/python -m unittest discover -s tests -p test_fixed_frame_acceptance.py -v
```

manifest schema trellis.fixed-frame-inputs.v1，records每项id/glb/export_contract(formal_standard_glb)；可选input、checkpoint_contract、infer_log、command_file等。父补003正式产物后用新manifest、新output重跑，不覆盖当前missing证据。

固定同长度profile图沿用父先前three_acceptance.png，输入与old/axis10各固定原坐标/同长度，不重复昂贵渲染；仅覆盖001/002/004，图中没有003不能说四图预览齐全。
