# Stage1 全量alpha风险扫描（2026-10-13）

## 完成状态

CPU only，CUDA_VISIBLE_DEVICES=''，8进程；未import torch/模型、不渲染、不重训、不修改原训练数据/缓存/源码/nju。远程新输出根：/data/lzx/TRELLIS.2/diagnostics/remediation_20261013/alpha_scan。

全1182对象每对象前24render完成，共28368帧：633derived/15192帧、549native symlink/13176帧；不扫描后续real照片。每32对象一个原子JSONchunk，37chunks（36×32+30），每份有sha256完成标记，manifest包含脚本、metadata、offaxis cache hashes；每帧stored及derived original SHA256、transform hash。第二次自然续跑跳过32完整chunk并验证hash。脚本未修改manifest后覆盖原数据。风险阈值v1固定，候选不是ground truth。

第一段500.441秒PID2344206/exit0，自然budget_stop完成1024对象；第二段83.203秒PID2345300/exit0完成余158及聚合/contacts。两PID /proc均不存在；无剩余worker，未开启后台遗留任务。COMPLETE.json只在全部扫描+聚合+contact成功后写出；4个汇总文件SHA现场核验全通过。若frame读取失败将记录error，当前error=0；COMPLETE代表扫描覆盖完成，不代表数据质量合格。

## 指标与结果

mask area/fraction、empty/full、bbox/fill/aspect、alpha>0与>204、components、binary alpha；derived原RGBdiff threshold3→closing5×5→component数与LCC→fill重算，记录closed前景被stored移除数量/比例、填洞新增、near-bg在closedbbox内比例（仅风险，不是车身真值）。stored RGB与原图逐像素相等，算法expected与stored mask逐像素相等。

- 28368帧读取error0；empty0、nearfull>=.99为0。
- derived 15192帧RGB mismatch0、algorithm mismatch0；并非说明mask正确，只说明现存mask忠实执行了该不可靠算法。
- 风险候选85对象（都derived）；207帧removed closed components >10%；2帧mask area<.2%；2帧bboxfill<.15。flag有重合，不能相加当unique帧数。
- 当前cache≤10°输入derived1264、native1098；canonical staged548/1096另有存在性/合同门槛，此扫描仅计算cache角度。
- ≤10°且触发风险仅6帧/3对象：346b98fb789b8a58 v6/v18（绿车明显只留中后部）、3e988013c8136c9b v6/v18（白body丢）、dfe79d2920288220 v0/v12（白body/轮组丢）。风险contacts已读：original / alpha /实际512训练crop，保留原坐标图，不用于自动真值判定。
- 85中其余82主要在别的视角触发；没有flag的548derived不意味着可靠。近背景白body丢失甚至可能变成大的实心轮廓或不触发阈值；阈值有漏报/误报。

## 替代数据方案：只查可行性，不执行

已读现存render_new_data.py与Blender render_view.py。本地Blender4.5.1存在；--transparent设置film_transparent，真实mesh渲染可产真实geometry alpha，优于RGB背景差分或语义rembg猜mask。旧render_new_data默认GPU/输出同资产单元，不能直接对原data overwrite。本次未启动Blender或安装工具。

全633derived源资产存在：327OBJ无MTL（现场逐目录验证0/327），306GLB均有material，其中293有embedded images。327OBJ无法恢复原材质，只用其geometry渲染alpha并配旧RGB还需要严格验证真实相机、尺寸/归一化、投影/遮挡像素对齐，不能直接把新灰模RGB替换原训练图。13GLB无embedded image不一定无纹理（可能常色或外部URI），需资产级确认；asset_followup记录每份material/image/externalURI。全633都有pbr_dump和mesh_dump pickle，存在不等于dump含足够原材质，不加载不可信pickle或盲目声称可以纹理重渲。

透明重渲应先在独立候选目录以少量GLB核对RGB/相机/alpha、记录hash和真实外参；OBJ可geometryalpha-only与存档配准验证，若无法真实配准继续隔离；不自动全量rembg或render。

## Stage1建议与Stage2门槛

暂时只使用native549中合同允许的548对象/1096对作为staged候选，633derived一律quarantine最保守，不据85flag自动允许其它548derived。若扩大：先3严重对象单独隔离，再按资产材质可恢复/geometryalpha可配准分层人工验证，并新version保存；不改原alpha/cache。

native未flag也不保证真实RGB细节或姿态。父已发现shape监督原长轴2与SS规范target长轴1冲突；因此本扫描完成不授权训练，必须Stage2完整SS/shape/render坐标契约。不得因548native≤10通过就启动训练。

## 文件与hash

远程records.json（全帧指标/hashes）、risk_candidates.json（85对象）、summary.json、manifest.json、COMPLETE.json、asset_followup.json、contact_selection.json、valid_inputs_risk_contact_1.png与_2.png、chunks/37组、segment{1,2}.{pid,exit,log,command.txt}、alpha_quality_scan_20261013.py。

summary SHA256 616689e463c685e34b3d28c316dc4949a5f6d7695dd770a1841c851bccd2aeac
records SHA256 6e053546705585a73633d7615f93d2cfe7d13e52be663f9216bc60c2a89c80b3
risk_candidates SHA256 5d54b48deb48af2efa998e6bb35c833cf21efb6aabf699b06b14a5d743d991fd
manifest SHA256 069fb0f526663a676cc04f2f9a8722ed43bf287931f6b823dfd310492dd39b88

本地remediation目录交付脚本、报告、summary/COMPLETE/contact images；全部frames records存远程，避免用摘要代替完整复核。

可重启：同脚本/root/out/workers再次运行，manifest与chunk hash必须相同。当前全扫描完成，不需要重算。改变算法/阈值必须新的版本输出，不复用这些complete markers。

工程边界：脚本只校验manifest及已存chunk内容hash，未重读已完成chunk全部input hash来检查运行中源数据变动；父已对staged数据另做sourcehash gate，未来扫描跨长期恢复应先比对记录input hash。风险contacts排序顶24≤10输入中仅前6有flag，余用于比较未flag的derived，不能把整sheet所有对象描述为flagged。
