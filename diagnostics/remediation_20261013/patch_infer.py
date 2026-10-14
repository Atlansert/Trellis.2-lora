from pathlib import Path
import hashlib
p=Path('/data/lzx/TRELLIS.2/scripts/infer_finetuned.py');s=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='b45990e9ea91c86faa16104728aaab4a5a564a727043b0da2666fc3241af5cb1'
s=s.replace('import json\n','import json\nfrom scripts.utils.inference_contract import (TARGETS, require_file, HashCache, checked_load, validate_lora, glb_coordinates, export_metadata, timed, pipeline_timing, discover_images, run_batch, write_manifest, local_pipeline_weights)\n',1)
# Loading is centralized and consistent; full-state LoRA checkpoints only.
a=s.index('    loaded = []');b=s.index('\n\n\ndef infer_single',a)
s=s[:a]+'''    loaded = []
    full = [('vae_decoder', 'shape_slat_decoder', vae_decoder_path),
            ('flow_512', 'shape_slat_flow_model_512', flow_512_path),
            ('flow_1024', 'shape_slat_flow_model_1024', flow_1024_path)]
    adapters = [('sparse_structure_lora','sparse_structure_flow_model',sparse_structure_lora_path,ss_lora_rank,ss_lora_alpha),
                ('flow_512_lora','shape_slat_flow_model_512',flow_512_lora_path,shape_lora_rank,shape_lora_alpha),
                ('flow_1024_lora','shape_slat_flow_model_1024',flow_1024_lora_path,shape_lora_rank,shape_lora_alpha),
                ('tex_slat_lora','tex_slat_flow_model_1024',tex_slat_lora_path,tex_lora_rank,tex_lora_alpha)]
    # Validate every requested path before mutating any model.
    for _,_,path in full:
        if path:require_file(path)
    for _,_,path,_,_ in adapters:
        if path:require_file(path)
    for name,key,path in full:
        if not path:continue
        print(f'  Loading {name}: {path}',flush=True)
        state=torch.load(path,map_location='cpu',weights_only=True)
        checked_load(pipeline.models[key],state,allow_rope=True)
        loaded.append(name)
    for name,key,path,rank,alpha in adapters:
        if not path:continue
        rank=lora_rank if rank is None else rank
        alpha=lora_alpha if alpha is None else alpha
        model=pipeline.models[key]
        print(f'  Loading {name}: {path} rank={rank} alpha={alpha}',flush=True)
        replaced=apply_lora_to_linear_layers(model,TARGETS,rank=rank,alpha=alpha,dropout=lora_dropout)
        state=torch.load(path,map_location='cpu',weights_only=True)
        # Runtime-derived RoPE must not be overwritten by historical cast buffers.
        state.pop('rope_phases',None)
        validate_lora(model,replaced,state,rank,alpha,lora_dropout)
        checked_load(model,state,allow_rope=True)
        print(f'  Loaded {name}: 150 verified adapters, eval mode',flush=True)
        loaded.append(name)
    return loaded
''' + s[b:]
s=s.replace('no_video=False, geometry_only=False, fast_glb=False):','no_video=False, geometry_only=False, fast_glb=False, timings=None):',1)
s=s.replace('    mesh = pipeline.run(image, seed=seed)[0]\n    mesh.simplify(max_faces)','''    timings={} if timings is None else timings
    synchronize=torch.cuda.synchronize
    with pipeline_timing(pipeline,timings,synchronize):
        with timed(timings,'pipeline_run',synchronize):
            mesh = pipeline.run(image, seed=seed)[0]
    with timed(timings,'simplify',synchronize):
        mesh.simplify(max_faces)''',1)
s=s.replace('vertices=mesh.vertices.detach().cpu().numpy(),','vertices=glb_coordinates(mesh.vertices.detach().cpu().numpy()),')
# Add explicit export timing around each full branch (context boundaries).
a=s.index('    if geometry_only:');b=s.index('    # 渲染视频',a)
block=s[a:b]
# wrap individual branches preserving returns, formal block separate
x=block.index('    if fast_glb:');y=block.index('    # 导出 GLB',x)
g=block[:x];f=block[x:y];formal=block[y:]
def wrap_branch(text):
 lines=text.splitlines();return lines[0]+'\n        with timed(timings,\'export\',synchronize):\n'+'\n'.join('    '+line if line else '' for line in lines[1:])+'\n'
block=wrap_branch(g)+wrap_branch(f)+"    with timed(timings,'export',synchronize):\n"+'\n'.join('    '+line if line else '' for line in formal.splitlines())+'\n\n'
s=s[:a]+block+s[b:]
s=s.replace("help='仅用于几何调试：导出无纹理 GLB；正式结果不要使用，会跳过 O-Voxel 重网格/坐标转换'","help='几何调试无纹理 GLB；统一正式GLB坐标，但跳过remesh/UV，不能作为正式验收'")
s=s.replace('    opt = parser.parse_args()','''    parser.add_argument('--image_timeout_seconds',type=float,default=0,
                        help='Unix Python alarm per image; CUDA C extensions may delay it; use outer subprocess deadline')
    opt = parser.parse_args()
    import math
    if not math.isfinite(opt.image_timeout_seconds) or opt.image_timeout_seconds<0:parser.error('timeout must be finite nonnegative')
    if opt.geometry_only and opt.fast_glb:parser.error('geometry_only and fast_glb are mutually exclusive')
    explicit_fields=['vae_decoder','flow_512','flow_512_lora','flow_1024','flow_1024_lora','tex_flow_lora','sparse_structure_lora']
    explicit={k:getattr(opt,k) for k in explicit_fields if getattr(opt,k) is not None}
    # All argument/input failures precede pipeline construction or GPU allocation.
    for value in explicit.values():require_file(value)
    image_paths=discover_images(opt.image,opt.input_dir)
    os.makedirs(opt.output,exist_ok=True)
    manifest_path=os.path.join(opt.output,'run_manifest.json')
    if os.path.exists(manifest_path):raise FileExistsError('existing run_manifest.json: use a fresh output directory')
    hashes=HashCache()
    manifest={'schema':'trellis.inference-run.v1','status':'loading','flags':vars(opt).copy(),
              'seed':opt.seed,'pipeline_type':opt.pipeline_type,'export':export_metadata(opt.geometry_only,opt.fast_glb),
              'explicit_weights':{k:hashes.record(v) for k,v in explicit.items()},'images':[],
              'lora':{'targets':list(TARGETS),'expected_layers':150,'rank':opt.lora_rank,'alpha':opt.lora_alpha,
                      'ss_rank':opt.ss_lora_rank or opt.lora_rank,'ss_alpha':opt.lora_alpha if opt.ss_lora_alpha is None else opt.ss_lora_alpha,
                      'shape_rank':opt.shape_lora_rank or opt.lora_rank,'shape_alpha':opt.lora_alpha if opt.shape_lora_alpha is None else opt.shape_lora_alpha,
                      'tex_rank':opt.tex_lora_rank or opt.lora_rank,'tex_alpha':opt.lora_alpha if opt.tex_lora_alpha is None else opt.tex_lora_alpha,
                      'alpha_verification':'historical checkpoints lack alpha metadata; CLI value recorded, not inferred','dropout':opt.lora_dropout}}
    write_manifest(manifest_path,manifest)''',1)
s=s.replace("    if opt.pretrained_only and (opt.vae_only or opt.vae_decoder or opt.flow_512 or opt.flow_1024):","    if opt.pretrained_only and (opt.vae_only or explicit):")
s=s.replace("    # ── 加载 Pipeline",'''    manifest['resolved_weights']={k:hashes.record(getattr(opt,k)) for k in explicit_fields if getattr(opt,k)}
    manifest['base_weights']=local_pipeline_weights(opt.checkpoints_dir,hashes)
    manifest['automatic_weights']=[k for k in explicit_fields if getattr(opt,k) and k not in explicit]
    manifest['official_components_retained']=[k for k in explicit_fields if not getattr(opt,k)]
    write_manifest(manifest_path,manifest)

    # ── 加载 Pipeline''',1)
s=s.replace("    if not loaded:\n", "    manifest['loaded_components']=loaded\n    manifest['samplers']={k:getattr(pipeline,k,None) for k in ['sparse_structure_sampler_params','shape_slat_sampler_params','tex_slat_sampler_params']}\n    write_manifest(manifest_path,manifest)\n    if not loaded:\n",1)
s=s.replace("    envmap = EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)","    envmap = None if opt.no_video or opt.geometry_only or opt.fast_glb else EnvMap(torch.ones(256, 512, 3, dtype=torch.float32, device='cuda') * 0.5)",1)
a=s.index('    # ── 推理 ─');b=s.index("\n\nif __name__",a)
s=s[:a]+'''    # A failed image is recorded and any partial failure returns nonzero.
    def runner(path,timings):
        return infer_single(path,pipeline,opt.output,seed=opt.seed,envmap=envmap,
                            max_faces=opt.max_faces,no_video=opt.no_video,
                            geometry_only=opt.geometry_only,fast_glb=opt.fast_glb,timings=timings)
    code=run_batch(image_paths,runner,manifest,manifest_path,opt.image_timeout_seconds)
    print(f"Inference {'FAILED/PARTIAL' if code else 'SUCCESS'}: {manifest['successful_images']} success, {manifest['failed_images']} failed. Output: {opt.output}",flush=True)
    return code
''' +s[b:]
s=s.replace("    main()","    try:\n        sys.exit(main())\n    except Exception as e:\n        print(f'[RUN FAILED] {type(e).__name__}: {e}',flush=True)\n        import traceback\n        traceback.print_exc()\n        sys.exit(1)")
p.write_text(s)
print('PATCHED',hashlib.sha256(p.read_bytes()).hexdigest())
