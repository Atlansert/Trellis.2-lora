"""推理契约层：不 import torch/CUDA/模型，任何进程都能安全导入。

用途：给 infer_finetuned.py 提供"显式路径预检 → 严格校验 → 权重哈希 → 原子写 run_manifest"的公共设施。
关键输入：显式权重/图片路径、输出目录、每图超时秒数；关键输出：run_manifest.json 与进程退出码（非 0 = 有失败）。
要点：TARGETS 是 LoRA 目标层白名单（5 类 × 30 层 = 150）；校验一律严格，键/形状不符直接抛错而不是静默加载。
安全边界：只读输入，只写 manifest 路径（先写 .tmp 再 os.replace）；本模块不含模型逻辑，不分配 GPU。

Original (English):
Lightweight inference contracts; no torch/CUDA/model imports.
"""
import contextlib, hashlib, json, math, os, signal, time
from datetime import datetime, timezone
from pathlib import Path
TARGETS=('self_attn.to_qkv','self_attn.to_out','cross_attn.to_q','cross_attn.to_kv','cross_attn.to_out')

def require_file(path):
    """预检显式路径：必须是已存在的文件，返回绝对路径，否则抛 FileNotFoundError（在碰 GPU 之前失败）。"""
    p=Path(path).resolve()
    if not p.is_file():raise FileNotFoundError(f'explicit file unavailable: {p}')
    return str(p)

class HashCache:
    """按要求分块算 sha256 并缓存，避免同一个权重在 manifest 里被重复读盘。"""
    def __init__(self):self.cache={}
    def record(self,path):
        """返回 {path,size_bytes,sha256}；缓存键 = (绝对路径, 大小, mtime_ns)，文件一变就重算。"""
        path=require_file(path);s=Path(path).stat();key=(path,s.st_size,s.st_mtime_ns)
        if key not in self.cache:
            h=hashlib.sha256()
            with open(path,'rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            self.cache[key]={'path':path,'size_bytes':s.st_size,'sha256':h.hexdigest()}
        return dict(self.cache[key])

def checked_load(model,state,allow_rope=True,pop_rope=False):
    """严格加载 state_dict：键集合与每个张量形状必须与 model 完全一致，否则抛 ValueError；成功后置 eval。"""
    state=dict(state)
    if pop_rope:state.pop('rope_phases',None)
    expected=model.state_dict()
    missing=set(expected)-set(state);unexpected=set(state)-set(expected)
    # rope_phases 是运行时算出来的 buffer，历史 checkpoint 里没有属正常，不算缺失。
    if allow_rope:missing.discard('rope_phases')
    if missing or unexpected:raise ValueError(f'checkpoint key mismatch missing={sorted(missing)} unexpected={sorted(unexpected)}')
    for k,v in state.items():
        if hasattr(v,'shape') and tuple(v.shape)!=tuple(expected[k].shape):raise ValueError(f'checkpoint shape mismatch: {k}')
    result=model.load_state_dict(state,strict=False)
    missing=set(result[0]);unexpected=set(result[1])
    if allow_rope:missing.discard('rope_phases')
    if missing or unexpected:raise ValueError(f'load mismatch missing={sorted(missing)} unexpected={sorted(unexpected)}')
    model.eval()

def validate_lora(model,replaced,state,rank,alpha,dropout):
    """strict LoRA 校验：层数必须是 150、adapter 键与 TARGETS 一一对应、运行时 rank/alpha 与张量形状严格匹配。"""
    if rank<=0 or not math.isfinite(alpha) or alpha<=0 or not 0<=dropout<1:raise ValueError('invalid LoRA hyperparameters')
    if len(replaced)!=150 or len(set(replaced))!=150:raise ValueError(f'expected150 LoRA layers, got {len(replaced)}')
    expected_adapter={f'{n}.{suffix}.weight' for n in replaced for suffix in ['lora_down','lora_up']}
    actual_adapter={k for k in state if '.lora_down.' in k or '.lora_up.' in k}
    if actual_adapter!=expected_adapter:raise ValueError('LoRA adapter keys/targets mismatch')
    for name in replaced:
        if not any(t in name for t in TARGETS):raise ValueError('unexpected LoRA target')
        layer=model.get_submodule(name)
        if layer.rank!=rank or layer.alpha!=alpha:raise ValueError('LoRA runtime rank/alpha mismatch')
        down=state[f'{name}.lora_down.weight'];up=state[f'{name}.lora_up.weight']
        if tuple(down.shape)!=(rank,layer.in_features) or tuple(up.shape)!=(layer.out_features,rank):raise ValueError('LoRA rank/shape mismatch')
    # alpha is not present in historical state_dict; record caller value rather
    # than pretending checkpoint metadata verified it.

def glb_coordinates(vertices):
    """坐标契约：raw 轴 (depth, length, up) → 宿主 GLB (x, z, -y)。纯旋转（行列式 +1），不改手性。"""
    out=vertices.copy();out[:,1]=vertices[:,2];out[:,2]=-vertices[:,1];return out

def export_metadata(geometry_only=False,fast_glb=False):
    """记录本次导出模式：geometry_only/fast_glb 走 raw_no_remesh，不具正式验收资格。"""
    return {'coordinates':'GLB (x,z,-y), proper rotation','mode':'raw_no_remesh' if geometry_only or fast_glb else 'formal_remesh_textured',
            'formal_acceptance_eligible':not(geometry_only or fast_glb),'texture_sampling_still_runs':True}

@contextlib.contextmanager
def timed(record,name,synchronize=None):
    """计时上下文：进出各打一条 [STAGE START/END] 日志，并把耗时追加到 record[name]；synchronize 传 torch.cuda.synchronize 才能量准异步 kernel。"""
    if synchronize:synchronize()
    start=time.perf_counter();print(f'[STAGE START] {name} UTC={datetime.now(timezone.utc).isoformat()}',flush=True)
    try:yield
    finally:
        if synchronize:synchronize()
        elapsed=time.perf_counter()-start;record.setdefault(name,[]).append(elapsed)
        print(f'[STAGE END] {name}: {elapsed:.6f}s UTC={datetime.now(timezone.utc).isoformat()}',flush=True)

@contextlib.contextmanager
def pipeline_timing(pipeline,record,synchronize=None):
    """临时给 pipeline 的采样方法套上计时包装，退出时按原样恢复（原本不在实例 __dict__ 里的属性直接删掉）。"""
    methods={'sample_sparse_structure':'ss','sample_shape_slat':'shape','sample_shape_slat_cascade':'shape_cascade','sample_tex_slat':'texture'}
    saved={}
    for method,label in methods.items():
        if not hasattr(pipeline,method):continue
        saved[method]=(method in pipeline.__dict__,pipeline.__dict__.get(method))
        original=getattr(pipeline,method)
        def wrap(*args,_original=original,_label=label,**kwargs):
            with timed(record,_label,synchronize):return _original(*args,**kwargs)
        setattr(pipeline,method,wrap)
    try:yield
    finally:
        for method,(owned,value) in saved.items():
            if owned:setattr(pipeline,method,value)
            else:delattr(pipeline,method)

@contextlib.contextmanager
def image_deadline(seconds):
    """单图超时（Unix SIGALRM/setitimer）：seconds<=0 表示不限时；CUDA C 扩展可能延迟投递信号，所以外层还需子进程兜底。"""
    if seconds<0 or not math.isfinite(seconds):raise ValueError('timeout must be finite nonnegative')
    if not seconds:yield;return
    if not hasattr(signal,'setitimer'):raise RuntimeError('image timeout requires Unix setitimer')
    previous=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):raise TimeoutError('image deadline exceeded; CUDA C extensions may delay delivery; use outer subprocess deadline')
    signal.signal(signal.SIGALRM,expired);oldtimer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,*oldtimer);signal.signal(signal.SIGALRM,previous)

def write_manifest(path,data):
    """原子写 manifest：先写 <name>.tmp 再 os.replace，进程被杀也不会留下半个 JSON。"""
    p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(tmp,p)

def run_batch(paths,runner,manifest,manifest_path,timeout=0,hashes=None):
    """逐图跑 runner，把输入哈希/输出哈希/阶段耗时/失败原因都写进 manifest；任一图失败或输入为空则返回 1。"""
    hashes=HashCache() if hashes is None else hashes
    manifest.setdefault('created_utc',datetime.now(timezone.utc).isoformat())
    manifest['images']=[];manifest['status']='running';write_manifest(manifest_path,manifest)
    for path in paths:
        item={'input':str(Path(path).resolve()),'status':'running','timings':{},
              'started_utc':datetime.now(timezone.utc).isoformat()}
        manifest['images'].append(item)
        try:
            item['input_artifact']=hashes.record(path)
            write_manifest(manifest_path,manifest)
            with image_deadline(timeout):outputs=runner(path,item['timings'])
            item['outputs']=[hashes.record(p) for p in outputs if p]
            # 空输出视为失败：防止"跑完了但没产物"被记成 success。
            if not item['outputs'] or any(x['size_bytes']==0 for x in item['outputs']):raise RuntimeError('no nonempty output produced')
            item['status']='success'
        except Exception as e:
            item['status']='failed';item['error']=f'{type(e).__name__}: {e}';print(f'[IMAGE FAILED] {path}: {e}',flush=True)
        item['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path,manifest)
    failures=sum(x['status']!='success' for x in manifest['images'])
    manifest.update(status='failed' if failures or not paths else 'success',failed_images=failures,successful_images=len(paths)-failures,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    write_manifest(manifest_path,manifest)
    return 1 if failures or not paths else 0


def discover_images(image=None,input_dir=None):
    """解析输入：--image 走单图（须存在）；--input_dir 收集 png/jpg/jpeg/webp，并拒绝同名文件（会互相覆盖输出）。"""
    if image:return [require_file(image)]
    p=Path(input_dir)
    if not p.is_dir():raise FileNotFoundError(f'input directory unavailable: {p}')
    paths=sorted(str(x.resolve()) for x in p.iterdir() if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    if not paths:raise ValueError('no input images')
    stems=[Path(x).stem for x in paths]
    if len(stems)!=len(set(stems)):raise ValueError('duplicate image stems would overwrite outputs')
    return paths


def local_pipeline_weights(checkpoints_dir,hashes):
    """记录 checkpoints_dir 下 pipeline.json 及各组件权重的哈希；引用远程/缓存而本地取不到时标注 unresolved。"""
    root=Path(checkpoints_dir);config=root/'pipeline.json';records={}
    if not config.is_file():return {'unresolved_remote_pipeline':str(root),'warning':'remote/cache artifacts cannot be inferred from URI'}
    records['pipeline_config']=hashes.record(config)
    cfg=json.loads(config.read_text())
    for component,value in cfg['args']['models'].items():
        prefix=root/value;matches=[Path(str(prefix)+ext) for ext in ['.safetensors','.pt']]
        existing=[p for p in matches if p.is_file()]
        records[component]=hashes.record(existing[0]) if existing else {'configured_reference':value,'status':'remote_or_cache_unresolved'}
    return records
