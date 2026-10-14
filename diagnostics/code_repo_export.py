"""Build a code-only export tree of TRELLIS.2 (no data, weights, results).
Whitelist-by-extension plus per-file size cap. Never modifies the source tree."""
import os,shutil,sys,json
from pathlib import Path
SRC=Path('/data/lzx/TRELLIS.2');DST=Path(sys.argv[1]).resolve();MAX=512*1024
if DST==SRC or SRC in DST.parents:raise SystemExit('destination must be outside the source tree')
KEEP_EXT={'.py','.pyi','.sh','.bash','.json','.yaml','.yml','.toml','.cfg','.ini','.txt','.md','.rst','.c','.h','.cc','.cpp','.hpp','.cu','.cuh','.m','.js','.ts','.html','.css','.in','.cmake','.patch','.diff','.gitignore','.gitmodules','.gitattributes','.dockerfile','.env.example'}
KEEP_NAME={'LICENSE','Dockerfile','Makefile','CMakeLists.txt','.gitignore','.gitmodules','setup.sh','requirements.txt'}
SKIP_DIR={'nju','results','results_infer','training_data','data_for_train','data_test','new_data','new_data_staged','new_data_staged_bad','new_data_recover_unpromoted','checkpoints','trellis2_finetune_release','__pycache__','.git','node_modules','third_party','dist','build','.venv','venv','logs','output','outputs','wandb','.pytest_cache','.mypy_cache','.ipynb_checkpoints'}
DIAG_EXT={'.py','.md'}
def allowed(rel,path):
    if any(part in SKIP_DIR or part.startswith('.git') for part in rel.parts):return False
    if rel.parts and rel.parts[0]=='diagnostics' and path.suffix.lower() not in DIAG_EXT:return False
    if rel.parts and rel.parts[0]=='o-voxel' and len(rel.parts)>1 and rel.parts[1]=='third_party':return False
    return path.suffix.lower() in KEEP_EXT or path.name in KEEP_NAME
if DST.exists():raise SystemExit('destination exists: '+str(DST))
copied=[];skipped_size=0
for base,dirs,files in os.walk(SRC):
    dirs[:]=[d for d in dirs if d not in SKIP_DIR and not d.startswith('.git')]
    for name in files:
        path=Path(base)/name;rel=path.relative_to(SRC)
        if name.startswith('.export_manifest'):continue
        if not allowed(rel,path):continue
        try:size=path.stat().st_size
        except OSError:continue
        if size>MAX:skipped_size+=1;continue
        target=DST/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);copied.append(str(rel))
manifest={'files':len(copied),'bytes':sum((DST/f).stat().st_size for f in copied),'skipped_oversize':skipped_size,'list':sorted(copied)}
(DST/'.export_manifest.json').write_text(json.dumps(manifest,indent=1))
print(json.dumps({k:v for k,v in manifest.items() if k!='list'},indent=1))
