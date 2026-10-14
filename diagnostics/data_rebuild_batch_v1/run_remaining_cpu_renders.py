import json,subprocess,sys
from pathlib import Path
root=Path('/data/lzx/TRELLIS.2');b=root/'diagnostics/data_rebuild_batch_v1'
proof=json.loads((b/'first_render_validation_v2/first_render_validation.json').read_text())
if proof.get('passed') is not True:raise ValueError('first actual render validation has not passed')
reviews=json.loads((b/'asset_review/review.json').read_text())
for e in reviews['selected'][1:]:
 oid=e['object_id'];cmd=['/usr/bin/python3',str(b/'run_stage_bounded.py'),'--command',str(b/f'render_{oid}_cpu_v1.command.json'),'--record-prefix',str(b/f'render_{oid}_cpu_v1_execution'),'--timeout','600']
 rc=subprocess.call(cmd,cwd=root)
 if rc:sys.exit(rc)
print('ALL_FIVE_REMAINING_RENDERS_FINISHED',flush=True)
