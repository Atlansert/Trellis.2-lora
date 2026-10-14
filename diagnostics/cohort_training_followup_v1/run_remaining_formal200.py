import subprocess,sys
from pathlib import Path
r=Path('/data/lzx/TRELLIS.2');b=r/'diagnostics/cohort_training_followup_v1'
for n in ['001','003','004']:
 code=subprocess.call(['/usr/bin/python3',str(r/'diagnostics/data_rebuild_batch_v1/run_stage_bounded.py'),'--command',str(b/f'formal_step200_{n}.command.json'),'--record-prefix',str(b/f'formal_step200_{n}_execution'),'--timeout','600'],cwd=r)
 if code:sys.exit(code)
print('REMAINING_FORMAL200_COMPLETED',flush=True)
