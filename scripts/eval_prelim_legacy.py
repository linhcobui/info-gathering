"""    pixi run python scripts/eval_prelim_legacy.py decoder_run=<prelim pixel decoder run> [zarr_path=... outdir=...]

Run scripts/eval_memtest1.py on an OLD smt_prelim checkpoint that predates modules the current
code builds but those runs never used (teacher_dec.to_ball; the transformer `updater` of smtcell).
Loads non-strictly and refuses any missing/unexpected key outside that allow-list."""
import runpy, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from world_model.models.smt_prelim.model import SmtPrelimWM
ALLOWED = ("teacher_dec.to_ball.", "updater.")
def load(self, state):
    missing, unexpected = self.load_state_dict(state["modules"], strict=False)
    bad = [k for k in missing if not k.startswith(ALLOWED)] + list(unexpected)
    if bad:
        raise RuntimeError(f"legacy load: unexpected key mismatch {bad[:5]}")
    self._step = int(state.get("step", 0))
    print(f"legacy load: {len(missing)} allow-listed keys left at init: {sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in missing})}")
SmtPrelimWM.load_train_state_dict = load
sys.argv = ["scripts/eval_memtest1.py", "--config-dir", "configs/smt_prelim", "--config-name", "eval_memtest4_prelim"] + sys.argv[1:]
runpy.run_path("scripts/eval_memtest1.py", run_name="__main__")
