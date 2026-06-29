"""Preflight smoke for the FineWeb-Edu corpus override — run BEFORE the paid FineWeb A/B.

Verifies (no GPU, no network):
  (1) maybe_repoint() is a no-op without NANOCHAT_DATASET=fineweb, and repoints all THREE
      nanochat.dataset constants (BASE_URL, DATA_DIR, MAX_SHARD) with it.
  (2) the val split resolves to shard_01822 from the staged local FineWeb data.
  (3) both FineWeb tokenizers (surface_only_fineweb + baseline_fineweb) load / round-trip.

    uv run python -m wrappers.smoke_fineweb
"""
import os
import sys

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


os.environ.pop("NANOCHAT_DATASET", None)                      # ensure unset for the no-op check
os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

# --- (1) override semantics ---
import nanochat.dataset as ds
from overlay.dataset_fineweb import maybe_repoint
cm_url, cm_dir = ds.BASE_URL, ds.DATA_DIR
check("ClimbMix is the default", "climbmix" in cm_dir and "climbmix" in cm_url, os.path.basename(cm_dir))
check("no-op without env", maybe_repoint() is False and ds.DATA_DIR == cm_dir)

os.environ["NANOCHAT_DATASET"] = "fineweb"
check("repoints with env", maybe_repoint() is True)
check("BASE_URL → fineweb", "fineweb-edu-100b-shuffle" in ds.BASE_URL)
check("DATA_DIR → base_data_fineweb", ds.DATA_DIR.endswith("base_data_fineweb"))
check("MAX_SHARD → 1822", ds.MAX_SHARD == 1822, str(ds.MAX_SHARD))

# --- (2) val split = shard_01822 (the highest staged shard) ---
from nanochat.dataset import list_parquet_files
files = [os.path.basename(p) for p in list_parquet_files()]
val_file = files[-1] if files else None
check("val split = shard_01822", val_file == "shard_01822.parquet",
      f"last sorted = {val_file} ({len(files)} shards staged)")

# --- (3) FineWeb tokenizers load / round-trip ---
from nanochat.tokenizer import RustBPETokenizer
from overlay.surface_tokenizer import SurfaceTokenizer
from tools.stack_fm_spaceless import SPACELESS_BODY
H = os.path.expanduser("~")
sample = "The National Influenza Vaccination Week began in Ouagadougou."
try:
    b = RustBPETokenizer.from_directory(f"{H}/.cache/nanochat-variants/baseline_fineweb/tokenizer")
    check("baseline FineWeb tok round-trips", b.enc.decode(b.enc.encode_ordinary(sample)) == sample
          and b.enc.n_vocab == 32768, f"vocab {b.enc.n_vocab}")
except Exception as e:
    check("baseline FineWeb tok round-trips", False, str(e)[:60])
try:
    s = SurfaceTokenizer.from_directory(f"{H}/.cache/nanochat-variants/surface_only_fineweb/tokenizer",
                                        SPACELESS_BODY, kmax=1)
    check("surface FineWeb tok loads (32768)", s.n_vocab == 32768, f"vocab {s.n_vocab}")
except Exception as e:
    check("surface FineWeb tok loads (32768)", False, str(e)[:60])

print("\n" + ("SMOKE FAIL: " + ", ".join(FAIL) if FAIL else
             "SMOKE OK — FineWeb override repoints, val=shard_01822, both tokenizers ready"))
sys.exit(1 if FAIL else 0)
