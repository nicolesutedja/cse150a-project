"""
make_splits.py — Create reproducible train/val/test splits of valid_solutions.txt.

Shuffle with seed 42, then cut 70% train / 15% val / 15% test.
Writes three files to data/:
    split_train.txt
    split_val.txt
    split_test.txt
"""

import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
SOLUTIONS_PATH = DATA_DIR / "valid_solutions.txt"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test gets the remainder

words = SOLUTIONS_PATH.read_text().splitlines()
words = [w.strip() for w in words if w.strip()]

rng = random.Random(SEED)
rng.shuffle(words)

n = len(words)
n_train = int(n * TRAIN_FRAC)
n_val   = int(n * VAL_FRAC)

train = words[:n_train]
val   = words[n_train : n_train + n_val]
test  = words[n_train + n_val :]

(DATA_DIR / "split_train.txt").write_text("\n".join(train) + "\n")
(DATA_DIR / "split_val.txt").write_text("\n".join(val) + "\n")
(DATA_DIR / "split_test.txt").write_text("\n".join(test) + "\n")

print(f"Total: {n} words")
print(f"Train: {len(train)}  ({len(train)/n:.1%})")
print(f"Val:   {len(val)}  ({len(val)/n:.1%})")
print(f"Test:  {len(test)}  ({len(test)/n:.1%})")
