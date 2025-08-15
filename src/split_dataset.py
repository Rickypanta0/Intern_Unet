import os
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------
BASE_DIR   = Path("data/raw_neoplastic")
FOLD_NAME  = "Fold 3"          # puoi cambiarlo
VAL_RATIO  = 0.10              # 10 % in validation
SEED       = 42

# ---------------------------------------------------------------------
# Path utili
# ---------------------------------------------------------------------
fold_dir      = BASE_DIR / FOLD_NAME
img_path      = fold_dir / "images" / "images.npy"
dist_path     = fold_dir / "masks"  / "distance.npy"
maskneo_path  = fold_dir / "masks"  / "masks_neo.npy"

val_dir       = fold_dir / "validation"
val_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1) Carica in mem‑map (niente RAM sprecata)
# ---------------------------------------------------------------------
imgs   = np.load(img_path,      mmap_mode="r")
dists  = np.load(dist_path,     mmap_mode="r")
masks  = np.load(maskneo_path,  mmap_mode="r")

N = imgs.shape[0]
assert dists.shape[0] == N and masks.shape[0] == N, "Array non allineati!"

# ---------------------------------------------------------------------
# 2) Seleziona indici per la validation
# ---------------------------------------------------------------------
rng     = np.random.default_rng(SEED)
k       = int(round(N * VAL_RATIO))
val_idx = rng.choice(N, size=k, replace=False)
train_idx = np.setdiff1d(np.arange(N), val_idx)

print(f"Split: {k} in validation, {len(train_idx)} rimangono per il training")

# ---------------------------------------------------------------------
# 3) Salva i file di validation
# ---------------------------------------------------------------------
np.save(val_dir / "images.npy",      imgs[val_idx])
np.save(val_dir / "distance.npy",    dists[val_idx])
np.save(val_dir / "masks_neo.npy",   masks[val_idx])
print(f"✔  Salvati i campioni di validation in {val_dir}")

# ---------------------------------------------------------------------
# 4) Crea i nuovi array train (sovrascrive i vecchi file)
#    NB: chiudo i mem‑map prima di scrivere sugli stessi path
# ---------------------------------------------------------------------
del imgs, dists, masks      # libera i file descriptor

# ricarica in RAM solo ciò che serve e sovrascrive
np.save(img_path,     np.load(img_path,     mmap_mode="r")[train_idx])
np.save(dist_path,    np.load(dist_path,    mmap_mode="r")[train_idx])
np.save(maskneo_path, np.load(maskneo_path, mmap_mode="r")[train_idx])
print("✔  Aggiornati i file originali (senza i campioni di validation)")
