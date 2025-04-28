# src/data_loader.py
"""
Moduli per il caricamento e preprocessing dei dati.
"""
import os
from typing import List, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt


def load_npy_file(path: str, normalize: bool = True) -> np.ndarray:
    """
    Carica un file .npy e, se richiesto, normalizza i valori su [0,1].

    Args:
        path: percorso del file .npy
        normalize: se True divide per 255.0

    Returns:
        array NumPy di dimensioni variabili
    """
    data = np.load(path)
    if normalize:
        data = data.astype(np.float32) / 255.0
    return data


def load_folds(folds: List[Tuple[str, str]], normalize_images: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica e concatena immagini e maschere da più fold.

    Args:
        folds: lista di tuple (path_immagini, path_maschere)
        normalize_images: se True normalizza le immagini

    Returns:
        X: array shape (N, H, W, C)
        Y: array shape (N, H, W, ...)
    """
    X_list, Y_list = [], []
    for img_path, mask_path in folds:
        print(f"> Caricamento immagini: {img_path}")
        imgs = load_npy_file(img_path, normalize=normalize_images)
        X_list.append(imgs)

        print(f"> Caricamento maschere: {mask_path}")
        masks = np.load(mask_path)
        Y_list.append(masks)

    # Concatenazione
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    print(f"Dati caricati: X={X.shape}, Y={Y.shape}")
    return X, Y


def show_random_sample(X: np.ndarray, Y: np.ndarray, seed: int = None) -> None:
    """
    Visualizza un campione casuale di immagine e maschera.

    Args:
        X: array immagini (N, H, W, C)
        Y: array maschere (N, H, W, ...)
        seed: seed per la riproducibilità
    """
    if seed is not None:
        random.seed(seed)
    idx = random.randint(0, X.shape[0] - 1)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X[idx])
    plt.title("Immagine")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(Y[idx].squeeze())
    plt.title("Maschera")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Esempio di utilizzo
    base = os.path.join('..', 'data','raw')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'))
    ]
    X, Y = load_folds(folds)
    show_random_sample(X, Y, seed=42)



