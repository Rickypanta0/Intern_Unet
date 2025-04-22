# src/utils/visualization.py
"""
Funzioni di visualizzazione per immagini e maschere.
"""
import numpy as np
import matplotlib.pyplot as plt


def show_threshold_pairs(imgs: np.ndarray,
                         gth: np.ndarray,
                         preds_list: list[np.ndarray],
                         thresholds: list[float]) -> None:
    """
    Per ciascuna soglia in `thresholds`, pesca un’immagine casuale da `imgs`
    e mostra a coppie: [Immagine originale | Maschera predetta].
    Si assume che `np.random.seed(...)` sia già stato settato all'inizio del flusso.

    Args:
        imgs: array di immagini shape (N, H, W, C)
        preds_list: lista di array di predizioni binarie corrispondenti alle soglie
        thresholds: lista di soglie float
    """
    n = len(thresholds)
    # estrai n indici casuali
    sample_idxs = np.random.randint(0, len(imgs), size=n)

    fig, axs = plt.subplots(n, 3,
                            figsize=(8, n * 4),
                            squeeze=False)

    for row, thr in enumerate(thresholds):
        idx_img = sample_idxs[row]
        pred_mask = preds_list[row][idx_img].squeeze()

        # Colonna 1: immagine originale
        axs[row, 0].imshow(imgs[idx_img])
        axs[row, 0].set_title("Original Image")
        axs[row, 0].axis("off")

        axs[row, 1].imshow(gth[idx_img], cmap='gray')
        axs[row, 1].set_title("GTh")
        axs[row, 1].axis("off")

        # Colonna 2: maschera predetta
        axs[row, 2].imshow(pred_mask, cmap='gray')
        axs[row, 2].set_title(f"Prediction @ {thr}")
        axs[row, 2].axis("off")

    plt.tight_layout()
    plt.show()

def show_threshold_pairs_test(imgs: np.ndarray,
                              gth: np.ndarray,
                              preds_list: list[np.ndarray],
                              thresholds: list[float]) -> None:
    """
    Mostra: [img originale | GT | pred@thr1] sulla prima riga,
            [pred@thr2 | pred@thr3 | ...] sulla seconda.
    """
    assert len(preds_list) == len(thresholds), "preds_list e thresholds devono avere la stessa lunghezza"

    # scegli un'immagine a caso
    idx_img = np.random.randint(0, len(imgs))

    # crea la figura
    n_preds = len(thresholds)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Riga 1: immagine originale, GT, prima soglia
    axs[0, 0].imshow(imgs[idx_img])
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gth[idx_img], cmap='gray')
    axs[0, 1].set_title("Ground Truth")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(preds_list[0][idx_img].squeeze(), cmap='gray')
    axs[0, 2].set_title(f"Prediction @ {thresholds[0]}")
    axs[0, 2].axis("off")

    # Riga 2: le altre predizioni
    for col in range(3):
        if col + 1 >= n_preds:
            axs[1, col].axis("off")
        else:
            axs[1, col].imshow(preds_list[col + 1][idx_img].squeeze(), cmap='gray')
            axs[1, col].set_title(f"Prediction @ {thresholds[col + 1]}")
            axs[1, col].axis("off")

    plt.tight_layout()
    plt.show()