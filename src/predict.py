# src/predict.py
"""
Modulo per caricare un modello salvato, eseguire predizioni su dataset e
restituire maschere binarie con diverse soglie.
"""
import os
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split


def load_model(checkpoint_path: str) -> tf.keras.Model:
    """
    Carica un modello Keras da file .keras o analoghi.

    Args:
        checkpoint_path: percorso del file di checkpoint
    Returns:
        Modello compilato
    """
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    # Se hai bisogno di ricompilare con le tue loss/metriche:
    # from src.losses import csca_binary_loss
    # from keras.metrics import MeanIoU
    # model.compile(optimizer='adam', loss=csca_binary_loss, metrics=[MeanIoU(num_classes=2)])
    return model


def predict_masks(
    model: tf.keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    split: float = 0.9,
    thresholds: list[float] = [0.3, 0.4, 0.5,0.6]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Genera predizioni binarie sul train, validation e test con soglie multiple.

    Args:
        model: modello Keras già compilato e con pesi caricati
        X_train: array delle immagini di train
        X_test:  array delle immagini di test
        split: frazione di train da usare come validazione interna
        thresholds: lista di soglie tra 0 e 1
    Returns:
        Ptrain, Pval, Ptest: tre liste di array di predizioni binarie
    """
    n = len(thresholds)
    # Calcolo indice split
    idx_split = int(split * len(X_train))

    Ptrain = [None] * n
    Pval   = [None] * n
    Ptest  = [None] * n

    for i, thr in enumerate(thresholds):
        # predizioni probabilistiche
        prob_train = model.predict(X_train[:idx_split], verbose=1)
        prob_val   = model.predict(X_train[idx_split:],   verbose=1)
        prob_test  = model.predict(X_test,                verbose=1)

        # applichiamo soglia e convertiamo in uint8
        Ptrain[i] = (prob_train > thr).astype(np.uint8)
        Pval[i]   = (prob_val   > thr).astype(np.uint8)
        Ptest[i]  = (prob_test  > thr).astype(np.uint8)

    return Ptrain, Pval, Ptest


def save_predictions(
    output_dir: str,
    Ptrain: list[np.ndarray],
    Pval: list[np.ndarray],
    Ptest: list[np.ndarray],
    thresholds: list[float]
) -> None:
    """
    Salva le predizioni come .npy per ciascuna soglia.

    Directory di output:
        output_dir/
            train_thr0.3.npy
            val_thr0.3.npy
            test_thr0.3.npy
            train_thr0.4.npy
            ...
    """
    os.makedirs(output_dir, exist_ok=True)
    for thr, p_tr, p_v, p_te in zip(thresholds, Ptrain, Pval, Ptest):
        np.save(os.path.join(output_dir, f"train_thr{thr:.2f}.npy"), p_tr)
        np.save(os.path.join(output_dir, f"val_thr{thr:.2f}.npy"),   p_v)
        np.save(os.path.join(output_dir, f"test_thr{thr:.2f}.npy"),  p_te)


if __name__ == "__main__":
    # Esempio d'uso minimale
    from src.train import train  # se vuoi lanciare training prima

    # 1) Carica modello
    checkpoint = os.path.join("models", "checkpoints", "model_for_nuclei.keras")
    model = load_model(checkpoint)

    # 2) Carica dati (X_train, X_test devono essere disponibili)
    from src.data_loader import load_folds
    BASE = os.path.join('..', 'dataSet')
    folds = [
        (os.path.join(BASE, 'Fold 1', 'images', 'fold1', 'images.npy'), os.path.join(BASE, 'Fold 1', 'masks', 'fold1', 'masks_processed.npy')),
        (os.path.join(BASE, 'Fold 2', 'images', 'fold2', 'images.npy'), os.path.join(BASE, 'Fold 2', 'masks', 'fold2', 'masks_processed.npy'))
    ]
    X, Y = load_folds(folds)
    X_train, X_test, _, _ = train_test_split(X, Y, test_size=0.1, random_state=42)

    # 3) Predizioni
    thresholds = [0.3, 0.4, 0.5]
    Ptrain, Pval, Ptest = predict_masks(model, X_train, X_test, split=0.9, thresholds=thresholds)

    # 4) Salvataggio
    save_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_predictions(save_dir, Ptrain, Pval, Ptest, thresholds)