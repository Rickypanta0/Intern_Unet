# src/train.py
"""
Modulo per orchestrare il training: carica i dati, costruisce il modello,
configura i callback e lancia il fit.
"""
import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.data_loader import load_folds
from src.model import get_model_BB,get_model
from src.callbacks import get_callbacks


def train(
    X_train,
    Y_train,
    X_test,
    Y_test,
    test_size: float = 0.1,
    random_seed: int = 42,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 50
) -> any:
    """
    Esegue il training completo:
      1. Caricamento e split dei dati
      2. Costruzione e compilazione del modello
      3. Setup dei callback
      4. Chiamata a model.fit
      5. Salvataggio e valutazione su test set

    Args:
        folds: lista di tuple (path_img, path_mask)
        test_size: frazione per test split
        random_seed: seed per split e sample
        learning_rate: lr per Adam
        batch_size: dimensione del batch
        epochs: numero di epoche
    Returns:
        history Keras di model.fit
        model 
    """
    # 1) Caricamento dati
    #X, Y = load_folds(folds, normalize_images=True)
    #Y = np.expand_dims(Y, axis=-1)
    #X_train, X_test, Y_train, Y_test = train_test_split(
    #    X, Y, test_size=test_size, random_state=random_seed
    #)

    # 2) Modello
    model = get_model(input_shape=X_train.shape[1:], learning_rate=learning_rate)

    # 3) Campione per ImageLogger
    idx = np.random.randint(len(X_train))
    x_sample = X_train[idx:idx+1]
    y_sample = Y_train[idx:idx+1]

    # 4) Callback
    log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint = os.path.join("models", "checkpoints", "model_for_nuclei.keras")
    callbacks = get_callbacks(
        log_dir=log_dir,
        x_sample=x_sample,
        y_sample=y_sample,
        checkpoint_path=checkpoint,
        patience=2
    )

    # 5) Fit
    history = model.fit(
        X_train, Y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # 6) Salvataggio finale
    os.makedirs("models/final", exist_ok=True)
    model.save("models/final/unet_final.keras")

    # 7) Valutazione
    loss, miou = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test loss: {loss:.4f} — Test MeanIoU: {miou:.4f}")

    return model, history


if __name__ == "__main__":
    # Esempio di utilizzo
    base = os.path.join('..', 'dataSet')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'fold1', 'masks_processed.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'fold2', 'masks_processed.npy'))
    ]
    train(folds)