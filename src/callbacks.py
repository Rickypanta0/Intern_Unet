# src/callbacks.py

import tensorflow as tf
import numpy as np
from datetime import datetime


class ImageLogger(tf.keras.callbacks.Callback):
    """
    Callback personalizzato per loggare su TensorBoard immagini di input,
    ground truth e predizione ad ogni fine epoca.
    """
    def __init__(self, log_dir: str, x_sample: np.ndarray, y_sample: np.ndarray):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(f"{log_dir}/images")
        self.x_sample = x_sample
        self.y_sample = y_sample

    def on_epoch_end(self, epoch, logs=None):
        # Predizione sul campione di validazione
        pred_mask = self.model.predict(self.x_sample, verbose=0)[0]

        # Input image (RGB)
        input_img = (self.x_sample[0] * 255).astype(np.uint8)

        # Ground truth mask
        gt_mask = (self.y_sample[0] * 255).astype(np.uint8)
        if gt_mask.ndim == 2:
            gt_mask = np.expand_dims(gt_mask, axis=-1)
        gt_rgb = np.repeat(gt_mask, 3, axis=-1)

        # Prediction mask
        pred_mask = (pred_mask * 255).astype(np.uint8)
        if pred_mask.ndim == 2:
            pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_rgb = np.repeat(pred_mask, 3, axis=-1)

        # Concatenazione orizzontale: input | gt | pred
        display = np.concatenate([input_img, gt_rgb, pred_rgb], axis=1)
        display = np.expand_dims(display, 0)  # shape: (1, H, W, 3)

        with self.file_writer.as_default():
            tf.summary.image("Input / GT / Prediction", display, step=epoch)


def get_callbacks(
    log_dir: str,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    checkpoint_path: str = "model_for_nuclei.keras",
    patience: int = 3
) -> list[tf.keras.callbacks.Callback]:
    """
    Crea e restituisce la lista di callback per il training:
      - ModelCheckpoint: salva il miglior modello
      - ImageLogger: log delle immagini su TensorBoard per ogni epoch
      - EarlyStopping: ferma il training per overfitting
      - TensorBoard: log metriche e immagini

    Args:
        log_dir: directory base per i log di TensorBoard
        x_sample: batch di immagini per ImageLogger
        y_sample: batch di maschere per ImageLogger
        checkpoint_path: percorso file per il ModelCheckpoint
        patience: numero di epoche senza miglioramento prima di fermare

    Returns:
        Lista di istanze di tf.keras.callbacks.Callback
    """
    # TensorBoard
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_images=True
    )

    # Checkpoint
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=1
    )

    # ImageLogger
    image_logger_cb = ImageLogger(
        log_dir=log_dir,
        x_sample=x_sample,
        y_sample=y_sample
    )

    #Dynamic Learning rate
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
        )

    return [checkpoint_cb, earlystop_cb, tensorboard_cb, lr_cb]#, image_logger_cb]
