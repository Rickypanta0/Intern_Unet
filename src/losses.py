# src/losses.py
"""
Moduli per le funzioni di perdita custom:
- Dice Loss
- BCE + Dice Loss
- Weighted BCE + Dice Loss
- CSCA Binary Loss (composita)
"""
import tensorflow as tf
import numpy as np


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Calcola la Dice Loss tra y_true e y_pred.

    Args:
        y_true: tensor di forma [B, H, W, 1], valori binari o continui in [0,1]
        y_pred: tensor di forma [B, H, W, 1], predizioni continui in [0,1]
        smooth: fattore di smoothing per evitare divisione per zero

    Returns:
        Dice Loss (1 - dice coefficient)
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combina Binary Crossentropy e Dice Loss.

    L = BCE(y_true, y_pred) + DiceLoss(y_true, y_pred)
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def weighted_bce_dice(y_true: tf.Tensor, y_pred: tf.Tensor,
                      weight: float = 5.0,
                      smooth: float = 1.0) -> tf.Tensor:
    """
    Perdita composita pesata:
      - BCE pixel-wise con mappa di pesi
      - Dice Loss

    Args:
        y_true: tensor [B,H,W,1]
        y_pred: tensor [B,H,W,1]
        weight: peso applicato ai pixel di foreground
        smooth: smoothing per Dice
    Returns:
        weighted BCE + Dice Loss
    """
    # BCE pixel-wise
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Mappa di pesi
    weight_map = tf.squeeze(y_true * weight + (1.0 - y_true), axis=-1)
    weighted_bce = bce * weight_map
    # Dice Loss
    y_t = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_t * y_p)
    dice = 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)
    return tf.reduce_mean(weighted_bce) + dice


def get_weighted_bce_dice_loss(weight: float = 5.0) -> callable:
    """
    Restituisce una funzione di perdita parzializzata di weighted_bce_dice con peso fisso.
    """
    return lambda y_true, y_pred: weighted_bce_dice(y_true, y_pred, weight=weight)

# Parametri default per CSCA
_lambda1 = 2.0
_lambda2 = 1.0
_smooth = 1e-6


def csca_binary_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Perdita composita CSCA per due classi (background vs foreground):
      L = λ1 * BCE + λ2 * DiceLoss(FG)

    Args:
        y_true: tensor [B,H,W,1]
        y_pred: tensor [B,H,W,1]
    Returns:
        Valore scalare della perdita composita
    """
    # Binary Crossentropy medio
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    # Dice sul foreground
    y_t = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_t * y_p)
    dice_coeff = (2.0 * intersection + _smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + _smooth)
    dice_loss_val = 1.0 - dice_coeff
    return _lambda1 * bce + _lambda2 * dice_loss_val
