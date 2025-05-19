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

from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
def hover_loss_fixed(y_true, y_pred):
    return hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0)(y_true, y_pred)

@register_keras_serializable()
def hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0):
    """
    Loss per la testa HV: combina MSE (L2) e gradient loss.
    """
    def gradient_x(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def gradient_y(img):
        return img[:, 1:, :, :] - img[:, :-1, :, :]

    def loss(y_true, y_pred):
        # MSE loss (LH1)
        mse_loss = K.mean(K.square(y_true - y_pred))

        # Gradient loss (LH2)
        gx_true = gradient_x(y_true)
        gx_pred = gradient_x(y_pred)
        gy_true = gradient_y(y_true)
        gy_pred = gradient_y(y_pred)

        grad_loss_x = K.mean(K.square(gx_pred - gx_true))
        grad_loss_y = K.mean(K.square(gy_pred - gy_true))
        grad_loss = grad_loss_x + grad_loss_y

        return lambda_h1 * mse_loss + lambda_h2 * grad_loss

    return loss
@register_keras_serializable()
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

@register_keras_serializable()
def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combina Binary Crossentropy e Dice Loss.

    L = BCE(y_true, y_pred) + DiceLoss(y_true, y_pred)
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

@register_keras_serializable()
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

@register_keras_serializable()
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
@register_keras_serializable()
def dice_loss_class(y_true, y_pred, k, smooth=1e-6):
    t = y_true[..., k]
    p = y_pred[..., k]
    inter = tf.reduce_sum(t * p, axis=[1,2])
    denom = tf.reduce_sum(t, axis=[1,2]) + tf.reduce_sum(p, axis=[1,2])
    return 1 - (2*inter + smooth)/(denom + smooth)
@register_keras_serializable()
def Lp(y_true, y_pred):
    # Cp = CE multiclasse
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    Cp  = tf.reduce_mean(cce)
    # D1 sul canale BD (k=0), D2 sul canale CB (k=2)
    D1  = tf.reduce_mean(dice_loss_class(y_true, y_pred, k=0))
    D2  = tf.reduce_mean(dice_loss_class(y_true, y_pred, k=2))
    return 2*Cp + 1*D1 + 2*D2

# ————————————————————————————————
# 2) Metriche binarie “nucleo vs background”:
#    consideriamo “nucleo” i canali BD (0) e CB (2), background = canale BG (1)
@register_keras_serializable()
def binary_iou(y_true, y_pred):
    # y_true, y_pred: (B,H,W,3) one-hot/softmax
    # facciamo due maschere booleane: nucleo vs non-nucleo
    true_labels = tf.argmax(y_true, axis=-1)    # 0=BD,1=BG,2=CB
    pred_labels = tf.argmax(y_pred, axis=-1)
    true_bin = tf.cast(tf.not_equal(true_labels, 1), tf.int32)
    pred_bin = tf.cast(tf.not_equal(pred_labels, 1), tf.int32)
    # intersection & union per batch
    intersection = tf.reduce_sum(tf.cast(true_bin & pred_bin, tf.float32), axis=[1,2])
    union        = tf.reduce_sum(tf.cast(true_bin | pred_bin, tf.float32), axis=[1,2])
    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(iou)