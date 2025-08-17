# src/losses.py
"""
Moduli per le funzioni di perdita custom:
- Dice Loss
- BCE + Dice Loss
- Weighted BCE + Dice Loss
- CSCA Binary Loss (composita)
"""
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
def hover_loss_fixed(y_true, y_pred):
    return hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0)(y_true, y_pred)#Oppure mettere direttamete 4.0

@register_keras_serializable()
def hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0):
    
    #Loss per la testa HV: combina MSE (L2) e gradient loss.
    
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

dice_loss_fn = tf.keras.losses.Dice(
    reduction=tf.keras.losses.Reduction.NONE,  # <‑‑ niente media interna
    name="dice"
)

@register_keras_serializable()
def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    
    #Combina Binary Crossentropy e Dice Loss.

    #L = BCE(y_true, y_pred) + DiceLoss(y_true, y_pred)
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred),
        axis=[1, 2]                          # riduci H, W, C
    ) 
    dice_per_img = dice_loss_fn(y_true, y_pred)
    #bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_per_img


@register_keras_serializable()
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    
    #Calcola la Dice Loss tra y_true e y_pred.

    #Args:
    #    y_true: tensor di forma [B, H, W, 1], valori binari o continui in [0,1]
    #    y_pred: tensor di forma [B, H, W, 1], predizioni continui in [0,1]
    #    smooth: fattore di smoothing per evitare divisione per zero
#
    #Returns:
    #    Dice Loss (1 - dice coefficient)
    
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)



def sobel_xy(t):
    # t: (B,H,W,2) -> applico sobel a ciascun canale, poi concateno
    # Sobel app standard via tf.image.sobel_edges (ritorna [..., 2, 2]: dy, dx)
    # Qui estraggo dx,dy e li concateno per i 2 canali HV.
    se = tf.image.sobel_edges(t)          # (B,H,W,2,2)
    dy = se[..., 0]                       # (B,H,W,2)
    dx = se[..., 1]                       # (B,H,W,2)
    return dx, dy

@tf.keras.utils.register_keras_serializable()
def hover_loss(y_true, y_pred):
    """
    y_true: (B,H,W,3) con [HVx_true, HVy_true, mask_fg]  oppure
            (B,H,W,2) con HV e mask_fg separata altrove: adatta di conseguenza.
    y_pred: (B,H,W,2) con [HVx_pred, HVy_pred]
    """
    # separa HV_true e mask_fg
    if y_true.shape[-1] == 3:
        hv_t   = y_true['hv_head']
        body, bg, border = tf.unstack(y_true['seg_head'], axis=-1)
        mask_fg = tf.cast(body + border > bg, tf.float32)
        #mask_fg = y_true[..., 2:3]     # (B,H,W,1), 1 nei nuclei, 0 altrove
    else:
        hv_t   = y_true
        # SE non hai mask in y_true, metti tutto a 1 (meno consigliato)
        mask_fg = tf.ones_like(hv_t[..., :1])

    hv_p = y_pred['hv_head']

    # Huber sui valori HV (masked)
    huber = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.NONE)
    map_loss = huber(hv_t, hv_p)                     # (B,H,W,2)
    map_loss = tf.reduce_sum(map_loss * mask_fg) / (tf.reduce_sum(mask_fg) + 1e-6)

    # Gradiente (Sobel) su HV
    dx_p, dy_p = sobel_xy(hv_p)
    dx_t, dy_t = sobel_xy(hv_t)

    grad_loss = huber(dx_t, dx_p) + huber(dy_t, dy_p)   # (B,H,W,2)
    # peso extra su edge: approx = grad della mask_fg
    edge = tf.image.sobel_edges(mask_fg)[...,0]**2 + tf.image.sobel_edges(mask_fg)[...,1]**2
    edge = tf.reduce_sum(edge, axis=-1, keepdims=True)    # (B,H,W,1)
    edge = edge / (tf.reduce_max(edge) + 1e-6)
    w = 0.5*mask_fg + 0.5*edge                            # metà fg, metà edge

    grad_loss = tf.reduce_sum(grad_loss * w) / (tf.reduce_sum(w) + 1e-6)

    # pesi come nel paper/implementazioni comuni
    return map_loss + 2.0 * grad_loss

@tf.keras.utils.register_keras_serializable()
def hovernet_hv_loss_tf(y_true, y_pred,
                        lambda_hv_mse=2.0,
                        lambda_hv_mse_grad=1.0):
    """
    y_true: (B,H,W,3) -> [HVx_true, HVy_true, focus_mask]
    y_pred: (B,H,W,2) -> [HVx_pred, HVy_pred]
    focus_mask: 1 sui nuclei (body|border), 0 sul background
    """
    print(y_pred)
    # separa target e mask
    hv_t   = tf.cast(y_true[..., :2], tf.float32)   # (B,H,W,2)
    focus  = tf.cast(y_true[..., 2:3], tf.float32)  # (B,H,W,1) 0/1
    hv_p   = tf.cast(y_pred, tf.float32)            # (B,H,W,2)

    # ---------- 1) MSE HV (mascherata) ----------
    mse_map = tf.square(hv_p - hv_t)                        # (B,H,W,2)
    mse_map = tf.reduce_sum(mse_map * focus)                # somma su H,W,2
    denom   = tf.reduce_sum(focus) * 2.0 + 1e-8            # *2 per due canali
    loss_mse = mse_map / denom

    # ---------- 2) MSE gradiente HV (mascherata) ----------
    # tf.image.sobel_edges -> (B,H,W,C,2) dove ...,[dy, dx]
    se_p = tf.image.sobel_edges(hv_p)   # (B,H,W,2,2)
    se_t = tf.image.sobel_edges(hv_t)   # (B,H,W,2,2)

    # MONAI: grad orizzontale del canale H (dx su canale 0) e grad verticale del canale V (dy su canale 1)
    grad_h_p = se_p[..., 0, 1]   # canale 0, dx
    grad_h_t = se_t[..., 0, 1]
    grad_v_p = se_p[..., 1, 0]   # canale 1, dy
    grad_v_t = se_t[..., 1, 0]

    grad_p = tf.stack([grad_h_p, grad_v_p], axis=-1)  # (B,H,W,2)
    grad_t = tf.stack([grad_h_t, grad_v_t], axis=-1)

    grad_se = tf.square(grad_p - grad_t)             # (B,H,W,2)
    grad_se = tf.reduce_sum(grad_se * focus)         # somma su H,W,2
    loss_grad = grad_se / denom

    return lambda_hv_mse * loss_mse + lambda_hv_mse_grad * loss_grad