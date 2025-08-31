# src/losses.py
# ====== backend Torch ======
import os
os.environ["KERAS_BACKEND"] = "torch"  # <<— fondamentale: prima di importare keras

import keras
from keras import layers, ops
from keras.utils import register_keras_serializable

# (opzionale) torch per le loss custom
import torch
import torch.nn.functional as F

@register_keras_serializable()
def bce_dice_loss_ops(y_true, y_pred, smooth=1e-6):
    """
    Esempio semplice: CE multiclass + soft Dice sul canale "cella" (body+border).
    Adatta se hai 3 classi [body, bg, border] con softmax.
    """
    # CE multiclass (usa keras.ops sotto il cofano)
    ce = keras.losses.categorical_crossentropy(y_true, y_pred)

    # soft-dice su "cella" = body+border
    cell_true = y_true[..., 0] + y_true[..., 2]     # [B,H,W]
    cell_pred = y_pred[..., 0] + y_pred[..., 2]     # [B,H,W] (prob)

    inter = ops.sum(cell_true * cell_pred, axis=(-2, -1))  # somma su H,W (mantieni batch)
    denom = ops.sum(cell_true, axis=(-2, -1)) + ops.sum(cell_pred, axis=(-2, -1))
    dice = (2.0 * inter + smooth) / (denom + smooth)

    # CE è per-pixel → media su batch; dice è per immagine → 1 - mean
    ce_mean = ops.mean(ce)
    dice_loss = 1.0 - ops.mean(dice)
    return ce_mean + dice_loss

import keras
import torch
import torch.nn.functional as F
from monai.transforms import SobelGradients
import matplotlib.pyplot as plt
@keras.saving.register_keras_serializable(package="losses")
class HVLossMonaiTorch(keras.losses.Loss):
    """
    y_true: (B,H,W,3) -> [H_gt, V_gt, focus_mask]
    y_pred: (B,H,W,2) -> [H_pred, V_pred]
    Loss = lambda_mse * MSE(H,V) + lambda_grad * MSE(∂x H, ∂y V) mascherata con focus.
    """
    def __init__(self, lambda_mse: float = 2.0, lambda_grad: float = 1.0, ksize: int = 5, name="hv_monai_loss"):
        # In Keras 3 usa stringhe per 'reduction' o ometti (default = "sum_over_batch_size")
        super().__init__(name=name, reduction="sum_over_batch_size")
        if ksize % 2 == 0 or ksize < 3:
            raise ValueError("ksize deve essere dispari e >= 3")
        self.lambda_mse  = float(lambda_mse)
        self.lambda_grad = float(lambda_grad)
        self.ksize = int(ksize)
        # Lazy init per evitare meta tensor
        self._sobel_h = None  # ∂/∂x (asse orizzontale)
        self._sobel_v = None  # ∂/∂y (asse verticale)

    def _ensure_sobels(self):
        if self._sobel_h is None or self._sobel_v is None:
            # Non passiamo dtype qui: MONAI porta i kernel sul device/dtype dell'input al volo
            self._sobel_h = SobelGradients(kernel_size=self.ksize, spatial_axes=1)  # dx
            self._sobel_v = SobelGradients(kernel_size=self.ksize, spatial_axes=0)  # dy

    def call(self, y_true, y_pred):
        # backend torch → tensori torch
        assert isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)

        h_t = y_true[..., 0]   # (B,H,W)
        v_t = y_true[..., 1]
        focus = y_true[..., 2] # (B,H,W) in {0,1}
        h_p = y_pred[..., 0]
        v_p = y_pred[..., 1]

        self._ensure_sobels()

        # MSE sulle mappe
        #mse_maps = torch.mean((h_p - h_t)**2 + (v_p - v_t)**2)
        mse_maps = F.mse_loss(y_pred, y_true[...,:2])
        # Gradienti con MONAI SobelGradients (accetta (B,H,W), restituisce (B,H,W))
        gh_p = self._sobel_h(h_p)
        gh_t = self._sobel_h(h_t)
        gv_p = self._sobel_v(v_p)
        gv_t = self._sobel_v(v_t)
        
        gdiff2 = (gh_p - gh_t).pow(2) + (gv_p - gv_t).pow(2)  # (B,H,W)

        num = torch.sum(gdiff2 * focus)
        den = torch.sum(focus) + 1e-8
        mse_grad = num / den

        return self.lambda_mse * mse_maps + self.lambda_grad * mse_grad
"""
Moduli per le funzioni di perdita custom:
- Dice Loss
- BCE + Dice Loss
- Weighted BCE + Dice Loss
- CSCA Binary Loss (composita)

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
import numpy as np

@register_keras_serializable()
def hover_loss_fixed(y_true, y_pred):
    y_true_ = y_true[:, :, :, :2]
    focus = y_true[:, :, :, 2]
    return hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0)(y_true_, y_pred, focus)

@register_keras_serializable()
def hover_mse_grad_loss(lambda_h1=1.0, lambda_h2=2.0):
    
    Loss per la testa HV: combina MSE (L2) e gradient loss.
    def gradient_x_(img):
        # dx (secondo TF: sobel[..., 0,1] = ∂x)
        sob = tf.image.sobel_edges(img)  # (B,H,W,2,C) con ordine [dy, dx]
        return sob[..., 0, 1]          # tieni la dimensione canale
    
    def gradient_y_(img):
#       # dy (sobel[..., 1,0] = ∂y)
        sob = tf.image.sobel_edges(img)
        return sob[..., 1, 0]
    
    def gradient_x(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def gradient_y(img):
        return img[:, 1:, :, :] - img[:, :-1, :, :]
    
    def loss(y_true, y_pred, focus):
        # MSE loss (LH1)
        mse_loss = K.mean(K.square(y_true - y_pred))
        # Gradient loss (LH2)
        gx_true = gradient_x(y_true)
        gx_pred = gradient_x(y_pred)
        gy_true = gradient_y(y_true)
        gy_pred = gradient_y(y_pred)
        #\gy_true_ = gradient_y_(hv_t_s)
        #\gx_true_ = gradient_x_(hv_t_s)
        #\#print(gx_true.shape,gx_true.shape,gy_pred.shape, gy_true.shape )
        #\fig, axs = plt.subplots(2,2,figsize=(8,8))
        #\axs[0,0].imshow(gx_true[0,...,0])
        #\axs[0,1].imshow(gy_true[0,...,0])
        #\axs[1,0].imshow(gx_true_[0])
        #\axs[1,1].imshow(gy_true_[0])
        #\plt.show()

        focus = tf.expand_dims(focus, axis=-1)
        focus = tf.concat([focus, focus], axis=-1)
        grad_loss_x = K.mean(K.square(gx_pred - gx_true))
        grad_loss_y = K.mean(K.square(gy_pred - gy_true))
        #grad_loss_x = K.sum(K.square(gx_pred - gx_true)*focus)/K.sum(focus+ 1e-8)
        #grad_loss_y = K.sum(K.square(gy_pred - gy_true)*focus)/K.sum(focus+ 1e-8)

        grad_loss = grad_loss_x + grad_loss_y
        print(mse_loss,grad_loss)
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

import matplotlib.pyplot as plt
@tf.keras.utils.register_keras_serializable()
def hovernet_hv_loss_tf(y_true, y_pred,
                        lambda_hv_mse=2.0,
                        lambda_hv_mse_grad=1.0):

    y_true: (B,H,W,3) -> [HVx_true, HVy_true, focus_mask]
    y_pred: (B,H,W,2) -> [HVx_pred, HVy_pred]
    focus_mask: 1 sui nuclei (body|border), 0 sul background

    # separa target e mask
    hv_t   = tf.cast(y_true[..., :2], tf.float32)   # (B,H,W,2)
    focus  = tf.cast(y_true[..., 2:3], tf.float32)  # (B,H,W,1) 0/1
    hv_p   = tf.cast(y_pred, tf.float32)            # (B,H,W,2)
    
    #arr = hv_t._numpy()
    #arr1 = focus._numpy()
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    #axs[0].imshow(arr[0,...,0])
    #axs[1].imshow(arr[0,...,1])
    #axs[2].imshow(arr1[0,...,0])
    #plt.show()
    #
    #arr2 = hv_p._numpy()
    #print(arr.shape,arr1.shape, arr2.shape)
    #for f in range(arr2.shape[0]):
    #    fig, axs = plt.subplots(1,3,figsize=(8,8))
    #    axs[0].imshow(arr2[f,...,0])
    #    axs[1].imshow(arr2[f,...,1])
    #    axs[2].imshow(arr[f,...,1])
    #    plt.show()

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

EPS = tf.constant(1e-6, tf.float32)
def _compute_sobel(y):
    sobel = tf.image.sobel_edges(y)
    sobel_y = sobel[:, :, :, 1, 0] # sobel in y-direction
    sobel_x = sobel[:, :, :, 0, 1] # sobel in x-direction

    return tf.stack([sobel_x, sobel_y], axis=-1)

def mse_gradient_loss(hv_p,
                      hv_t,
                      focus, 
                      lambda_dir=0.3, 
                      lambda_center=0.1, 
                      tau_center=0.05):
    grad_pred = _compute_sobel(hv_p)
    grad_true = _compute_sobel(hv_t)
    #print(f"grad_pred {grad_pred.shape}")
    
    #loss = tf.subtract(grad_true, grad_pred)  # broadcasting sicuro
    loss_grad = grad_true - grad_pred
    #print(f"loss_grad: {tf.square(loss_grad)}")
    
    focus = tf.expand_dims(focus, axis=-1)     #(None, 256,256) -> (None, 256,256,1)
    #fig, axs = plt.subplots(2,3,figsize=(8,8))
    #axs[0,0].imshow(hv_p[0,...,0])
    #axs[0,1].imshow(grad_pred[0,...,0])
    #axs[0,2].imshow(loss_grad[0,...,0])
    #axs[1,0].imshow(hv_p[0,...,1])
    #axs[1,1].imshow(grad_pred[0,...,1])
    #axs[1,2].imshow(loss_grad[0,...,1])
    #plt.show()
    #fig, axs = plt.subplots(2,3,figsize=(8,8))
    #axs[0,0].imshow(hv_t[0,...,0])
    #axs[0,1].imshow(grad_true[0,...,0])
    #axs[0,2].imshow(focus[0], cmap='gray')
    #axs[1,0].imshow(hv_t[0,...,1])
    #axs[1,1].imshow(grad_true[0,...,1])
    #plt.show()
    focus = tf.concat([focus, focus], axis=-1) #(None, 256,256,1) -> (None, 256,256,2)
    #plt.imshow(focus)
    #plt.show()
    grad_loss_y = tf.square(grad_pred[...,1]-grad_true[...,1])
    grad_loss_x = tf.square(grad_pred[...,0]-grad_true[...,0])
    loss = tf.reduce_sum(tf.square(loss_grad)*focus) / (tf.reduce_sum(focus) + 1.0e-8)
    #print(f"\nloss_ {loss}, reduce loss_grad {tf.reduce_sum(loss_grad)}, reduce focus {tf.reduce_sum(focus)}")
    return loss
import keras
def hv_keras_loss(y_true, 
                  y_pred,
                  lambda_dir=0.3, 
                  lambda_center=0.1, 
                  tau_center=0.05, 
                  w_pix=2.0):

    hv_t = y_true[:, :, :, :2]     # (B, H, W, 2)
    focus = y_true[:, :, :, 2]     # (B, H, W)
        
    mse_loss = keras.losses.mean_squared_error(hv_t, y_pred) 
    mse_gradient = mse_gradient_loss(y_pred, hv_t, focus,
                                 lambda_dir=lambda_dir,
                                 lambda_center=lambda_center,
                                 tau_center=tau_center) * w_pix
    

    return mse_loss + mse_gradient

"""