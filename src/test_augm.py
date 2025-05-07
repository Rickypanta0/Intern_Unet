from .train import train
from .data_loader import load_folds
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from .predict import predict_masks
from .utils.visualization import show_threshold_pairs_test
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from src.model import get_model_paper
from src.callbacks import get_callbacks
import math 
from tensorflow import keras 

def augm(seed: tuple[int, int],
         img: np.ndarray,
         mask: np.ndarray,
         zoom_size: int,
         IMG_SIZE: int) -> tuple[np.ndarray, np.ndarray]:

    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    padded = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 6, IMG_SIZE + 6)
    seed_ = tf.random.experimental.stateless_split(seed, num=2)
    seed = seed_[0]
    crop = tf.image.stateless_random_crop(padded, [zoom_size, zoom_size, 3], seed=seed)

    new_size = int(IMG_SIZE * (zoom_size / IMG_SIZE))
    pad_h = IMG_SIZE - new_size
    pad_w = IMG_SIZE - new_size
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    zoomed = tf.image.resize(crop, [new_size, new_size])
    zoomed_padded = tf.pad(
        zoomed,
        [[pad_top, pad_bottom],
         [pad_left, pad_right],
         [0, 0]],
        constant_values=0.0
    )

    mask = tf.expand_dims(mask, axis=-1)
    padded_mask = tf.image.resize_with_crop_or_pad(mask, IMG_SIZE + 6, IMG_SIZE + 6)
    crop_mask = tf.image.stateless_random_crop(padded_mask, [zoom_size, zoom_size, 1], seed=seed)

    zoomed_mask = tf.image.resize(crop_mask, [new_size, new_size])
    zoomed_padded_Y = tf.pad(
        zoomed_mask,
        [[pad_top, pad_bottom],
         [pad_left, pad_right],
         [0, 0]],
        constant_values=1
    )

    xp = zoomed_padded.numpy().astype(np.float32)
    yp = zoomed_padded_Y.numpy().astype(np.uint8)
    return xp, yp

def _make_target_3ch(m2: np.ndarray) -> np.ndarray:
    # 1) m2 deve essere (H,W), non (H,W,1)
    assert m2.ndim == 2, f"mi aspetto (H,W), ho {m2.shape}"

    # 2) erosion su 2D col struct 2D
    struct = np.ones((3, 3), dtype=bool)
    body = binary_erosion(m2.astype(bool), structure=struct).astype(np.uint8)

    # 3) border e background (su m2 e body 2D)
    border     = (m2.astype(np.uint8) - body).clip(0, 1)
    background = (1 - m2).astype(np.uint8)

    # 4) stack dei tre canali, ora sì 3-D (H,W,3)
    return np.stack([body, background, border], axis=-1)


def __data_generation(X, Y, list_temp, patch_size=None):
    n_aug = 7
    H, W, C = X.shape[1:] if patch_size is None else (patch_size,) * 2 + (X.shape[-1],)
    Xb = np.empty((len(list_temp) * n_aug, H, W, C), dtype=np.float32)
    Yb = np.empty((len(list_temp) * n_aug, H, W, 3), dtype=np.uint8)
    labels = []
    k = 0
    for idx in list_temp:
        img = X[idx]/255
        raw = Y[idx]

        msk = np.squeeze(raw, axis=-1) if raw.ndim == 3 and raw.shape[-1] == 1 else raw

        Xb[k] = img.astype(np.float32)
        Yb[k] = _make_target_3ch(msk)
        labels.append(f"{idx} - original")
        k += 1
        """
        for j in range(4):
            seed = (idx, j)
            xz, yz = augm(seed, img, msk, zoom_size=180, IMG_SIZE=256)
            Xb[k] = xz
            Yb[k] = _make_target_3ch(yz[..., 0])
            labels.append(f"{idx} - zoom {j}")
            k += 1
        """
        img_lr = np.fliplr(img)
        mask_lr = np.fliplr(msk)
        Xb[k] = img_lr.astype(np.float32)
        Yb[k] = _make_target_3ch(mask_lr)
        labels.append(f"{idx} - flip LR")
        k += 1

        img_ud = np.flipud(img)
        mask_ud = np.flipud(msk)
        Xb[k] = img_ud.astype(np.float32)
        Yb[k] = _make_target_3ch(mask_ud)
        labels.append(f"{idx} - flip UD")
        k += 1

    return Xb, Yb, labels


def show_batch(Xb, Yb, labels, n=7):
    plt.figure(figsize=(15, 4 * n))
    for i in range(n):
        plt.subplot(n, 2, 2*i + 1)
        plt.imshow(Xb[i])
        plt.title(f"Image - {labels[i]}")
        plt.axis('off')

        plt.subplot(n, 2, 2*i + 2)
        plt.imshow(Yb[i][..., 2], cmap='gray')  # mostra solo il canale "body"
        plt.title(f"Mask - {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

base = os.path.join('data','raw')
folds = [os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'), 
         os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy')]

X = np.load(folds[0])
Y = np.load(folds[1])

IMG_SIZE = 256
Xb, Yb, labels = __data_generation(X, Y, list_temp=[0])
show_batch(Xb, Yb,labels, n=len(labels))