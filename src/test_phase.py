
import os
import numpy as np
import matplotlib.pyplot as plt
from src.losses import bce_dice_loss, hover_loss_fixed
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.color import rgb2hed, hed2rgb, separate_stains,bro_from_rgb

def HE_deconv(img):
    ihc_hed = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    
    return ihc_h, ihc_e, ihc_d 

def load_model(checkpoint_path, weight_seg_head=1.0, weight_hv_head=0.5):
    model = load_model(
    checkpoint_path,
    custom_objects={
        "BCEDiceLoss": bce_dice_loss,   # stesso nome salvato
        "HVLoss": hover_loss_fixed
    },
    compile=False                    # ← evita la ricompilazione automatica
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"seg_head": bce_dice_loss, "hv_head": hover_loss_fixed},
        loss_weights={"seg_head": weight_seg_head, "hv_head": weight_hv_head},
    )
    return model



base = os.path.join('data', 'raw')
folds = [
    (os.path.join(base,'Fold 3', 'images', 'images.npy'),
     os.path.join(base,'Fold 3', 'masks', 'binary_masks.npy'),
     os.path.join(base, 'Fold 3','masks', 'distance.npy'))
]

N = 14
imgs_list, masks_list, dmaps_list = [], [], []
SEED = 42

for i, (img_path, mask_path, dist_path) in enumerate(folds):
    rng = np.random.default_rng(SEED + i)
    imgs = np.load(img_path, mmap_mode='r')
    masks = np.load(mask_path, mmap_mode='r')
    dmaps = np.load(dist_path, mmap_mode='r')
    
    idxs = rng.choice(len(imgs), size=N, replace=False)
    imgs_list.append(imgs[idxs])
    masks_list.append(masks[idxs])
    dmaps_list.append(dmaps[idxs])

X = np.concatenate(imgs_list, axis=0)
Y = np.concatenate(masks_list, axis=0)
HV = np.concatenate(dmaps_list, axis=0)

# Preprocess input images
X = X + 15
X = X / 255
idx = np.random.randint(X.shape[0]-6)
for i in range(idx,idx+5):
    ihc_h, ihc_e, ihc_d = HE_deconv(X[i])
    
    fig, axs = plt.subplots(1,3,figsize=(8,8))
    axs[0].imshow(X[i])
    axs[1].imshow(ihc_h)
    axs[2].imshow(1-X[i,...,2])
    plt.tight_layout()
    plt.show()
    


