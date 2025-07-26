
import os
import numpy as np
import matplotlib.pyplot as plt
from src.losses import bce_dice_loss, hover_loss_fixed
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.color import rgb2hed, hed2rgb, separate_stains,bro_from_rgb
from tqdm import tqdm

def HE_deconv(img):
    ihc_hed = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    
    return ihc_h, ihc_e, ihc_d 

def loadmodel(checkpoint_path, weight_seg_head=1.0, weight_hv_head=0.5):
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

from src.postprocessing import __proc_np_hv
from scipy.stats import pearsonr
def grafo_diffusione(test_set_img, test_set_gt, test_set_hv, preds):
    N_gt = []
    N_pred = []

    seg_preds = preds['seg_head']
    hv_preds  = preds['hv_head']  # (batch, H, W, 2)
    
    for i in tqdm(range(test_set_img.shape[0])):
        mask_gt = test_set_gt[i]
        #costruzione maschera binaria
        seg = seg_preds[i]
        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]
        mask_pred = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)

        #determinazione labels pred
        hv = hv_preds[i]
        pred = np.stack([mask_pred, hv[..., 0], hv[..., 1]], axis=-1)
        # segmentazione con watershed guidata da HV map
        label_pred = __proc_np_hv(pred)

        #determinazione labels GT
        hv_t = test_set_hv[i]
        pred_ = np.stack([mask_gt.squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
        label_gt = __proc_np_hv(pred_)

        N_pred.append(np.max(np.unique(label_pred)))
        N_gt.append(np.max(np.unique(label_gt)))
    N_gt   = np.array(N_gt)
    N_pred = np.array(N_pred)

    # 2. Scatter + linea y=x
    plt.figure(figsize=(4,4))
    plt.scatter(N_gt, N_pred, s=15, alpha=0.7)
    plt.plot([0, N_gt.max()], [0, N_gt.max()], 'k--')
    plt.xlabel('Nuclei GT'); plt.ylabel('Nuclei predetti')

    # 3. Correlazione e MAE
    r, _ = pearsonr(N_gt, N_pred)
    mae  = np.mean(np.abs(N_gt - N_pred))
    plt.title(f'ρ = {r:.3f}, MAE = {mae:.1f}')
    plt.tight_layout(); plt.show()


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
    
    #idxs = rng.choice(len(imgs), size=N, replace=False)
    imgs_list.append(imgs)
    masks_list.append(masks)
    dmaps_list.append(dmaps)

X = np.concatenate(imgs_list, axis=0)
Y = np.concatenate(masks_list, axis=0)
HV = np.concatenate(dmaps_list, axis=0)

# Preprocess input images
X = X + 15
X = X / 255
k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 5))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)  

for i in range(idx,idx+5):
    ihc_h, ihc_e, ihc_d = HE_deconv(X[i])
    
    fig, axs = plt.subplots(1,3,figsize=(8,8))
    axs[0].imshow(X[i])
    axs[1].imshow(ihc_h)
    axs[2].imshow(1-X[i,...,2])
    plt.tight_layout()
    plt.show()

checkpoint_path='models/checkpoints/model_RGB_HV_v3.keras'

model = loadmodel(checkpoint_path=checkpoint_path, weight_hv_head=0.05)
preds = model.predict(X)
grafo_diffusione(X,Y,HV,preds)
    


