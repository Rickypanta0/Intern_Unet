import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
from skimage.color import rgb2hed, hed2rgb
from src.postprocessing import nucle_counting, __proc_np_hv,count_blob
from matplotlib.cm import get_cmap

def print_results(img, blb, countour_GT, labels, hv_map, countour_blb, isolated_count_blb, isolated_count_gt):
    fig, axs = plt.subplots(2,3,figsize=(8,8))
    fig.suptitle(f"Binart nuceli count: {isolated_count_blb}, GT nuclei count: {isolated_count_gt}", fontsize=14)
    axs[0,0].imshow(img)
    axs[0,0].set_title('Input Grayscale')
    axs[0,1].imshow(blb,cmap='gray')
    axs[0,1].set_title('Binary pred')
    axs[0,2].imshow(countour_GT)
    axs[0,2].set_title('Count GT')
    axs[1,0].imshow(labels, cmap='nipy_spectral')
    axs[1,0].set_title("Segmentazione HV-Watershed")
    axs[1,1].imshow(hv_map)
    axs[1,1].set_title("Distance map X")
    axs[1,2].imshow(countour_blb)
    axs[1,2].set_title("Count Binary Pred")
    plt.tight_layout()
    plt.show()
# Load data
base = os.path.join('data', 'raw_neoplastic')
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
X = X + 10
X_gray = X / 255
#X_gray = np.dot(X_gray[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
#HESOINE EXTRACTION
            
def extract_hematoxylin_rgb(img):
    ihc_hed = rgb2hed(img)
    null = np.zeros_like(ihc_hed[:, :, 0])
    h_rgb = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    
    mean = h_rgb.mean(axis=(0, 1), keepdims=True)
    h_rgb_norm = np.clip(mean + 1.4 * (h_rgb - mean), 0.0, 1.0)
    
    return h_rgb_norm.astype(np.float32)

# Applica al batch
X_processed = np.stack([extract_hematoxylin_rgb(img / 255.0) for img in X], axis=0) 
#X_ = []
#for f in X_gray:
#    blu = f[...,2]
#    blu_enhanced = np.clip(blu + 0.05, 0, 1)
#    cmap = get_cmap('Blues')
#    img_rgb = cmap(blu_enhanced)[:, :, :3] 
#
#    X_.append(img_rgb)
#X_ = np.stack(X_,axis=0)
#X_ = np.repeat(X_gray, 3, axis=-1).astype(np.float32)

# Split
X_val, _, Y_val, _, HV_val, _ = train_test_split(
    X_gray, Y, HV, test_size=0.1, random_state=SEED
)
from matplotlib.cm import get_cmap
#X_val_rgb = np.repeat(X_val, 3, axis=-1)
X_val_rgb = X_val
# Load model
from src.losses import bce_dice_loss,hover_loss_fixed
checkpoint_path='models/checkpoints/neo/model_RGB.keras'

model = load_model(
    checkpoint_path,
    custom_objects={
        "BCEDiceLoss": bce_dice_loss,   # stesso nome salvato
        "HVLoss": hover_loss_fixed
    },
    compile=False                    # ← evita la ricompilazione automatica
)

# --------------------------------------------
# 3) ricompila con le tue loss/optimizer
# --------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={"seg_head": bce_dice_loss, "hv_head": hover_loss_fixed},
    loss_weights={"seg_head": 1.0, "hv_head": 1.5},
)
preds = model.predict(X_val_rgb)
seg_val_preds = preds['seg_head']
hv_val_preds  = preds['hv_head']  # (batch, H, W, 2)
#VISUAL CHECK

#for i in range(5):
#    isolated_count_blb, contour_img_blb, isolated_count_hv, contour_img_gt = nucle_counting(X_val, HV_val, Y_val, preds,i)
# 
#    seg = seg_preds[i]
#    hv = hv_preds[i]
#    hv_t = HV_val[i]
#    body_prob = seg[..., 0]
#    border_prob = seg[..., 2]
#    bg_prob = seg[..., 1]
#    # mappa di probabilità nuclei
#    prob_nucleus = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)
#    
#    print_results(X_val_rgb[i], prob_nucleus,contour_img_gt,prob_nucleus, hv_t[...,1],contour_img_blb,isolated_count_blb, isolated_count_hv)
#
#CALC TOP TRESHOLD

def tune_params(X_val, Y_val, HV_val, seg_val, hv_val,
                t_fg_grid, min_area_grid):
    best = (1e9, 1e9)  # (MAE, sMAPE)
    best_params = None
    for t_fg in tqdm(t_fg_grid):
        for min_area in min_area_grid:
            abs_err = []
            smape   = []
            for i in range(len(X_val)):
                P  = seg_val[i]
                prob_pred = ((P[...,0]+P[...,2]) > P[...,1]).astype(np.float32)
                pred_stack = np.stack([prob_pred, hv_val[i, ..., 0], hv_val[i, ..., 1]], axis=-1)
                Lp = __proc_np_hv(pred_stack, GT=False, trhld=t_fg, min_area=min_area)
                n_pred = int(Lp.max())
                prob_gt = Y_val[i]
                pred_gt = np.stack([prob_gt.squeeze(), HV_val[i,...,0], HV_val[i,...,1]], axis=-1)
                Lg = __proc_np_hv(pred_gt, GT=True, trhld=0.45, min_area=min_area)
                n_gt = int(Lg.max())
                abs_err.append(abs(n_pred - n_gt))
                smape.append(2*abs(n_pred-n_gt)/(n_pred+n_gt+1e-6))
            mae = float(np.mean(abs_err))
            sm = float(np.mean(smape))
            if (mae, sm) < best:
                best = (mae, sm)
                best_params = (t_fg, min_area, mae, sm)
    return best_params

# Esempio d'uso (usa il VALIDATION set, non il train):
t_fg_grid    = np.round(np.arange(0.35, 0.61, 0.02), 2)
t_seed_grid  = [0.55, 0.60, 0.65]
min_area_grid= [10, 20, 30]

best = tune_params(X_val, Y_val, HV_val, seg_val_preds, hv_val_preds,
                   t_fg_grid, min_area_grid)
print(f"Migliori: t_fg={best[0]}, t_seed={best[1]}, min_area={best[2]} | MAE={best[3]:.2f}, sMAPE={best[4]:.3f}")


    


