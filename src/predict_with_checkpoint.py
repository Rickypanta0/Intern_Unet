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
X = X + 15
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
#X_processed = np.stack([extract_hematoxylin_rgb(img / 255.0) for img in X], axis=0) 
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
X_train, _, Y_train, _, HV_train, _ = train_test_split(
    X_gray, Y, HV, test_size=0.1, random_state=SEED
)
from matplotlib.cm import get_cmap
#X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_train_rgb = X_train
# Load model
from src.losses import bce_dice_loss,hover_loss_fixed
checkpoint_path='models/checkpoints/neo/model_BB_RGB.keras'

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
    loss_weights={"seg_head": 1.0, "hv_head": 1.0},
)
preds = model.predict(X_train_rgb)
seg_preds = preds['seg_head']
hv_preds  = preds['hv_head']  # (batch, H, W, 2)
#VISUAL CHECK

for i in range(5):
    isolated_count_blb, contour_img_blb, isolated_count_hv, contour_img_gt = nucle_counting(X_train, HV_train, Y_train, preds,i)
 
    seg = seg_preds[i]
    hv = hv_preds[i]
    hv_t = HV_train[i]
    body_prob = seg[..., 0]
    border_prob = seg[..., 2]
    bg_prob = seg[..., 1]
    # mappa di probabilità nuclei
    prob_nucleus = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)
    
    print_results(X_train_rgb[i], prob_nucleus,contour_img_gt,prob_nucleus, hv_t[...,1],contour_img_blb,isolated_count_blb, isolated_count_hv)

#CALC TOP TRESHOLD

floats = np.arange(0.30, 0.61, 0.01)
floats = np.round(floats, 2)  # per evitare problemi di arrotondamento
errors = []
for thld in tqdm(floats):
    total_gt = 0
    total_pred = 0
    for i in range(X_train.shape[0]):
        img = X_train[i]
        seg = seg_preds[i]
        hv = hv_preds[i]
        hv_t = HV_train[i]

        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]

        # mappa di probabilità nuclei
        prob_nucleus = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)
        pred = np.stack([prob_nucleus, hv[..., 0], hv[..., 1]], axis=-1)

        # segmentazione con watershed guidata da HV map
        label_map = __proc_np_hv(pred,trhld=thld)

        #print(Y_train[i].shape, hv_t[..., 0].shape)
        pred_GT = np.stack([Y_train[i].squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
        label_map_GT = __proc_np_hv(pred_GT, GT=True)
        _, isolated_count_pred, _ = count_blob(label_map, prob_nucleus)
        _, isolated_count_gt, _ = count_blob(label_map_GT, Y_train[i], GT=True)
        
        total_gt += isolated_count_gt
        total_pred += isolated_count_pred

    abs_error = abs(total_gt - total_pred)
    rel_error = abs_error / total_gt if total_gt > 0 else float('inf')
    errors.append((thld, abs_error, rel_error))
    # Ordina per errore relativo
errors.sort(key=lambda x: x[2])
best_threshold = errors[0][0]
print(f"Miglior treshold: {best_threshold}")

#grafico errore

plt.plot(floats, [e[2] for e in errors])
plt.xlabel("Treshold")
plt.ylabel("Relative Error")
plt.title("Treshold optimization")
plt.grid(True)
plt.show()


    


