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

# Predict
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]
from scipy import ndimage
def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def __proc_np_hv(pred, GT=False, trhld=0.56):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    
    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    
    overall[blb>0] = 3
    #plt.imshow(overall)
    #plt.show()
    treshold = 0.45 if GT else trhld
    boundary_mask = overall > treshold  # es. 0.4
    kernel = np.ones((3,3),np.uint8)
    boundary_mask = boundary_mask.astype(np.uint8)
    boundary_mask_ = cv2.morphologyEx(boundary_mask,cv2.MORPH_OPEN,kernel, iterations = 2) if not GT else boundary_mask
    from skimage.measure import label
    markers = label(1-boundary_mask_)
    #fig, axs = plt.subplots(1,3, figsize = (8,8))
    #axs[0].imshow(boundary_mask_)
    #axs[1].imshow(markers)
    #axs[2].imshow(overall)
    #plt.show()
    #print("label markers: ", list(np.unique(markers)))
    from skimage.segmentation import watershed
    blb_ = 1-blb
    #plt.imshow(markers)
    #plt.show()
    instance_map = watershed(overall, markers, mask=blb_)

    return instance_map

def count_blob(labels, blb, GT=False):
    # Se blb ha valori float o non è in formato uint8, lo normalizzo per la visualizzazione
    if blb.dtype != np.uint8:
        blb_norm = cv2.normalize(blb, None, 0, 255, cv2.NORM_MINMAX)
        blb_uint8 = blb_norm.astype(np.uint8)
    else:
        blb_uint8 = blb
    if 255 not in np.unique(blb_uint8):
        blb_uint8 = blb_uint8 * 255
    
    # Converti in BGR per disegnare i contorni
    contour_img = cv2.cvtColor(blb_uint8, cv2.COLOR_GRAY2BGR)
    #fig, axs = plt.subplots(1,2,figsize=(8,8))
    #axs[0].imshow(contour_img)
    #axs[1].imshow(labels)
    #plt.show()
    #print(np.unique(labels))
    isolated_count = 0
    cluster_count = 0
    #print(np.unique(labels))
    for label in np.unique(labels):
        if label <= 0:
            continue

        single_mask = (labels == label).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        treshold = 1 if GT else 11
        
        def debug_visual(label, single_mask, labels, blb_uint8, contour_img):
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            axs[0].imshow(single_mask, cmap='gray')
            axs[0].set_title(f"Mask for label {label}")

            axs[1].imshow(labels, cmap='nipy_spectral')
            axs[1].set_title("Labels")

            axs[2].imshow(blb_uint8, cmap='gray')
            axs[2].set_title("blb_uint8")

            axs[3].imshow(contour_img)
            axs[3].set_title("Contours")

            for ax in axs:
                ax.axis('off')

            plt.tight_layout()
            plt.show()
        
        if not cnts or all(len(c) < treshold for c in cnts):
            #print(f"SKIPP - label {label}, {cnts}")
            #debug_visual(label, single_mask, labels, blb_uint8, contour_img)
            continue

        cntr = cnts[0]
        area = cv2.contourArea(cntr)
        convex_hull = cv2.convexHull(cntr)
        convex_hull_area = cv2.contourArea(convex_hull)

        if convex_hull_area == 0:
            continue  # Evita divisione per zero

        ratio = area / convex_hull_area
        #if ratio < 0.91:
        #    cv2.drawContours(contour_img, [cntr], 0, (0, 0, 255), 2)
        #    cluster_count += 1
        #else:
        cv2.drawContours(contour_img, [cntr], 0, (0, 255, 0), 2)
        isolated_count += 1

    return (contour_img, isolated_count, cluster_count)

def nucle_counting(X_train, HV_train, Y_train, preds, i):
    seg_preds = preds['seg_head']
    hv_preds  = preds['hv_head']  # (batch, H, W, 2)
    
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
    label_map = __proc_np_hv(pred)
    print(Y_train[i].shape, hv_t[..., 0].shape)
    pred_GT = np.stack([Y_train[i].squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
    label_map_GT = __proc_np_hv(pred_GT, GT=True)
    contour_img_blb, isolated_count_blb, _ = count_blob(label_map, prob_nucleus)
    contour_img_gt, isolated_count_hv, _ = count_blob(label_map_GT, Y_train[i], GT=True)
    #stampa
    print(f"Blb: {isolated_count_blb}, HV: {isolated_count_hv}\n")

    return isolated_count_blb, contour_img_blb, isolated_count_hv, contour_img_gt

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
X_gray = X / 255
#X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
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
mean = X_gray.mean(axis=(0, 1, 2), keepdims=True)
Xc = np.clip(mean + 1.4 * (X_gray - mean), 0.0, 1.0)

# Split
X_train, _, Y_train, _, HV_train, _ = train_test_split(
    X_processed, Y, HV, test_size=0.1, random_state=SEED
)
#X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_train_rgb = X_train
# Load model
from src.losses import bce_dice_loss,hover_loss_fixed
checkpoint_path='models/checkpoints/model_HE_HV.keras'

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
    loss_weights={"seg_head": 1.0, "hv_head": 0.05},
)
preds = model.predict(X_train_rgb)
seg_preds = preds['seg_head']
hv_preds  = preds['hv_head']  # (batch, H, W, 2)
#VISUAL CHECK
"""
for i in range(X_train.shape[0]):
    isolated_count_blb, contour_img_blb, isolated_count_hv, contour_img_gt = nucle_counting(X_train, HV_train, Y_train, preds,i)
 
    seg = seg_preds[i]
    hv = hv_preds[i]
    hv_t = HV_train[i]
    body_prob = seg[..., 0]
    border_prob = seg[..., 2]
    bg_prob = seg[..., 1]
    # mappa di probabilità nuclei
    prob_nucleus = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)
    print_results(X_train_rgb[i],prob_nucleus,contour_img_gt,prob_nucleus, hv_t[...,1],contour_img_blb,isolated_count_blb, isolated_count_hv)
"""
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


    


