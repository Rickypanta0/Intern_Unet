import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from src.losses import  hover_loss_fixed

# Load data
base = os.path.join('data', 'raw', 'val')
folds = [
    (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'distance.npy'))
]

N = 13
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
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
mean = X_gray.mean(axis=(0, 1, 2), keepdims=True)
Xc = np.clip(mean + 1.4 * (X_gray - mean), 0.0, 1.0)

# Split
X_train, _, Y_train, _, HV_train, _ = train_test_split(
    Xc, Y, HV, test_size=0.1, random_state=SEED
)
X_train_rgb = np.repeat(X_train, 3, axis=-1)
from src.losses import bce_dice_loss
# Load model
checkpoint_path = 'models/checkpoints/model_grayscale_HV.keras'

model = load_model(
    checkpoint_path,
    custom_objects={
        'bce_dice_loss': bce_dice_loss,
        'hover_loss_fixed': hover_loss_fixed
    }
)

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

preds = model.predict(X_train_rgb)
seg_preds = preds['seg_head']
hv_preds  = preds['hv_head']  # (batch, H, W, 2)
def __proc_np_hv(pred):
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

    boundary_mask = overall > 0.5  # es. 0.4
    kernel = np.ones((3,3),np.uint8)
    boundary_mask = boundary_mask.astype(np.uint8)
    boundary_mask_ = cv2.morphologyEx(boundary_mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    from skimage.measure import label
    markers = label(1-boundary_mask_)
    from skimage.segmentation import watershed
    blb_ = 1-blb
    instance_map = watershed(overall, markers, mask=blb_)
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    #axs[0].imshow(overall)
    #axs[1].imshow(markers)
    #axs[1].set_title('Markers')
    #b_ = -boundary_mask
    #axs[2].imshow(boundary_mask_)
    #axs[2].set_title('BM no rumore')
    #plt.show()
    """
    overall = overall - (1 - blb)

    overall[overall < 0] = 0

    dist = (1.0 - overall) * (blb)

    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    plt.imshow(dist)
    plt.show()
    overall = np.array(overall >= 0.55, dtype=np.int32)
    
    marker = blb - overall
    marker[marker < 0] = 0
    fig, axs = plt.subplots(1,3,figsize=(8,8))
    axs[0].imshow(marker, cmap='nipy_spectral')
    axs[0].set_title('markers')
    axs[1].imshow(blb)
    axs[1].set_title('Binary')
    axs[2].imshow(overall)
    axs[2].set_title('overall')
    plt.show()
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    plt.imshow(marker)
    plt.show()
    proced_pred = watershed(dist, markers=marker, mask=blb)
    """
    
    return instance_map
# Watershed + HV map visualization and segmentation
for i in range(X_train.shape[0]):
    img = X_train[i, ..., 0]
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

    # visualizzazione
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs[0,0].imshow(img,cmap='gray')
    axs[0,0].set_title('Input Grayscale')
    axs[0,1].imshow(prob_nucleus,cmap='gray')
    axs[0,1].set_title('Binary pred')
    axs[1,0].imshow(label_map, cmap='nipy_spectral', vmin=0, vmax=label_map.max())
    axs[1,0].set_title("Segmentazione HV-Watershed")
    axs[1,1].imshow(hv_t[...,0])
    axs[1,1].set_title("Segmentazione HV-Watershed")
    plt.tight_layout()
    plt.show()

