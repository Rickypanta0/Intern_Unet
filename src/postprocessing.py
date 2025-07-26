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

def __proc_np_hv(pred, GT=False, trhld=0.55):
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