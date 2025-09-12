print(f"Eseguito: {__file__}")

import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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

from scipy.ndimage import label
def build_instance_map_valuewise(mask_6ch):
    instance_map = np.zeros(mask_6ch.shape[:2], dtype=np.uint16)
    current_id = 1
    
    for ch in range(5):
        unique_vals = np.unique(mask_6ch[..., ch])
        
        unique_vals = unique_vals[unique_vals > 0]
        for val in unique_vals:
            mask = (mask_6ch[..., ch] == val)
            if np.any(mask):
                instance_map[mask] = current_id
                current_id += 1
        #fig, axs = plt.subplots(1,2,figsize=(8,8))
        #fig.suptitle(f"{unique_vals}")
        #axs[0].imshow(mask_6ch[..., ch])
        #axs[1].imshow(instance_map)
        #plt.show()
    return instance_map


def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

from scipy.ndimage import measurements
from skimage import morphology as morph
from tensorpack.dataflow.imgaug import ImageAugmentor
from tensorpack.utils.utils import get_rng

class GenInstance(ImageAugmentor):
    def __init__(self, crop_shape=None):
        super(GenInstance, self).__init__()
        self.crop_shape = crop_shape
    
    def reset_state(self):
        self.rng = get_rng(self)

    def _fix_mirror_padding(self, ann):
        """
        Deal with duplicated instances due to mirroring in interpolation
        during shape augmentation (scale, rotation etc.)
        """
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        if 0 in inst_list:
            inst_list.remove(0) # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += current_max_id
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann

class GenInstanceHV(GenInstance):   
    """
        Input annotation must be of original shape.
        
        The map is calculated only for instances within the crop portion
        but based on the original shape in original image.
    
        Perform following operation:
        Obtain the horizontal and vertical distance maps for each
        nuclear instance.
    """

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[...,0] # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)

        # TODO: deal with 1 label warning
        crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

        x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(crop_ann))
        #print("list:", inst_list)
        if 0 in inst_list:
            inst_list.remove(0)
        #if len(inst_list)>0:
        #    inst_list.pop()
        #print("list:",inst_list)
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = bounding_box(inst_map)

            #print("P",inst_map.shape, inst_box)
            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            H, W = fixed_ann.shape
            inst_box[0] = max(inst_box[0] - 2, 0)
            inst_box[1] = min(inst_box[1] + 2, H)
            inst_box[2] = max(inst_box[2] - 2, 0)
            inst_box[3] = min(inst_box[3] + 2, W)

            inst_map = inst_map[inst_box[0]:inst_box[1],
                                inst_box[2]:inst_box[3]]

            if inst_map.shape[0] < 2 or \
                inst_map.shape[1] < 2:
                #print("SKIIP")
                continue
            #print("D",inst_map.shape, inst_box)
            inst_com = list(measurements.center_of_mass(inst_map))
            
            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1]+1)
            inst_y_range = np.arange(1, inst_map.shape[0]+1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]
            
            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype('float32')
            inst_y = inst_y.astype('float32')

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= (-np.amin(inst_x[inst_x < 0]))
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= (-np.amin(inst_y[inst_y < 0]))
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= (np.amax(inst_x[inst_x > 0]))
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= (np.amax(inst_y[inst_y > 0]))

            ####
            x_map_box = x_map[inst_box[0]:inst_box[1],
                              inst_box[2]:inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0]:inst_box[1],
                              inst_box[2]:inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        img = img.astype('float32')

        img = np.dstack([img, x_map, y_map])

        return img
from tqdm import tqdm
def build_instance_map(mask_6ch):
    instance_map = np.zeros(mask_6ch.shape[:2], dtype=np.uint16)
    current_id = 1
    for ch in range(2):
        channel = mask_6ch[..., ch]
        labeled, n = label(channel > 0)
        if n > 0:
            labeled[labeled > 0] += current_id
            instance_map[labeled > 0] = labeled[labeled > 0]
            current_id += n
    return instance_map

def generate_distance_maps(masks_path: str, out_name: str = "distance.npy"):
    # mmap: niente RAM sprecata
    masks = np.load(masks_path, mmap_mode='r')  # shape: (N, H, W, 6)
    N, H, W, _ = masks.shape
    # Prealloca array per salvare le HV maps
    distance_maps = np.zeros((N, H, W, 2), dtype=np.float32)
    blbs = np.zeros((N, H, W, 1), dtype=np.float32)
    #count per il check random
    count = 0
    for i in tqdm(range(N), desc="Generating HV maps"):
        mask = masks[i]  # shape (H, W, 6)
        instance_map = build_instance_map_valuewise(mask)

        # usa GenInstanceHV per calcolare HV map
        instance_input = instance_map[..., np.newaxis]
        gen = GenInstanceHV(crop_shape=(H, W))
        out = gen._augment(instance_input, None)

        distance_maps[i] = out[..., 1:3]  # only HV channels
        #fig,axs = plt.subplots(1,3,figsize=(8,8))
        #axs[0].imshow(instance_input)
        #axs[1].imshow(out[...,1])
        #axs[2].imshow(masks[i,...,0])
        #plt.show()

        #create binary masks with only neoplastic cells
        #blb = mask[...,5]
        blbs[i] = np.where(instance_map!=0, 1.0, 0.0).astype(np.float32)[..., np.newaxis]
        
        #blbs[i] = blb[...,np.newaxis]
        
        #if np.random.rand() < 0.25 and count<30:
        #    count+=1
        #    fig, axs = plt.subplots(1,3,figsize=(8,8))
        #    axs[0].imshow(blbs[i], cmap='gray')
        #    axs[1].imshow(out[...,1])
        #    axs[2].imshow(instance_input)
        #    plt.show()

    # Salva
    binary_path = os.path.join(os.path.dirname(masks_path), "binary_masks.npy")
    np.save(binary_path, blbs)
    print(f"Salvato: {binary_path}")
    out_path = os.path.join(os.path.dirname(masks_path), out_name)
    np.save(out_path, distance_maps)
    print(f"Salvato: {out_path}")

# Esempio d'uso
folds = [os.path.join('data', 'raw_neoplastic', 'Fold 2', 'masks', 'masks.npy'),
         os.path.join('data', 'raw_neoplastic', 'Fold 3', 'masks', 'masks.npy'),
        os.path.join('data', 'raw_neoplastic', 'Fold 1', 'masks', 'masks.npy')] 

#for i in folds:
#    generate_distance_maps(i)
"""
base = os.path.join('data','raw_neoplastic')
fold = [os.path.join(base, 'Fold 3', 'images', 'images.npy'), 
         os.path.join(base, 'Fold 3', 'masks', 'distance.npy'),
         os.path.join(base, 'Fold 3', 'masks', 'masks_neo.npy')]

X = np.load(fold[0])
Y = np.load(fold[1])
Z= np.load(fold[2])
for i in range(Z.shape[0]):
    fig, axs = plt.subplots(1, 4, figsize=(10, 4))

    # ---- RGB (o BGR) normalizzato ----
    rgb = X[i].astype(np.float32)           # se è uint8 puoi saltare il /255
    if rgb.max() > 1.0:
        rgb = rgb / 255.0                   # ora è in [0,1]
    axs[0].imshow(rgb)
    axs[0].set_title("Input")

    # ---- HV distance maps ----
    axs[1].imshow(Y[i][..., 0], cmap="jet") # componente H
    axs[1].set_title("dx")

    axs[2].imshow(Y[i][..., 1], cmap="jet") # componente V
    axs[2].set_title("dy")

    # ---- maschera binaria ----
    axs[3].imshow(Z[i][..., 0], cmap="gray")
    axs[3].set_title("Mask neo")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
"""
"""
distance_maps = np.zeros((Z.shape[0], 256,256, 2), dtype=np.float32)
for i in range(Z.shape[0]):
    instance_map = build_instance_map_valuewise(Z[i])
    print("instance",np.unique(instance_map))
    instance_input = instance_map[..., np.newaxis]
    print(instance_input.shape)
    gen = GenInstanceHV(crop_shape=(256,256))
    out = gen._augment(instance_input, None)
    
    distance_maps[i] = out[..., 1:3]  # [x_map, y_map]
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))
    
    #axs[0].imshow(instance_map, cmap='nipy_spectral')
    #axs[0].set_title("Instance Map")
    #
    #axs[1].imshow(distance_maps[i, ..., 0], cmap='coolwarm')
    #axs[1].set_title("Horizontal (HV_x)")
    #
    #axs[2].imshow(distance_maps[i, ..., 1], cmap='coolwarm')
    #axs[2].set_title("Vertical (HV_y)")

    test = np.where(instance_map != 0, 1.0, 0.0).astype(np.float32)[..., np.newaxis]
    plt.tight_layout()
    axs[0,0].imshow(instance_map)
    axs[0,1].imshow(Z[i,...,0])
    axs[0,2].imshow(Z[i,...,1])
    axs[1,0].imshow(Z[i,...,2])
    axs[1,1].imshow(Z[i,...,3])
    axs[1,2].imshow(test)
    plt.show()
"""