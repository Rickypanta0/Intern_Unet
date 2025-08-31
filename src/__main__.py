
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    # opzionale ma consigliato
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"            # meno rumore nei log
os.environ["KERAS_BACKEND"] = "torch"
import os
os.environ["KERAS_BACKEND"] = "torch"  # <<— fondamentale: PRIMA di importare keras

import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import math
from matplotlib.cm import get_cmap
import cv2
from skimage.exposure import rescale_intensity

import torch
import keras
from keras import ops
from keras.utils import register_keras_serializable

from src.model import get_model_paper  # deve usare keras.layers, non tf.keras
from skimage.color import rgb2hed, hed2rgb
from src.hv_GT_generator import build_instance_map_valuewise, GenInstanceHV
# ------------------------------
# Data generator (solo NumPy)
# ------------------------------
BACKBONE = 'resnet34'

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folds, list_IDs, batch_size=1, patch_size=None,
                 shuffle=True, augment=True, skip_empty=True):
        self.folds = folds  # lista di triple (img_path, mask_path, hv_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.augment = augment
        self.skip_empty = skip_empty
        self.preprocess_input = BACKBONE
        self.list_IDs = np.asarray(list_IDs)  # global indices
        self.ratio = 2

        self.sample_map = []
        for fold_idx, (img_path, _, _) in enumerate(folds):
            n_samples = np.load(img_path, mmap_mode='r').shape[0]
            for i in range(n_samples):
                self.sample_map.append((fold_idx, i))

        sample_img = np.load(folds[0][0], mmap_mode='r')[0]
        self.default_shape = sample_img.shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.sample_map[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def _make_target_3ch(self, m2):
        struct = np.ones((3, 3), dtype=bool)

        m2_ = 1 - m2
        body_ = binary_erosion(m2_, structure=struct).astype(np.uint8)
        body = 1 - body_

        border = (m2_ - body_).clip(0, 1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)

        return np.stack([body, background, border], axis=-1)

    def hema_rgb(self, img_rgb01):  # img in [0,1]
        ihc = rgb2hed(np.clip(img_rgb01, 0, 1))
        H = ihc[..., 0]
        hema_only = np.stack([H, np.zeros_like(H), np.zeros_like(H)], axis=-1)
        img_h = hed2rgb(hema_only).astype(np.float32)  # torna in RGB
        return np.clip(img_h, 0, 1)

    def __data_generation(self, list_temp):
        if self.patch_size is None:
            H, W, _ = self.default_shape
        else:
            H = W = self.patch_size
        C = 3

        N = self.batch_size

        X = np.empty((N, H, W, C), dtype=np.float32)
        Y = np.empty((N, H, W, 3), dtype=np.float32)
        HV = np.empty((N, H, W, 3), dtype=np.float32)

        for i, (fold_idx, local_idx) in enumerate(list_temp):
            img_path, mask_path, instance_path = self.folds[fold_idx]

            img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
            img_rgb = (img + 5) / 255.0

            mask = np.load(mask_path, mmap_mode='r')[local_idx]
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)

            #hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)
            instance_M = np.load(instance_path, mmap_mode='r')[local_idx].astype(np.float32)
            instance_map = build_instance_map_valuewise(instance_M)

            # Augmentation (NumPy-only)
            if self.augment:
                if np.random.rand() < 0.5:
                    img_rgb = np.fliplr(img_rgb)
                    mask    = np.fliplr(mask)
                    instance_map      = np.fliplr(instance_map)
                    #hv[..., 0] *= -1  # inverti x
                if np.random.rand() < 0.5:
                    img_rgb = np.flipud(img_rgb)
                    mask    = np.flipud(mask)
                    instance_map      = np.flipud(instance_map)
                    #hv[..., 1] *= -1  # inverti y
                k = np.random.randint(4)
                img_rgb = np.rot90(img_rgb, k, axes=(0, 1))
                mask    = np.rot90(mask,    k, axes=(0, 1))
                instance_map      = np.rot90(instance_map,      k, axes=(0, 1))
                #if   k == 1: hv = np.stack([-hv[...,1],  hv[...,0]], axis=-1)
                #elif k == 2: hv = np.stack([-hv[...,0], -hv[...,1]], axis=-1)
                #elif k == 3: hv = np.stack([ hv[...,1], -hv[...,0]], axis=-1)

            instance_input = instance_map[..., np.newaxis]
            gen = GenInstanceHV(crop_shape=(H, W))
            out = gen._augment(instance_input, None)
            hv = out[..., 1:3]

            X[i]  = img_rgb
            Y[i]  = self._make_target_3ch(mask)
            HV[i] = np.dstack([hv, mask])

        return X, {'seg_head': Y, 'hv_head': HV}


# ------------------------------
# Metriche (keras.ops, backend-agnostiche)
# ------------------------------
@register_keras_serializable()
class CellDice(keras.metrics.Metric):
    """F1 (Dice) su maschera cella = body ∪ border (vs background)."""
    def __init__(self, name="cell_dice", smooth=1e-6, **kw):
        super().__init__(name=name, **kw)
        self.smooth = float(smooth)
        self.intersection = self.add_weight(name="inter", initializer="zeros", dtype="float32")
        self.union        = self.add_weight(name="union", initializer="zeros", dtype="float32")

    def _bin_mask(self, y):
        body   = y[..., 0]
        bg     = y[..., 1]
        border = y[..., 2]
        return ops.cast((body + border) > bg, "float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_b = self._bin_mask(y_true)
        y_pred_b = self._bin_mask(y_pred)
        inter = ops.sum(y_true_b * y_pred_b)
        union = ops.sum(y_true_b) + ops.sum(y_pred_b)
        self.intersection.assign_add(ops.cast(inter, "float32"))
        self.union.assign_add(ops.cast(union, "float32"))

    def result(self):
        s = ops.cast(self.smooth, "float32")
        return (2.0 * self.intersection + s) / (self.union + s)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


# ------------------------------
# Losses (importa la tua HV Torch + BCE/Dice ops)
# ------------------------------
from src.losses import HVLossMonaiTorch, bce_dice_loss_ops
# Assicurati che src/losses.py contenga:
#  - class HVLossMonaiTorch(keras.losses.Loss) -> usa torch/monai per Sobel
#  - def bce_dice_loss_ops(y_true, y_pred, ...) -> usa keras.ops (no tf.*)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = os.path.join('data', 'raw_neoplastic')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks',  'binary_masks.npy'),
         os.path.join(base, 'Fold 1', 'masks',  'masks.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks',  'binary_masks.npy'),
         os.path.join(base, 'Fold 2', 'masks',  'masks.npy'))
    ]

    num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)
    all_IDs = np.arange(num_all)

    # split 90/10
    split = int(0.9 * num_all)
    train_IDs = all_IDs[:split]
    val_IDs   = all_IDs[split:]

    # Generators
    train_gen = DataGenerator(folds, train_IDs, batch_size=8, shuffle=True,  augment=True)
    val_gen   = DataGenerator(folds, val_IDs,   batch_size=8, shuffle=False, augment=False)

    monitor_metric = 'val_seg_head_cell_dice'
    #monitor_metric = 'val_loss'
    

    # Callbacks (versione Keras 3)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.5, patience=3,
        min_lr=1e-4, mode="max", verbose=1
    )
    # TensorBoard è opzionale; se non ti serve, commenta:
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs', 'fit'),
        histogram_freq=1, write_images=False
    )
    checkpoint_path = 'models/checkpoints/neo/model_RGB_mse2_grad1_L.keras'
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True, save_weights_only=False,
        monitor=monitor_metric, mode="max", verbose=1
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor=monitor_metric, mode="max", patience=5, restore_best_weights=True
    )

    # Build model (assicurati che src.model usi keras.layers)
    model = get_model_paper()

    # Optimizer (semplice e compatibile)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)

    # Compile: seg_head con BCE+Dice (ops), hv_head con HV Torch
    model.compile(
        run_eagerly=False,
        optimizer=optimizer,
        loss={
            'seg_head': bce_dice_loss_ops,
            'hv_head' : HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=1.0, ksize=5),
        },
        loss_weights={'seg_head': 1.0, 'hv_head': 0.5},
        metrics={'seg_head': [CellDice()], 'hv_head': []},
    )

    # Debug shapes
    Xb, Yb = train_gen[0]
    
    k = min(4, Xb.shape[0])                                 # quante immagini mostrare
    high = max(1, Xb.shape[0] - (k - 1))                    # ultimo indice di partenza valido + 1
    idx = np.random.randint(0, high)    
    
    print(Xb.shape, Yb['seg_head'].shape, Yb['hv_head'].shape)

    hv_batch = Yb['hv_head']
    #print(f"min: {np.min(hv_batch)}, max {np.max(hv_batch)}, mean {np.mean(hv_batch)}")
 
    for i in range(idx,idx + 4):
            fig, axs = plt.subplots(2, 3, figsize=(6,6))
            image = Xb[i]       # (H, W, 3) – RGB image (non usata da GenInstanceHV)
            mask_3ch = Yb['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
            hv = Yb['hv_head'][i]
            body_mask = mask_3ch[..., 0].astype(np.uint8)  # binaria
            border_mask = mask_3ch[..., 2].astype(np.uint8)
            background = mask_3ch[..., 1].astype(np.uint8)

            C = np.logical_or(body_mask, border_mask).astype(np.uint8)
            axs[0,0].imshow(image)                         # immagine normalizzata [0,1]
            axs[0,1].imshow(body_mask, cmap='gray')  # body
            axs[0,2].imshow(background, cmap='gray')  # background
            axs[1,0].imshow(border_mask, cmap='gray')  # border
            axs[1,1].imshow(hv[...,0])  # body
            axs[1,2].imshow(hv[...,1])  # background

            plt.tight_layout()
            plt.show()

    from monai.transforms import SobelGradients
    
    print(Xb.shape, Yb['seg_head'].shape, Yb['hv_head'].shape)
    class HVSobelDebugCB(keras.callbacks.Callback):
        def __init__(self, val_gen, out_dir="logs/hv_debug", n=2):
            self.val_gen = val_gen
            self.n = n
            self.out_dir = out_dir
            os.makedirs(out_dir, exist_ok=True)
            # MONAI Sobel (una volta sola, fuori dalla loss)
            self.sobel_h = SobelGradients(kernel_size=3, spatial_axes=1)  # ∂x
            self.sobel_v = SobelGradients(kernel_size=3, spatial_axes=0)  # ∂y

        def on_epoch_end(self, epoch, logs=None):
            x, y = self.val_gen[0]
            pred = self.model.predict(x[:self.n], verbose=0)['hv_head']  # (n,H,W,2)
            hv_t = y['hv_head'][:self.n, ...,:2]                         # (n,H,W,2)
            focus = y['hv_head'][:self.n, ..., 2] > 0.5                  # (n,H,W)

            # torch tensors per usare MONAI sobel
            Hp = torch.from_numpy(pred[...,0])
            Vp = torch.from_numpy(pred[...,1])
            Ht = torch.from_numpy(hv_t[...,0])
            Vt = torch.from_numpy(hv_t[...,1])

            gh_p = self.sobel_h(Hp); gh_t = self.sobel_h(Ht)
            gv_p = self.sobel_v(Vp); gv_t = self.sobel_v(Vt)

            for i in range(self.n):
                f = focus[i]
                fig, axs = plt.subplots(2,3, figsize=(8,6))
                axs[0,0].imshow(Ht[i], cmap='coolwarm'); axs[0,0].set_title("H true")
                axs[0,1].imshow(Hp[i], cmap='coolwarm'); axs[0,1].set_title("H pred")
                axs[0,2].imshow((Hp[i]-Ht[i])*f, cmap='coolwarm'); axs[0,2].set_title("H err (nuclei)")

                axs[1,0].imshow(gh_t[i], cmap='gray'); axs[1,0].set_title("∂x H true")
                axs[1,1].imshow(gh_p[i], cmap='gray'); axs[1,1].set_title("∂y V true")
                axs[1,2].imshow(((gh_p[i]-gh_t[i])**2 + (gv_p[i]-gv_t[i])**2)*f, cmap='magma')
                axs[1,2].set_title("grad err (nuclei)")
                for ax in axs.ravel(): ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"epoch{epoch:03d}_sample{i}.png"))
                plt.close(fig)
    # Fit
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[earlystop_cb, checkpoint_cb, reduce_lr_cb,HVSobelDebugCB(val_gen)],
        verbose=1
    )
    from keras.models import load_model

    model = load_model(
        checkpoint_path,
        custom_objects={
            "BCEDiceLoss": bce_dice_loss_ops,   # stesso nome salvato
            "HVLoss": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=1.0, ksize=5)
        },
        compile=False                    # ← evita la ricompilazione automatica
    )

    model.compile(
        optimizer= keras.optimizers.Adam(1e-3),
        loss={"seg_head": bce_dice_loss_ops,
               "hv_head": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=4.0, ksize=5)},
        loss_weights={"seg_head": 1.0, "hv_head": 0.2},
    )
    sobel_h = SobelGradients(kernel_size=3, spatial_axes=1)  # ∂x
    sobel_v = SobelGradients(kernel_size=3, spatial_axes=0)  # ∂y

    # batch piccolo dalla val
    x, y = val_gen[0]
    yp = model.predict(x[:2], verbose=0)['hv_head']         # (n,H,W,2)
    yt = y['hv_head'][:2]                                   # (n,H,W,3)

    Hp = torch.from_numpy(yp[...,0]); Vp = torch.from_numpy(yp[...,1])
    Ht = torch.from_numpy(yt[...,0]); Vt = torch.from_numpy(yt[...,1])
    focus = torch.from_numpy((y['seg_head'][:2][...,0] + y['seg_head'][:2][...,2]) > 0.5)  # body∪border

    gx_t = sobel_h(Ht); gy_t = sobel_v(Vt)
    gx_p = sobel_h(Hp); gy_p = sobel_v(Vp)

    # medie assolute dentro ai nuclei
    m_true = ((gx_t.abs() + gy_t.abs()) * focus).sum() / (focus.sum() + 1e-8)
    m_pred = ((gx_p.abs() + gy_p.abs()) * focus).sum() / (focus.sum() + 1e-8)
    print("⟨|∂H_true|+|∂V_true|⟩_nuclei =", float(m_true))
    print("⟨|∂H_pred|+|∂V_pred|⟩_nuclei =", float(m_pred))
"""
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from src.model import get_model_paper
import math 
from tensorflow import keras 
from matplotlib.cm import get_cmap
#from src.test_augm import build_instance_map_valuewise, GenInstanceHV
import cv2
from skimage.exposure import rescale_intensity
BACKBONE = 'resnet34'
"""
"""
    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         hv_np : np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:

        #img_np: np.ndarray (H,W) o (H,W,1) o (H,W,3), valori [0..1] o [0..255]
        #mask_np: np.ndarray (H,W), valori {0,1}
        
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        # 1) Assicuriamoci di avere (H,W,1) o (H,W,3)
        if img_np.ndim == 2:
            img_np = img_np[...,None]
        # 2) Tensor float32
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)
        C_orig = img_t.shape[-1]
        # 3) Se mono-canale, duplichiamo su 3 canali
        if C_orig == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)
        # adesso img_t.shape == (H,W,3)
        # 4) Pad/center-crop per avere shape statica (IMG_SIZE×IMG_SIZE×3)
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE, IMG_SIZE)

        # 5) Preparo il seed TF
        seed_tf = tf.convert_to_tensor(seed, dtype=tf.int32)

        # 6) Random crop **dal padded**, taglio un quadrato zoom_size×zoom_size×3
        crop = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, 3],
            seed=seed_tf
        )
        # 7) Ridimensiono back a IMG_SIZE×IMG_SIZE
        xz = tf.image.resize(crop, [IMG_SIZE, IMG_SIZE])

        # 8) Stessa cosa per la mask (ha sempre 1 canale)
        mask_t = tf.convert_to_tensor(mask_np[...,None], dtype=tf.float32)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE, IMG_SIZE)
        crop_m = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, 1],
            seed=seed_tf
        )
        yz = tf.image.resize(crop_m, [IMG_SIZE, IMG_SIZE])

        # HV map (H,W,2)
        if hv_np.ndim != 3 or hv_np.shape[-1] != 2:
            raise ValueError(f"hv_np deve avere shape (H,W,2), ma ha {hv_np.shape}")

        # Separazione dei canali x e y
        hv_x = hv_np[..., 0][..., None]  # shape (H,W,1)
        hv_y = hv_np[..., 1][..., None]

        # Tensor conversione
        hv_x_t = tf.convert_to_tensor(hv_x, dtype=tf.float32)
        hv_y_t = tf.convert_to_tensor(hv_y, dtype=tf.float32)

        # Pad + crop + resize per x
        padded_x = tf.image.resize_with_crop_or_pad(hv_x_t, IMG_SIZE, IMG_SIZE)
        crop_x = tf.image.stateless_random_crop(padded_x, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cx = tf.image.resize(crop_x, [IMG_SIZE, IMG_SIZE])

        # Pad + crop + resize per y
        padded_y = tf.image.resize_with_crop_or_pad(hv_y_t, IMG_SIZE, IMG_SIZE)
        crop_y = tf.image.stateless_random_crop(padded_y, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cy = tf.image.resize(crop_y, [IMG_SIZE, IMG_SIZE])

        # 9) Torno a NumPy
        hv_cx_np = hv_cx.numpy().squeeze(-1).astype(np.float32)
        hv_cy_np = hv_cy.numpy().squeeze(-1).astype(np.float32)
        hv_np_f  = np.stack([hv_cx_np, hv_cy_np], axis=-1)

        xz_np = xz.numpy().astype(np.float32)
        yz_np = yz.numpy().astype(np.uint8)

        return xz_np, yz_np, hv_np_f
    """
"""
from skimage.color import rgb2hed, hed2rgb
class DataGenerator(keras.utils.Sequence):
    def __init__(self, folds, list_IDs, batch_size=1, patch_size=None,
                 shuffle=True, augment=True, skip_empty=True):
        self.folds = folds  # lista di triple (img_path, mask_path, hv_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.augment = augment
        self.skip_empty = skip_empty
        self.preprocess_input = BACKBONE
        self.list_IDs = np.asarray(list_IDs)  # global indices
        self.ratio = 2

        self.sample_map = []
        for fold_idx, (img_path, _, _) in enumerate(folds):
            n_samples = np.load(img_path, mmap_mode='r').shape[0]
            for i in range(n_samples):
                self.sample_map.append((fold_idx, i))

        sample_img = np.load(folds[0][0], mmap_mode='r')[0]
        self.default_shape = sample_img.shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.sample_map[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def _make_target_3ch(self, m2):
        struct = np.ones((3, 3), dtype=bool)

        m2_ = 1-m2
        body_ = binary_erosion(m2_, structure=struct).astype(np.uint8)
        
        body = 1-body_

        border = (m2_ - body_).clip(0, 1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)

        return np.stack([body,background, border], axis=-1)
    """
"""
    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         hv_np : np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:
        
        #img_np: np.ndarray (H,W) o (H,W,1) o (H,W,3), valori [0..1] o [0..255]
        #mask_np: np.ndarray (H,W), valori {0,1}
        
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        # 1) Assicuriamoci di avere (H,W,1) o (H,W,3)
        if img_np.ndim == 2:
            img_np = img_np[...,None]
        # 2) Tensor float32
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)
        C_orig = img_t.shape[-1]
        # 3) Se mono-canale, duplichiamo su 3 canali
        if C_orig == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)
        # adesso img_t.shape == (H,W,3)
        # 4) Pad/center-crop per avere shape statica (IMG_SIZE×IMG_SIZE×3)
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE, IMG_SIZE)

        # 5) Preparo il seed TF
        seed_tf = seed

        # 6) Random crop **dal padded**, taglio un quadrato zoom_size×zoom_size×3
        crop = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, 3],
            seed=seed_tf
        )
        # 7) Ridimensiono back a IMG_SIZE×IMG_SIZE
        xz = tf.image.resize(crop, [IMG_SIZE, IMG_SIZE])

        # 8) Stessa cosa per la mask (ha sempre 1 canale)
        mask_t = tf.convert_to_tensor(mask_np[...,None], dtype=tf.float32)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE, IMG_SIZE)
        crop_m = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, 1],
            seed=seed_tf
        )
        yz = tf.image.resize(crop_m, [IMG_SIZE, IMG_SIZE])

        # HV map (H,W,2)
        if hv_np.ndim != 3 or hv_np.shape[-1] != 2:
            raise ValueError(f"hv_np deve avere shape (H,W,2), ma ha {hv_np.shape}")

        # Separazione dei canali x e y
        hv_x = hv_np[..., 0][..., None]  # shape (H,W,1)
        hv_y = hv_np[..., 1][..., None]

        # Tensor conversione
        hv_x_t = tf.convert_to_tensor(hv_x, dtype=tf.float32)
        hv_y_t = tf.convert_to_tensor(hv_y, dtype=tf.float32)

        # Pad + crop + resize per x
        padded_x = tf.image.resize_with_crop_or_pad(hv_x_t, IMG_SIZE, IMG_SIZE)
        crop_x = tf.image.stateless_random_crop(padded_x, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cx = tf.image.resize(crop_x, [IMG_SIZE, IMG_SIZE])

        # Pad + crop + resize per y
        padded_y = tf.image.resize_with_crop_or_pad(hv_y_t, IMG_SIZE, IMG_SIZE)
        crop_y = tf.image.stateless_random_crop(padded_y, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cy = tf.image.resize(crop_y, [IMG_SIZE, IMG_SIZE])

        # 9) Torno a NumPy
        hv_cx_np = hv_cx.numpy().squeeze(-1).astype(np.float32)
        hv_cy_np = hv_cy.numpy().squeeze(-1).astype(np.float32)
        hv_np_f  = np.stack([hv_cx_np, hv_cy_np], axis=-1)

        xz_np = xz.numpy().astype(np.float32)
        yz_np = yz.numpy().astype(np.uint8)
        print("xz_np:", xz_np.shape)
        print("yz_np:", yz_np.shape)
        print("hv_np_f:", hv_np_f.shape)
        yz_np = np.squeeze(yz_np)
        if yz_np.ndim != 2:
            raise ValueError(f"La maschera deve essere 2D dopo lo squeeze, ma ha shape {yz_np.shape}")
        
        return xz_np, yz_np, hv_np_f
    """
"""
    def hema_rgb(self, img_rgb01):  # img in [0,1]
        ihc = rgb2hed(np.clip(img_rgb01, 0, 1))
        H = ihc[..., 0]
        hema_only = np.stack([H, np.zeros_like(H), np.zeros_like(H)], axis=-1)
        img_h = hed2rgb(hema_only).astype(np.float32)  # torna in RGB
        return np.clip(img_h, 0, 1) 

    def __data_generation(self, list_temp):
        if self.patch_size is None:
            H, W, _ = self.default_shape
        else:
            H = W = self.patch_size
        C = 3

        N = self.batch_size

        X = np.empty((N, H, W, C), dtype=np.float32)
        Y = np.empty((N, H, W, 3), dtype=np.float32)
        HV = np.empty((N, H, W, 3), dtype=np.float32)

        i=0
        #for fold_idx, local_idx in list_temp:
        #    img_path, mask_path, hv_path = self.folds[fold_idx]

        #    img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
        #    img = (img + 5) / 255.0
        #    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        #    img_gray = img_gray[..., np.newaxis]
        #    img_rgb = np.repeat(img_gray, 3, axis=-1)

        #    mask = np.load(mask_path, mmap_mode='r')[local_idx]
        #    if mask.ndim == 3 and mask.shape[-1] == 1:
        #        mask = np.squeeze(mask, axis=-1)

        #    hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)

        for i, (fold_idx, local_idx) in enumerate(list_temp):
            img_path, mask_path, hv_path = self.folds[fold_idx]

            img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
            img_rgb = (img + 5) / 255.0
            
            #HESOINE EXTRACTION
            
            #ihc_hed = rgb2hed(img_rgb)
#
            ## Create an RGB image for each of the stains
            #null = np.zeros_like(ihc_hed[:, :, 0])
            #img_rgb = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            #img_rgb = self.hema_rgb(img_rgb)
            #BLU CHANNEL
            
            #blu = img_rgb[...,2]
            #blu_enhanced = np.clip(blu + 0.05, 0, 1)
            #cmap = get_cmap('Blues')
            #img_rgb = cmap(blu_enhanced)[:, :, :3] 

            #GRAY SCALE
            #img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
            #img_gray = img_gray[..., np.newaxis]
            #img_rgb = np.repeat(img_gray, 3, axis=-1)
            #
            mask = np.load(mask_path, mmap_mode='r')[local_idx]
        
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)

            hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)

            # Augmentation
            if self.augment:
                if np.random.rand() < 0.5:
                    img_rgb = np.fliplr(img_rgb)
                    mask   = np.fliplr(mask)
                    hv      = np.fliplr(hv)
                    hv[...,0] *= -1  # inverti x
                if np.random.rand() < 0.5:
                    img_rgb = np.flipud(img_rgb)
                    mask   = np.flipud(mask)
                    hv      = np.flipud(hv)
                    hv[...,1] *= -1  # inverti y
                k = np.random.randint(4) 
                #print(mask.shape)
                img_rgb = np.rot90(img_rgb, k, axes=(0, 1))
                mask   = np.rot90(mask,   k, axes=(0, 1))
                hv      = np.rot90(hv,      k, axes=(0, 1))

                if   k == 1: hv = np.stack([-hv[...,1],  hv[...,0]], axis=-1)
                elif k == 2: hv = np.stack([-hv[...,0], -hv[...,1]], axis=-1)
                elif k == 3: hv = np.stack([ hv[...,1], -hv[...,0]], axis=-1)

            X[i] = img_rgb#preprocess_input(img_rgb*255)
            Y[i] = self._make_target_3ch(mask)
            HV[i] = np.dstack([hv, mask])   

        return X, {'seg_head': Y, 'hv_head': HV}
    
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CellDice(tf.keras.metrics.Metric):
    #F1 (o Dice) su maschere cella = body ∪ border (canali 0 e 2).
    def __init__(self, name="cell_dice", smooth=1e-6, **kw):
        super().__init__(name=name, **kw)
        self.smooth = smooth
        self.intersection = self.add_weight(name="inter", initializer="zeros")
        self.union        = self.add_weight(name="union", initializer="zeros")

    def _bin_mask(self, y):
        body, bg, border = tf.unstack(y, axis=-1)
        return tf.cast(body + border > bg, tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_b = self._bin_mask(y_true)           # bool
        y_pred_b = self._bin_mask(y_pred)
        inter = tf.reduce_sum(y_true_b * y_pred_b)
        union = tf.reduce_sum(y_true_b) + tf.reduce_sum(y_pred_b)
        self.intersection.assign_add(inter)
        self.union.assign_add(union)
        
    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # attiva memory growth **prima** di qualunque allocazione
        tf.config.experimental.set_memory_growth(gpus[0], True)
    SEED = 42
    np.random.seed = SEED
    base = os.path.join( 'data','raw_neoplastic')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'binary_masks.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'distance.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'binary_masks.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'distance.npy'))
    ]

    num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)

    all_IDs = np.arange(num_all)
    # split 90/10
    split = int(0.9 * num_all)
    train_IDs = all_IDs[:split]
    val_IDs   = all_IDs[split:]
    # generatori
    #print(val_IDs)
    train_gen = DataGenerator(folds, train_IDs, batch_size=8, shuffle=True, augment=True)
    val_gen = DataGenerator(folds, val_IDs, batch_size=8, shuffle=False, augment=False)

    monitor_metric = 'val_seg_head_cell_dice'
    #monitor_metric = 'val_hv_head_hv_mae_builtin'
    #monitor_metric = "val_loss"
    #mixed_precision.set_global_policy('mixed_float16')
    #Dynamic Learning rate
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-4,
        mode="min",
        verbose=1
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs','fit'),
            histogram_freq=1,
            write_images=True
        )
    checkpoint_path='models/checkpoints/neo/model_BB_RGB2.keras'
        # Checkpoint
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=monitor_metric,
            mode="min",
            verbose=1
        )

        # EarlyStopping
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=5,
            restore_best_weights=True
        )
    Xb, Yb = train_gen[0]
    k = min(4, Xb.shape[0])                                 # quante immagini mostrare
    high = max(1, Xb.shape[0] - (k - 1))                    # ultimo indice di partenza valido + 1
    idx = np.random.randint(0, high)    
    
    print(Xb.shape, Yb['seg_head'].shape, Yb['hv_head'].shape)

    hv_batch = Yb['hv_head']
    #print(f"min: {np.min(hv_batch)}, max {np.max(hv_batch)}, mean {np.mean(hv_batch)}")
    """
"""  
for i in range(idx,idx + 4):
        fig, axs = plt.subplots(2, 3, figsize=(6,6))
        image = Xb[i]       # (H, W, 3) – RGB image (non usata da GenInstanceHV)
        mask_3ch = Yb['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
        hv = Yb['hv_head'][i]
        body_mask = mask_3ch[..., 0].astype(np.uint8)  # binaria
        border_mask = mask_3ch[..., 2].astype(np.uint8)
        background = mask_3ch[..., 1].astype(np.uint8)

        C = np.logical_or(body_mask, border_mask).astype(np.uint8)
        axs[0,0].imshow(image)                         # immagine normalizzata [0,1]
        axs[0,1].imshow(body_mask, cmap='gray')  # body
        axs[0,2].imshow(background, cmap='gray')  # background
        axs[1,0].imshow(border_mask, cmap='gray')  # border
        axs[1,1].imshow(hv[...,0])  # body
        axs[1,2].imshow(hv[...,1])  # background
    
        plt.tight_layout()
        plt.show()
"""
"""
    #from src.losses import hover_loss_fixed,bce_dice_loss,hover_loss_fixed,hv_keras_loss,hovernet_hv_loss_tf
    
    model = get_model_paper()
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10*len(train_gen),   # 10 epoche
        t_mul=2.0,                                # lunghezza fase raddoppia
        m_mul=0.8,                                # ampiezza si riduce
        alpha=1e-5 / 1e-3                         # min_lr
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         weight_decay=1e-4)
    

    @tf.keras.utils.register_keras_serializable()
    def hv_mae_builtin(y_true, y_pred):
        # MAE mascherata sui primi 2 canali (HV)
        hv_t = tf.cast(y_true[..., :2], tf.float32)
        hv_p = tf.cast(y_pred,          tf.float32)
        mask = tf.cast(y_true[...,  2], tf.float32)    # (B,H,W)

        # per-pixel MAE sui 2 canali
        per_pix = tf.reduce_mean(tf.square(hv_t - hv_p), axis=-1)      # (B,H,W)
        return tf.reduce_sum(per_pix * mask) / (tf.reduce_sum(mask) + 1e-8)
    from src.losses import HVLossMonaiTorch,bce_dice_loss_ops

    model.compile(
    optimizer=optimizer,
    run_eagerly=True,
    loss={
        'seg_head': bce_dice_loss_ops,
        'hv_head': HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=1.0, ksize=5)#HoVerHVLoss(lambda_mse=3.0, lambda_grad=1.0, kernel_size=5)  # <--- funzione non parametrica
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 2.0
    },
    metrics={'seg_head': [CellDice()],
             'hv_head' : []}
)
    print(len(model.trainable_weights))
    h1 = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=10,
                        callbacks=[
                            earlystop_cb,
                            checkpoint_cb,
                            tensorboard_cb,
                        ],
                        verbose=1
                        )
                        """
"""
    from src.model import build_unet_resnet50

    model, base = build_unet_resnet50()

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10*len(train_gen),   # 10 epoche
        t_mul=2.0,                                # lunghezza fase raddoppia
        m_mul=0.8,                                # ampiezza si riduce
        alpha=1e-5 / 1e-3                         # min_lr
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         weight_decay=1e-4,
                                         clipnorm=1.0)

    model.compile(
    optimizer=optimizer,
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hv_keras_loss  # <--- funzione non parametrica
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 1.5
    },
    metrics={'seg_head': [CellDice()],
             'hv_head' : []}
)
    print(len(model.trainable_weights))
    h1 = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=50,
                        callbacks=[
                            earlystop_cb,
                            checkpoint_cb,
                            tensorboard_cb,
                        ],
                        verbose=1
                        )

    for L in model.layers[int(0.25*len(model.layers)):]:
        L.trainable = True

    opt = tf.keras.optimizers.SGD(learning_rate=3e-4, 
                                  momentum=0.9,
                                  nesterov=True, 
                                  weight_decay=1e-4
                                  )
    
    model.compile(optimizer=opt,
                  loss={'seg_head': bce_dice_loss,
                        'hv_head': hv_keras_loss},
                  loss_weights={'seg_head': 1.0,
                                 'hv_head': 1.5}
                                 )
    
    checkpoint_path='models/checkpoints/neo/model_BB_RGB2_second.keras'
        # Checkpoint
    monitor_metric = 'val_seg_head_cell_dice'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=monitor_metric,
            mode="max",
            verbose=1
        )
    print(len(model.trainable_weights))
    h2 = model.fit(train_gen, 
              validation_data=val_gen, 
              epochs=20,
              callbacks=[
                earlystop_cb,
                checkpoint_cb,
                tensorboard_cb,
                        ],
              verbose=1
                         )
    
    # tutto il backbone
    for L in model.layers: 
        L.trainable = True

    opt = tf.keras.optimizers.SGD(learning_rate=1e-4, 
                                  momentum=0.9,
                                  nesterov=True, 
                                  weight_decay=1e-4
                                  )
    
    model.compile(optimizer=opt,
                  loss={'seg_head': bce_dice_loss,
                        'hv_head': hv_keras_loss},
                  loss_weights={'seg_head': 1.0,
                                 'hv_head': 2.0}
                                 )
    print(len(model.trainable_weights))
    checkpoint_path='models/checkpoints/neo/model_BB_RGB2_third.keras'
        # Checkpoint
    monitor_metric = "val_loss"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=monitor_metric,
            mode="min",
            verbose=1
        )

    h3 = model.fit(train_gen, 
              validation_data=val_gen,
             epochs=30,
             callbacks=[
                earlystop_cb,
                checkpoint_cb,
                tensorboard_cb,
            ],
            verbose=1)
    #OVERFITTING ?


#OVERFITTING ?
histories = [h1.history, h2.history, h3.history]
metrics = set().union(*[h.keys() for h in histories])

merged = {
    m: np.concatenate([np.array(h.get(m, []), dtype=float) for h in histories])
    for m in metrics
}

# indici epoca cumulativi
n1, n2, n3 = len(h1.history['loss']), len(h2.history['loss']), len(h3.history['loss'])
epochs = np.arange(1, n1 + n2 + n3 + 1)

# --- PLOT ---
plt.figure(figsize=(7,4))
plt.plot(epochs, merged['loss'], label='train loss')
plt.plot(epochs, merged['val_loss'], label='val loss')
# linee verticali a separare le fasi
plt.axvline(n1+0.5, ls='--', alpha=0.5, label='Phase boundary')
plt.axvline(n1+n2+0.5, ls='--', alpha=0.5)
plt.xlabel('Epoch (cumulative)'); plt.ylabel('Loss'); plt.title('Unified loss')
plt.legend(); plt.tight_layout(); plt.show()
"""