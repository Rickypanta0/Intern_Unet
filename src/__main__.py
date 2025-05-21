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
from src.model import get_model_paper, build_unet_with_resnet50
from src.callbacks import get_callbacks
import math 
from src.losses import csca_binary_loss, bce_dice_loss,hover_loss_fixed
from tensorflow import keras 
import tensorflow_io as tfio
import segmentation_models as sm
from src.test_augm import build_instance_map_valuewise, GenInstanceHV
import cv2
if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # attiva memory growth **prima** di qualunque allocazione
        tf.config.experimental.set_memory_growth(gpus[0], True)
    SEED = 42
    np.random.seed = SEED
BACKBONE = 'resnet50'
class NpyDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 folds: list[tuple[str,str]],
                 list_IDs: list[int],
                 batch_size: int = 16,
                 patch_size: int = None,
                 shuffle: bool = True,
                 mmap: bool = False):
        """
        folds      : lista di tuple (path_images_npy, path_masks_npy)
        list_IDs   : lista di indici di sample [0 … total_samples-1]
        batch_size : dimensione del batch
        patch_size : se non None, estrae patch di lato patch_size
        shuffle    : se True mescola gli indici a ogni epoca
        mmap       : se True usa np.load(mmap_mode='r') per non caricare tutto in RAM
        """
        self.folds = folds
        self.list_IDs   = list_IDs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle    = shuffle
        """
        # --- 1) carica tutti i .npy in due unici array X e Y ---
        loader = lambda p: np.load(p, mmap_mode='r') if mmap else np.load(p)
        X_list = []
        Y_list = []
        Z_list = []
        for img_npy, mask_npy, mask_multi_ch in folds:
            X_list.append(loader(img_npy))
            Y_list.append(loader(mask_npy))
            Z_list.append(loader(mask_multi_ch))
        # concatenazione
        self.X = np.concatenate(X_list, axis=0)
        self.Y = np.concatenate(Y_list, axis=0)[...,None]  # aggiunge canale
        self.Z = np.concatenate(Z_list, axis=0)
        self.n_samples = self.X.shape[0]
        """
        self.sample_map = []  # [(fold_idx, local_idx), ...]

        for fold_idx, (img_path, mask_path, hv_path) in enumerate(folds):
            n_samples = np.load(img_path, mmap_mode='r').shape[0]
            for i in range(n_samples):
                self.sample_map.append((fold_idx, i))
        self.on_epoch_end()
        self.preprocess_input = sm.get_preprocessing(BACKBONE)

    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        # indici [start:stop) nella lista list_IDs
        batch_idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_temp = [self.sample_map[i] for i in batch_idxs]
        return self.__data_generation(list_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         hv_np : np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        img_np: np.ndarray (H,W) o (H,W,1) o (H,W,3), valori [0..1] o [0..255]
        mask_np: np.ndarray (H,W), valori {0,1}
        """
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

    def _make_target_3ch(self, m2: np.ndarray) -> np.ndarray:
        """Dalla maschera 2D binaria (H,W) ritorna (H,W,3) [body,bg,border]."""
        struct = np.ones((3,3), dtype=bool)
        body       = binary_erosion(m2, structure=struct).astype(np.uint8)
        border     = (m2 - body).clip(0,1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)
        return np.stack([body, background, border], axis=-1)


    def __data_generation(self, list_temp):
        # numero di augmentazioni per sample
        n_aug =  1  # 4 zoom + flipLR + flipUD, ad esempio
        batch_n = len(list_temp) * n_aug

        # dimensioni immagine
        if self.patch_size is None:
            sample_img = np.load(self.folds[0][0], mmap_mode='r')[0]
            H, W, C = sample_img.shape
        else:
            H = W = self.patch_size
            C = 3
        
        # pre-allocazione
        Xb = np.empty((batch_n, H, W, C), dtype=np.float32)
        Yb = np.empty((batch_n, H, W, 3),   dtype=np.float32)#"""uint8"""
        HVb = np.empty((batch_n, H, W, 2), dtype=np.float32)

        k = 0   # contatore totale delle righe di Xb/Yb

        for idx in list_temp:
            img_ = (self.X[idx]).astype(np.float32)
            img_ = img_ + 15
            img_ = img_ / 255      # np.ndarray (H,W,C), valori 0–255
            img_gray = np.dot(img_[...,:3], [0.2989, 0.5870, 0.1140])
            img = img_gray[..., np.newaxis].astype(np.float32)  # (H,W,1)
            
            hv_map = self.Z[idx]  # shape (H,W,6)
            #instance_map = build_instance_map_valuewise(mask_6ch)  # definita prima
            #img_input = instance_map[..., np.newaxis]
#
            ## HV map tramite GenInstanceHV
            #gen = GenInstanceHV(crop_shape=instance_map.shape)
            #img_out = gen._augment(img_input, None)
            #hv_map = img_out[..., 1:3]  # (H,W,2)

            #print(img.shape)
            raw = self.Y[idx]          # np.ndarray (H,W,1)
            msk = np.squeeze(raw, axis=-1) if raw.ndim == 3 and raw.shape[-1] == 1 else raw           # ora (H,W)

            # 1) originale
            Xb[k] = img.astype(np.float32)
            Yb[k] = self._make_target_3ch(msk)
            HVb[k] = hv_map
            k += 1
            """"""
            # 2) zoom augmentations (4 seed diversi)
            #for j in range(1):
            #    seed = (idx * 31 + j * 17, idx * 127 + j * 29)  # o qualunque combinazione di int
            #    xz, yz, hv_n = self.augm(seed, img, msk,hv_map, zoom_size=180, IMG_SIZE=256)
            #    Xb[k] = xz
            #    yz_ = np.squeeze(yz, axis=-1) if yz.ndim == 3 and yz.shape[-1] == 1 else yz
            #    HVb[k] = hv_n
            #    Yb[k] = self._make_target_3ch(yz_)
            #    k += 1
            # 3) flip left–right
            #img_lr = np.fliplr(img)
            #mask_lr = np.fliplr(msk)
            #hv_map_lr = np.fliplr(hv_map)
            #Xb[k] = img_lr.astype(np.float32)
            #Yb[k] = self._make_target_3ch(mask_lr)
            #HVb[k] = hv_map_lr
            #k += 1
#
            ## 4) flip up–down
            #img_ud = np.flipud(img)
            #mask_ud= np.flipud(msk)
            #hv_map_ud= np.flipud(hv_map)
            #Xb[k] = img_ud.astype(np.float32)
            #Yb[k] = self._make_target_3ch(mask_ud)
            #HVb[k] = hv_map_ud
            #k += 1
        preprocess_input = sm.get_preprocessing(BACKBONE)
        Xb = preprocess_input(Xb)
        return Xb, {'seg_head': Yb, 'hv_head': HVb}

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folds, list_IDs, batch_size=1, patch_size=None,
                 shuffle=True, augment=True):
        self.folds = folds  # lista di triple (img_path, mask_path, hv_path)
        self.list_IDs = list_IDs  # global indices
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.augment = augment
        self.preprocess_input = sm.get_preprocessing(BACKBONE)

        # Build global index map: global_idx -> (fold_idx, local_idx)
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
        body = binary_erosion(m2, structure=struct).astype(np.uint8)
        border = (m2 - body).clip(0, 1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)
        return np.stack([body, background, border], axis=-1)

    def __data_generation(self, list_temp):
        if self.patch_size is None:
            H, W, _ = self.default_shape
        else:
            H = W = self.patch_size
        C = 3

        X = np.empty((self.batch_size, H, W, C), dtype=np.float32)
        Y = np.empty((self.batch_size, H, W, 3), dtype=np.float32)
        HV = np.empty((self.batch_size, H, W, 2), dtype=np.float32)

        for i, (fold_idx, local_idx) in enumerate(list_temp):
            img_path, mask_path, hv_path = self.folds[fold_idx]

            img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
            img = (img + 15) / 255.0
            img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            img_gray = img_gray[..., np.newaxis]
            img_rgb = np.repeat(img_gray, 3, axis=-1)

            mask = np.load(mask_path, mmap_mode='r')[local_idx]
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)

            hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)

            # Augmentation
            if self.augment:
                if np.random.rand() < 0.5:
                    img_rgb = np.fliplr(img_rgb)
                    mask = np.fliplr(mask)
                    hv = np.fliplr(hv)
                if np.random.rand() < 0.5:
                    img_rgb = np.flipud(img_rgb)
                    mask = np.flipud(mask)
                    hv = np.flipud(hv)

            X[i] = self.preprocess_input(img_rgb)
            Y[i] = self._make_target_3ch(mask)
            HV[i] = hv

        return X, {'seg_head': Y, 'hv_head': HV}

base = os.path.join( 'data','raw')
folds = [
    (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
     os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy'),
     os.path.join(base, 'Fold 1', 'masks', 'fold1', 'distance.npy')),
     (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'distance.npy')),
    (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
     os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'),
     os.path.join(base, 'Fold 2', 'masks', 'fold2', 'distance.npy'))
]


num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)
all_IDs = list(range(num_all))
# split 90/10
split = int(0.9 * num_all)
train_IDs = all_IDs[:split]
val_IDs   = all_IDs[split:]
# generatori
#train_gen = NpyDataGenerator(folds, train_IDs, batch_size=1, shuffle=True, mmap=True)
#val_gen   = NpyDataGenerator(folds, val_IDs,   batch_size=1, shuffle=False, mmap=True)
train_gen = DataGenerator(folds, train_IDs, batch_size=8, shuffle=True, augment=True)
val_gen = DataGenerator(folds, val_IDs, batch_size=8, shuffle=False, augment=False)
#Dynamic Learning rate
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
)

tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs','fit'),
        histogram_freq=1,
        write_images=True
    )
checkpoint_path='models/checkpoints/model_grayscale_HV_v2.keras'
    # Checkpoint
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=False,
        monitor='val_loss',
        verbose=1
    )

    # EarlyStopping
earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1
    )
#unfreeze_cb = UnfreezeCallback(unfreeze_epoch=1)
Xb, Yb = train_gen[0]
#idx = np.random.randint(Xb.shape[0]-14)
idx = 0
print(Xb.shape, Yb['seg_head'].shape, Yb['hv_head'].shape)

print(Xb.shape)
#for i in range(idx,idx + 5):
#    fig, axs = plt.subplots(2, 2, figsize=(6,6))
#    image = Xb[i]          # (H, W, 3) – RGB image (non usata da GenInstanceHV)
#    mask_3ch = Yb['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
#    body_mask = mask_3ch[..., 0].astype(np.uint8)  # binaria
#
#    axs[0,0].imshow(Xb[i,...,0])                         # immagine normalizzata [0,1]
#    axs[0,1].imshow(body_mask,  vmin=0, vmax=1)  # body
#    axs[1,0].imshow(Yb['hv_head'][i, ..., 0])  # background
#    axs[1,1].imshow(Yb['hv_head'][i, ..., 1])  # border
#
#    plt.tight_layout()
#    plt.show()
from segmentation_models import Unet, get_preprocessing
from segmentation_models.utils import set_trainable
BACKBONE = 'resnet34'
#preprocess_input = get_preprocessing(BACKBONE)
#train_gen = preprocess_input(train_gen)
#val_gen = preprocess_input(val_gen)

# Costruzione del backbone base da segmentation_models
base_unet = Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',
    input_shape=(256, 256, 3),
    decoder_block_type='upsampling',
    decoder_use_batchnorm=True,
    decoder_filters=(256,128,64,32,16),
    encoder_freeze=True,
    activation=None  # niente attivazione sull'output di default
)

# Output intermedio dell'ultimo layer del decoder
decoder_output = base_unet.output

# Testa segmentazione (3 classi - softmax)
seg_head = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(decoder_output)
seg_head = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax', name='seg_head')(seg_head)

# Testa HV (2 canali - regressione)
hv_head = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(decoder_output)
hv_head = tf.keras.layers.Conv2D(2, (1, 1), activation='linear', name='hv_head')(hv_head)

# Modello finale multi-output
model = tf.keras.Model(inputs=base_unet.input, outputs={'seg_head': seg_head, 'hv_head': hv_head})

# Compilazione con le loss desiderate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hover_loss_fixed
    },
    loss_weights={
        'seg_head': 2.0,
        'hv_head': 1.0
    }
)

model.fit(train_gen,
          validation_data=val_gen,
          epochs=2,
          callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
          verbose=1)

# Sblocco encoder
for layer in model.layers:
    layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hover_loss_fixed
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 2.0
    }
)

# Allenamento completo
model.fit(train_gen,
          validation_data=val_gen,
          epochs=100,
          callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
          verbose=1)

"""
model = get_model_paper()

model.fit(train_gen,
        validation_data=val_gen, 
        epochs=50, 
        callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
        verbose=1)
"""
