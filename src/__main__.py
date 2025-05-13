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
from src.losses import csca_binary_loss, bce_dice_loss
from tensorflow import keras 
import tensorflow_io as tfio
import segmentation_models as sm
import cv2
if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # attiva memory growth **prima** di qualunque allocazione
        tf.config.experimental.set_memory_growth(gpus[0], True)
    SEED = 42
    np.random.seed = SEED
BACKBONE = 'resnet34'
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
        # --- 1) carica tutti i .npy in due unici array X e Y ---
        loader = lambda p: np.load(p, mmap_mode='r') if mmap else np.load(p)
        X_list = []
        Y_list = []
        for img_npy, mask_npy in folds:
            X_list.append(loader(img_npy))
            Y_list.append(loader(mask_npy))
        # concatenazione
        self.X = np.concatenate(X_list, axis=0)
        self.Y = np.concatenate(Y_list, axis=0)[...,None]  # aggiunge canale
        self.n_samples = self.X.shape[0]

        self.list_IDs   = list_IDs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        # indici [start:stop) nella lista list_IDs
        batch_idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_temp  = [self.list_IDs[i] for i in batch_idxs]
        return self.__data_generation(list_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    """
    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:
        
        img_np: numpy array, shape (H,W,1) o (H,W,3), valori 0–255 o già normalizzati
        mask_np: numpy array, shape (H,W)
        

        # 1) Normalizza e porta a float32
        # Se è (H,W) → (H,W,1)
        if img_np.ndim == 2:
            img_np = img_np[..., np.newaxis]

        # 2) TensorFlow tensor
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)  # shape (H,W,C)

        # 3) Se è single-channel, duplica in RGB
        C = int(img_np.shape[-1])             # 1 o 3

        if C == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)            # ora (H,W,3)
            C = 3

        # 4) Padding
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE + 6, IMG_SIZE + 6)

        # 5) Stateless random crop, usando C canali
        seed_tf = tf.convert_to_tensor(seed, dtype=tf.int32)
        seed0   = tf.random.experimental.stateless_split(seed_tf, num=1)[0]
        crop    = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, C],
            seed=seed0
        )

        # 6) Zoom-centered pad
        zoomed_padded = tf.image.resize(
            tf.image.central_crop(padded, central_fraction=zoom_size/IMG_SIZE),
            [IMG_SIZE, IMG_SIZE]
        )

        # 7) Stessa logica per la mask (sempre 1 canale)
        mask_t = tf.expand_dims(tf.convert_to_tensor(mask_np, dtype=tf.float32), axis=-1)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE + 6, IMG_SIZE + 6)
        crop_m   = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, 1],
            seed=seed0
        )
        zoomed_padded_m = tf.image.resize(
            tf.image.central_crop(padded_m, central_fraction=zoom_size/IMG_SIZE),
            [IMG_SIZE, IMG_SIZE]
        )

        # 8) Torna a NumPy
        xz = zoomed_padded.numpy().astype(np.float32)
        yz = zoomed_padded_m.numpy().astype(np.uint8)

        return xz, yz
            """
    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
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

        # 9) Torno a NumPy
        xz_np = xz.numpy().astype(np.float32)
        yz_np = yz.numpy().astype(np.uint8)

        return xz_np, yz_np

    def _make_target_3ch(self, m2: np.ndarray) -> np.ndarray:
        """Dalla maschera 2D binaria (H,W) ritorna (H,W,3) [body,bg,border]."""
        struct = np.ones((3,3), dtype=bool)
        body       = binary_erosion(m2, structure=struct).astype(np.uint8)
        border     = (m2 - body).clip(0,1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)
        return np.stack([body, background, border], axis=-1)


    def __data_generation(self, list_temp):
        # numero di augmentazioni per sample
        n_aug =  5  # 4 zoom + flipLR + flipUD, ad esempio
        batch_n = len(list_temp) * n_aug

        # dimensioni immagine
        H, W, C = self.X.shape[1:] if self.patch_size is None else (self.patch_size,)*2 + (self.X.shape[-1],)
        
        # pre-allocazione
        Xb = np.empty((batch_n, H, W, C), dtype=np.float32)
        Yb = np.empty((batch_n, H, W, 1),   dtype=np.float32)#"""uint8"""

        k = 0   # contatore totale delle righe di Xb/Yb

        for idx in list_temp:
            img_ = (self.X[idx]).astype(np.float32)
            img_ = img_ + 15
            img_ = img_ / 255      # np.ndarray (H,W,C), valori 0–255
            img_gray = np.dot(img_[...,:3], [0.2989, 0.5870, 0.1140])
            img = img_gray[..., np.newaxis].astype(np.float32)  # (H,W,1)
            
            #print(img.shape)
            raw = self.Y[idx]          # np.ndarray (H,W,1)
            #msk = np.squeeze(raw, axis=-1) if raw.ndim == 3 and raw.shape[-1] == 1 else raw           # ora (H,W)

            # 1) originale
            Xb[k] = img.astype(np.float32)
            Yb[k] = raw#self._make_target_3ch(msk)
            k += 1
            """"""
            # 2) zoom augmentations (4 seed diversi)
            for j in range(2):
                seed = (idx * 31 + j * 17, idx * 127 + j * 29)  # o qualunque combinazione di int
                xz, yz = self.augm(seed, img, raw, zoom_size=180, IMG_SIZE=256)
                Xb[k] = xz
                #yz_ = np.squeeze(yz, axis=-1) if yz.ndim == 3 and yz.shape[-1] == 1 else yz
                Yb[k] = yz#self._make_target_3ch(yz_)
                k += 1
            # 3) flip left–right
            img_lr = np.fliplr(img)
            mask_lr = np.fliplr(raw)
            Xb[k] = img_lr.astype(np.float32)
            Yb[k] = mask_lr#self._make_target_3ch(mask_lr)
            k += 1

            # 4) flip up–down
            img_ud = np.flipud(img)
            mask_ud= np.flipud(raw)
            Xb[k] = img_ud.astype(np.float32)
            Yb[k] = mask_ud#self._make_target_3ch(mask_ud)
            k += 1
        preprocess_input = sm.get_preprocessing(BACKBONE)
        Xb = preprocess_input(Xb)
        return Xb, Yb



base = os.path.join( 'data','raw')
folds = [
    (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
     os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy')),
     (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy')),
    (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
     os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'))
]


num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)
all_IDs = list(range(num_all))
# split 90/10
split = int(0.9 * num_all)
train_IDs = all_IDs[:split]
val_IDs   = all_IDs[split:]
# generatori
train_gen = NpyDataGenerator(folds, train_IDs, batch_size=4, shuffle=True, mmap=True)
val_gen   = NpyDataGenerator(folds, val_IDs,   batch_size=4, shuffle=False, mmap=True)

#Dynamic Learning rate
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
)

tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs','fit'),
        histogram_freq=1,
        write_images=True
    )
checkpoint_path='models/checkpoints/model_grayscale_v3.keras'
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
idx = np.random.randint(Xb.shape[0]-14)
"""
for i in range(idx,idx + 14):
    fig, axs = plt.subplots(2, 2, figsize=(6,6))

    axs[0,0].imshow(Xb[i], cmap='gray',        vmin=0, vmax=1)                         # immagine normalizzata [0,1]
    axs[0,1].imshow(Yb[i,:,:,0], cmap='gray', vmin=0, vmax=1)  # body
    axs[1,0].imshow(Yb[i,:,:,1], cmap='gray', vmin=0, vmax=1)  # background
    axs[1,1].imshow(Yb[i,:,:,2], cmap='gray', vmin=0, vmax=1)  # border

    plt.tight_layout()
    plt.show()
"""
print("Xb shape:", Xb.shape, "dtype:", Xb.dtype)
print("Yb shape:", Yb.shape, "dtype:", Yb.dtype)

mean = Xb.mean(axis=(0,1,2), keepdims=True)
    # se X è singola immagine (H,W,C), usa mean = Xf.mean(axis=(0,1), keepdims=True)

alpha = 1.4  # >1 aumenta il contrasto
# 2) formula del contrast stretch
Xc = mean + alpha * (Xb - mean)
# 3) ritaglia al range originale
if Xb.dtype == np.uint8:
    Xc = np.clip(Xc, 0, 255).astype(np.uint8)
else:
    Xc = np.clip(Xc, 0.0, 1.0)


os.environ["SM_FRAMEWORK"] = "tf.keras"
#BACKBONE = 'resnet34'
#preprocess_input = sm.get_preprocessing(BACKBONE)


# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(train_gen,
        validation_data=val_gen,
        batch_size=4,
        epochs=30
)
model.save('backbone_test.hdf5')
"""
model = get_model_paper()

model.fit(train_gen,
        validation_data=val_gen, 
        epochs=50, 
        callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
        verbose=1)
"""
#model, history= train(X_train, Y_train, X_test, Y_test,epochs=50,batch_size=8)
