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
import cv2
if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limita l'uso della memoria GPU a 4 GB (ad esempio)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=9500)]
            )
        except RuntimeError as e:
            print(e)
    SEED = 42
    np.random.seed = SEED

class UnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.done = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self.done and epoch == self.unfreeze_epoch:
            print(f"\n>> Epoca {epoch}: scongelo tutto il backbone")
            self.model.backbone.trainable = True

            # crei anche i momenti m/v di Adam per i nuovi pesi
            self.model.optimizer._create_all_weights(self.model.trainable_variables)

            self.done = True


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
        img_np: np.ndarray (H,W,1) o (H,W,3), valori 0–255 o già normalizzati
        mask_np: np.ndarray (H,W)
        """
        # 1) Assicuriamoci uint8 [0–255] e 2D/3D coerente
        #if img_np.dtype != np.uint8:
        #    img_np = (img_np*255).clip(0,255).astype(np.uint8)
        if img_np.ndim == 2:
            img_np = img_np[...,None]

        # 2) Da numpy a tensor float32
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)
        C     = img_t.shape[-1]
        if C == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)
            C     = 3

        # 3) Pad per consentire il crop
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE, IMG_SIZE)

        # 4) Genera un seed diverso per ogni chiamata (passato da fuori)
        seed_tf = tf.convert_to_tensor(seed, dtype=tf.int32)
        margin = (zoom_size // 2) + 1  
        padded_m = tf.pad(
            img_t,
            paddings=[[margin, margin],
                      [margin, margin],
                      [0, 0]],
            mode='REFLECT'
        )
        # 5) Ritaglio vero e proprio (random crop)
        crop = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, C],
            seed=seed_tf
        )

        # 6) Ridimensiono il ritaglio a IMG_SIZE
        xz = tf.image.resize(crop, [IMG_SIZE, IMG_SIZE])
# pad riflettente ai bordi
        # 7) Stessa cosa per la mask
        mask_t = tf.convert_to_tensor(mask_np[...,None], dtype=tf.float32)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE, IMG_SIZE)
        padded = tf.pad(
            mask_t,
            paddings=[[margin, margin],
                      [margin, margin],
                      [0, 0]],
            mode='REFLECT'
        )
        crop_m   = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, 1],
            seed=seed_tf
        )
        yz = tf.image.resize(crop_m, [IMG_SIZE, IMG_SIZE])

        # 8) Da tensor a numpy
        xz_np = xz.numpy().astype(np.float32)   # /255 se ti serve [0,1]
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
        n_aug =  6  # 4 zoom + flipLR + flipUD, ad esempio
        batch_n = len(list_temp) * n_aug

        # dimensioni immagine
        H, W, C = self.X.shape[1:] if self.patch_size is None else (self.patch_size,)*2 + (self.X.shape[-1],)

        # pre-allocazione
        Xb = np.empty((batch_n, H, W, C), dtype=np.float32)
        Yb = np.empty((batch_n, H, W, 3),   dtype=np.uint8)

        k = 0   # contatore totale delle righe di Xb/Yb

        for idx in list_temp:
            img_ = (self.X[idx]/255).astype(np.float32)      # np.ndarray (H,W,C), valori 0–255
            img_gray = np.dot(img_[...,:3], [0.2989, 0.5870, 0.1140])
            img = img_gray[..., np.newaxis].astype(np.float32)  # (H,W,1)
            raw = self.Y[idx]          # np.ndarray (H,W,1)
            msk = np.squeeze(raw, axis=-1) if raw.ndim == 3 and raw.shape[-1] == 1 else raw           # ora (H,W)

            # 1) originale
            Xb[k] = img.astype(np.float32)
            Yb[k] = self._make_target_3ch(msk)
            k += 1
            """"""
            # 2) zoom augmentations (4 seed diversi)
            for j in range(3):
                seed = (idx * 31 + j * 17, idx * 127 + j * 29)  # o qualunque combinazione di int
                xz, yz = self.augm(seed, img, msk, zoom_size=180, IMG_SIZE=256)
                Xb[k] = xz
                yz_ = np.squeeze(yz, axis=-1) if yz.ndim == 3 and yz.shape[-1] == 1 else yz
                Yb[k] = self._make_target_3ch(yz_)
                k += 1
            # 3) flip left–right
            img_lr = np.fliplr(img)
            mask_lr = np.fliplr(msk)
            Xb[k] = img_lr.astype(np.float32)
            Yb[k] = self._make_target_3ch(mask_lr)
            k += 1

            # 4) flip up–down
            img_ud = np.flipud(img)
            mask_ud= np.flipud(msk)
            Xb[k] = img_ud.astype(np.float32)
            Yb[k] = self._make_target_3ch(mask_ud)
            k += 1

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
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
        )

tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs','fit'),
        histogram_freq=1,
        write_images=True
    )
checkpoint_path='models/checkpoints/model_grayscale_v2.keras'
    # Checkpoint
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        #save_weights_only=True,
        monitor='val_loss',
        verbose=1
    )

    # EarlyStopping
earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        verbose=1
    )
#unfreeze_cb = UnfreezeCallback(unfreeze_epoch=1)
Xb, Yb = train_gen[0]
idx = np.random.randint(Xb.shape[0]-14)

for i in range(idx,idx + 14):
    fig, axs = plt.subplots(2, 2, figsize=(6,6))

    axs[0,0].imshow(Xb[i],         vmin=0, vmax=1)                         # immagine normalizzata [0,1]
    axs[0,1].imshow(Yb[i,:,:,0], cmap='gray', vmin=0, vmax=1)  # body
    axs[1,0].imshow(Yb[i,:,:,1], cmap='gray', vmin=0, vmax=1)  # background
    axs[1,1].imshow(Yb[i,:,:,2], cmap='gray', vmin=0, vmax=1)  # border

    plt.tight_layout()
    plt.show()
print("Xb shape:", Xb.shape, "dtype:", Xb.dtype)
print("Yb shape:", Yb.shape, "dtype:", Yb.dtype)

model = get_model_paper()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb],
    verbose=1
)
"""
model = build_unet_with_resnet50()
for layer in model.backbone.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=bce_dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=3)]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb],
    verbose=1
)

# Salva i pesi migliori dai callback:

# (assicurati che checkpoint_cb avesse save_weights_only=True e filepath=checkpoint_path)


# ——— PULIZIA DELLA SESSIONE ———
# questo svuota il grafo TF e libera la memoria GPU
tf.keras.backend.clear_session()
import gc; gc.collect()

# RI-CONFIGURA la GPU (memory growth, visible devices, ecc.)
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    for g in gpus:
#        tf.config.experimental.set_memory_growth(g, True)
#    # (opzionale) tf.config.set_visible_devices(gpus[0], 'GPU')

# ——— FASE 2: fine-tuning ———
model = build_unet_with_resnet50()
# sblocca solo gli ultimi N layer
model.backbone.trainable = True
fine_tune_at = 100
for i, layer in enumerate(model.backbone.layers):
    layer.trainable = (i >= fine_tune_at)

# carica i pesi dalla prima fase
model.load_weights(checkpoint_path)

# ricompila con lr più basso
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss=bce_dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=3)]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=1,
    epochs=50,
    callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb],
    verbose=1
)
"""
#model, history= train(X_train, Y_train, X_test, Y_test,epochs=50,batch_size=8)
