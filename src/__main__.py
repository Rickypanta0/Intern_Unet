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

    def augm(self,
            seed: tuple[int,int],
            img: np.ndarray,
            mask: np.ndarray,
            zoom_size:int,
            IMG_SIZE: int) -> tuple[np.ndarray,np.ndarray]:

        img  = tf.convert_to_tensor(img,  dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)


        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        #augmentation img
        padded = tf.image.resize_with_crop_or_pad(img, IMG_SIZE+6, IMG_SIZE+6)
        seed_ = tf.random.experimental.stateless_split(seed, num=2)
        seed = seed_[0]
        crop = tf.image.stateless_random_crop(padded, [zoom_size, zoom_size, 3], seed=seed)

        zoom_factor = zoom_size/IMG_SIZE
        new_size = int(IMG_SIZE * zoom_factor)

        delta = IMG_SIZE - zoom_size
        pad_h = IMG_SIZE - new_size
        pad_w = IMG_SIZE - new_size
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        """
        zoomed = tf.image.resize(crop, [new_size, new_size])  
        zoomed_padded = tf.pad(
            zoomed,
            [[pad_top, pad_bottom],
             [pad_left, pad_right],
             [0,       0      ]],            # nessun padding sui canali
            constant_values=0.0               # riempi con zeri
        )
        """
        zoomed_padded = tf.image.resize(
        tf.image.central_crop(padded, central_fraction=zoom_size/IMG_SIZE),
        [IMG_SIZE, IMG_SIZE]
        )


        mask = tf.expand_dims(mask, axis=-1)

        padded_mask = tf.image.resize_with_crop_or_pad(mask, IMG_SIZE+6, IMG_SIZE+6)
        crop_mask = tf.image.stateless_random_crop(padded_mask, [zoom_size, zoom_size, 1], seed=seed)

        zoom_factor = zoom_size/IMG_SIZE
        new_size = int(IMG_SIZE * zoom_factor)

        delta = IMG_SIZE - zoom_size
        pad_h = IMG_SIZE - new_size
        pad_w = IMG_SIZE - new_size
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        """
        zoomed = tf.image.resize(crop_mask, [new_size, new_size])  

        zoomed_padded_Y = tf.pad(
            zoomed,
            [[pad_top, pad_bottom],
             [pad_left, pad_right],
             [0,       0      ]],            # nessun padding sui canali
            constant_values=1               # riempi con zeri
        )
        """
        zoomed_padded_Y = tf.image.resize(
        tf.image.central_crop(padded_mask, central_fraction=zoom_size/IMG_SIZE),
        [IMG_SIZE, IMG_SIZE]
        )
        xp = zoomed_padded.numpy().astype(np.float32)
        yp = zoomed_padded_Y.numpy().astype(np.uint8)
        return xp, yp


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
            img = self.X[idx]/255          # np.ndarray (H,W,C), valori 0–255
            raw = self.Y[idx]          # np.ndarray (H,W,1)
            msk = np.squeeze(raw, axis=-1) if raw.ndim == 3 and raw.shape[-1] == 1 else raw           # ora (H,W)

            # 1) originale
            Xb[k] = img.astype(np.float32)
            Yb[k] = self._make_target_3ch(msk)
            k += 1
            """"""
            # 2) zoom augmentations (4 seed diversi)
            for j in range(3):
                seed = (idx, j)  # o qualunque combinazione di int
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

model = build_unet_with_resnet50()
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

    # Checkpoint
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models','checkpoints','model_backbone.weights.h5'),
        save_best_only=True,
        save_weights_only=True,
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
idx = np.random.randint(Xb.shape[0]-8)

for i in range(idx,idx + 7):
    fig, axs = plt.subplots(2, 2, figsize=(6,6))

    axs[0,0].imshow(Xb[i],         vmin=0, vmax=1)                         # immagine normalizzata [0,1]
    axs[0,1].imshow(Yb[i,:,:,0], cmap='gray', vmin=0, vmax=1)  # body
    axs[1,0].imshow(Yb[i,:,:,1], cmap='gray', vmin=0, vmax=1)  # background
    axs[1,1].imshow(Yb[i,:,:,2], cmap='gray', vmin=0, vmax=1)  # border

    plt.tight_layout()
    plt.show()
print("Xb shape:", Xb.shape, "dtype:", Xb.dtype)
print("Yb shape:", Yb.shape, "dtype:", Yb.dtype)

for layer in model.backbone.layers:
    layer.trainable = False

# Compilo con un LR “alto” per il solo decoder
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=bce_dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=3)]
)

# Alleno decoder per 5 epoche
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    batch_size=4,
    callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, lr_cb],
    verbose=1
)

tf.keras.backend.clear_session()
import gc; gc.collect()

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
model = build_unet_with_resnet50()

check = os.path.join('models', 'checkpoints', 'model_backbone.weights.h5')
    #X, Y = load_folds(folds=folds)

model.backbone.trainable = True
fine_tune_at = 100

# 2) Ri‐congelo solo gli ultimi layer del backbone
for i, layer in enumerate(model.backbone.layers):
    layer.trainable = (i >= fine_tune_at)

model.load_weights(checkpoint_path=check)

# 3) Ricompilo con un LR molto più basso
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss=bce_dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=3)]
)

# 4) Riprendo il training da epoca 5 a 50
model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=1,
    epochs=50,
    batch_size=4,
    callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, lr_cb],
    verbose=1
)


#model, history= train(X_train, Y_train, X_test, Y_test,epochs=50,batch_size=8)
