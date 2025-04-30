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
from src.model import get_model_paper
from src.callbacks import get_callbacks
import math 
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

    def __data_generation(self, list_temp):
        print(f"[DataGen] Generating batch per indici {list_temp[:5]}…")

        # 1) decidiamo le dimensioni di Xb e Yb
        if self.patch_size is None:
            H, W, C = self.X.shape[1:]    # img shape = (N, H, W, C)
        else:
            H = W = self.patch_size
            C = self.X.shape[3]

        # 2) allochiamo gli array batch: Xb per le immagini, Yb per i 3 canali target
        Xb = np.empty((len(list_temp), H, W, C), dtype=np.float32)
        Yb = np.empty((len(list_temp), H, W, 3), dtype=np.uint8)

        for i, idx in enumerate(list_temp):
            # 3) estrai l'immagine e la maschera binaria  
            img = self.X[idx]           # shape (H, W, C), valori 0–255
            msk = self.Y[idx]           # shape (H, W) o (H, W, 1)
            if msk.ndim == 3 and msk.shape[-1] == 1:
                msk = msk[..., 0]       # ora msk è (H, W)

            # 4) (opzionale) crop a patch_size se lo desideri
            #    qui commentato perché non comune per istologia
            # if self.patch_size is not None:
            #     ps = self.patch_size
            #     img = img[:ps, :ps, :]
            #     msk = msk[:ps, :ps]

            # 5) normalizza l'immagine in [0,1]
            Xb[i] = (img.astype(np.float32) / 255.0)

            # 6) calcola i 3 canali target da msk binaria
            #    body = erode 1px
            struct = np.ones((3,3), dtype=bool)
            body       = binary_erosion(msk.astype(bool), structure=struct).astype(np.uint8)
            #    border = la differenza tra mask intera e body
            border     = (msk.astype(np.uint8) - body).clip(0,1)
            #    background = 1 - msk
            background = (1 - msk).astype(np.uint8)

            # 7) impacchetta i 3 canali in Yb
            #    canale 0 = body, 1 = background, 2 = border
            Yb[i, ..., 0] = body
            Yb[i, ..., 1] = background
            Yb[i, ..., 2] = border

        # 8) cast finale e ritorno
        return Xb.astype(np.float32), Yb.astype(np.float32)


base = os.path.join( 'data','raw')
folds = [
    (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
     os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy')),
     (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy')),
    (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
     os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'))
]

"""X, Y = load_folds(folds, normalize_images=True)
print(X.shape,Y.shape)
Y = np.expand_dims(Y, axis=-1)
Y_ = []
for M in Y:
    M2= np.squeeze(M,axis=-1)
    struct = np.ones((3,3), dtype=bool)
    body = binary_erosion(M2.astype(bool), structure=struct).astype(np.uint8)
    # body = 1 sulle regioni interne più “sicure”
    # 2) Il "bordo" è ciò che resta togliendo il body dal mask intero
    border = (M2.astype(np.uint8) - body)
    # border = 1 in corrispondenza dell’anello di 1 px attorno al body
    # 3) Il "background" è l’inverso della mask originale
    background = (1 - M2).astype(np.uint8)
    # 4) Costruisci il GT 3‐canali
    #    ordine: [body, background, border]
    Y_gt = np.stack([body, background, border], axis=-1)
    Y_.append(Y_gt)
Y_ = np.stack(Y_,axis=0)"""
num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)
all_IDs = list(range(num_all))
# split 90/10
split = int(0.9 * num_all)
train_IDs = all_IDs[:split]
val_IDs   = all_IDs[split:]
# generatori
train_gen = NpyDataGenerator(folds, train_IDs, batch_size=8, shuffle=True, mmap=True)
val_gen   = NpyDataGenerator(folds, val_IDs,   batch_size=8, shuffle=False, mmap=True)

model = get_model_paper()
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
        filepath=os.path.join('models','checkpoints','model_paper_v4.keras'),
        save_best_only=True,
        verbose=1
    )

    # EarlyStopping
earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        verbose=1
    )

Xb, Yb = train_gen[0]
print("Xb shape:", Xb.shape, "dtype:", Xb.dtype)
print("Yb shape:", Yb.shape, "dtype:", Yb.dtype)
model.fit(train_gen, validation_data=val_gen, epochs=50,verbose=1, callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, lr_cb])
#model, history= train(X_train, Y_train, X_test, Y_test,epochs=50,batch_size=8)
