import tensorflow as tf
import os
import numpy as np
import math
import random
#tqdm viene usato in un for loop per mostrare la progress bar
from tqdm import tqdm
from datetime import datetime
import gc
from skimage.transform import resize
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU
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

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, x_sample, y_sample):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.x_sample = x_sample  # shape: (1, 256, 256, 3)
        self.y_sample = y_sample  # shape: (1, 256, 256, 1)

    def on_epoch_end(self, epoch, logs=None):
        pred_mask = self.model.predict(self.x_sample, verbose=0)[0]  # shape: (256, 256, 1)

        # Input image (già RGB)
        input_img = (self.x_sample[0] * 255).astype(np.uint8)  # (256, 256, 3)

        # Ground truth
        gt_mask = (self.y_sample[0] * 255).astype(np.uint8)    # (256, 256) o (256, 256, 1)
        if gt_mask.ndim == 2:
            gt_mask = np.expand_dims(gt_mask, axis=-1)
        gt_rgb = np.repeat(gt_mask, 3, axis=-1)                # (256, 256, 3)

        # Prediction
        pred_mask = (pred_mask * 255).astype(np.uint8)         # (256, 256) o (256, 256, 1)
        if pred_mask.ndim == 2:
            pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_rgb = np.repeat(pred_mask, 3, axis=-1)            # (256, 256, 3)

        # Concatenazione orizzontale: input | gt | prediction
        display = np.concatenate([input_img, gt_rgb, pred_rgb], axis=1)  # (256, 768, 3)
        display = np.expand_dims(display, 0)  # (1, H, W, 3)  
        with self.file_writer.as_default():
            tf.summary.image("Input / Ground Truth / Prediction", display, step=epoch)

def dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

matplotlib.use("TkAgg")    
seed = 42
np.random.seed = seed

folds = [
    ('../dataSet/Fold 1/images/fold1/images.npy', '../dataSet/Fold 1/masks/fold1/masks_processed.npy'),
    ('../dataSet/Fold 2/images/fold2/images.npy', '../dataSet/Fold 2/masks/fold2/masks_processed.npy')
]
#    ('../dataSet/Fold 3/images/fold3/images.npy', '../dataSet/Fold 3/masks/fold3/masks_processed.npy')

X_list = []
Y_list = []

for img_path, mask_path in folds:
    print(f"Caricamento immagini: {img_path}")
    imgs = np.load(img_path).astype(np.float32) / 255.0
    X_list.append(imgs)

    print(f"Processing maschere: {mask_path}")
    masks = np.load(mask_path)                    # shape: (N, 256, 256, 6)
    #masks = masks_raw[:, :, :, 5].astype(np.uint8)    # shape: (N, 256, 256)
    Y_list.append(masks)
    X = np.concatenate(X_list, axis=0)  # (N, 256, 256, 3)
    Y = np.concatenate(Y_list, axis=0)  # (N, 256, 256)

print("✅ Caricamento e salvataggio completato!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
idx = random.randint(0, X.shape[0])
fg, axes = plt.subplots(1,2)
axes[0].imshow(X[idx])
axes[1].imshow(Y[idx])
plt.show()


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def weighted_bce_dice(y_true, y_pred, weight=5.0, smooth=1.0):
    # y_true, y_pred sono shape [B, H, W, 1]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 1) BCE pixel‑wise → [B, H, W]
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # 2) Mappa pesi su [B, H, W, 1] poi squeeze → [B, H, W]
    weight_map = y_true * weight + (1.0 - y_true)
    weight_map = tf.squeeze(weight_map, axis=-1)

    # 3) BCE pesato
    weighted_bce = bce * weight_map

    # 4) Dice loss
    y_t = tf.squeeze(y_true, axis=-1)  # [B, H, W]
    y_p = tf.squeeze(y_pred, axis=-1)
    y_t_flat = tf.reshape(y_t, [-1])
    y_p_flat = tf.reshape(y_p, [-1])
    intersection = tf.reduce_sum(y_t_flat * y_p_flat)
    dice_coeff = (2.0 * intersection + smooth) / (tf.reduce_sum(y_t_flat) + tf.reduce_sum(y_p_flat) + smooth)
    dice_loss = 1.0 - dice_coeff

    # 5) Loss finale
    return tf.reduce_mean(weighted_bce) + dice_loss

# helper per compile
def get_weighted_bce_dice_loss(weight=5.0):
    return lambda y_true, y_pred: weighted_bce_dice(y_true, y_pred, weight=weight)

λ1, λ2 = 2.0, 1.0
smooth = 1e-6

def csca_binary_loss(y_true, y_pred):
    """
    Lp adattato a P=2 classi (BG vs FG):
      L = λ1 * BCE + λ2 * DiceLoss(FG)
    y_true, y_pred: shape [B,H,W,1], valori in [0,1]
    """
    # 1) Binary cross‑entropy, medio su tutti i pixel
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    # 2) Dice loss sul foreground (pixel=1)
    y_t = tf.reshape(y_true,  [-1])
    y_p = tf.reshape(y_pred,  [-1])
    intersection = tf.reduce_sum(y_t * y_p)
    dice_coeff = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth
    )
    dice_loss = 1.0 - dice_coeff

    # 3) Loss composita
    return λ1 * bce + λ2 * dice_loss

Y = np.expand_dims(Y, axis=-1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0)

input = tf.keras.Input((256,256,3))
c1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal', padding='same')(input)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[input], outputs=[outputs])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=csca_binary_loss, metrics=[MeanIoU(num_classes=2)])
model.summary()

idx = np.random.randint(len(X_train))
x_sample = X_train[idx:idx+1]  # shape (1, 256, 256, 3)
y_sample = Y_train[idx:idx+1] 

#CALLBACKS
#Checkpoints -> per salvare i pesi del modello raggiunti fino a quel modello invece di ricominciare da capo
#Early Stopping -> troppe epochs (overfitting) troppe poche (underfitting), se non ci sono abbastanza miglioramenti dopo
#                    n epochs viene bloccato
#TensorBoard -> visualizzazione validation los in funzione delle epoche 
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
image_logger = ImageLogger(log_dir=log_dir, x_sample=x_sample, y_sample=y_sample)
callbacks = [
            tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.keras',verbose=1, save_best_only=True),
            image_logger,
            tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_images=True)]

result = model.fit(X_train,Y_train, validation_split=0.1,batch_size=16,epochs=50,callbacks=callbacks)

idx = random.randint(0, len(X_train))


split = int(0.9 * len(X_train))    # 90% train – 10% val

# 2) Definisci soglie e pre‐alloca liste
thresholds = [0.3, 0.4, 0.5]
Ptrain = [None] * len(thresholds)
Pval   = [None] * len(thresholds)
Ptest  = [None] * len(thresholds)

# 3) Ciclo di predizione
for idx, thr in enumerate(thresholds):
    # > thr restituisce bool, poi convertiamo in uint8
    preds_train = (model.predict(X_train[:split], verbose=1) > thr).astype(np.uint8)
    preds_val   = (model.predict(X_train[split:],   verbose=1) > thr).astype(np.uint8)
    preds_test  = (model.predict(X_test,            verbose=1) > thr).astype(np.uint8)

    Ptrain[idx] = preds_train
    Pval[idx]   = preds_val
    Ptest[idx]  = preds_test

# 4) Funzione per mostrare 3 righe × 2 immagini per soglia
def show_threshold_pairs(imgs, preds_list, thresholds):
    """
    Per ciascuna soglia in `thresholds`, pesca un’immagine casuale da `imgs`
    e mostra: [Originale | Predizione].
    Si assume che np.random.seed(seed) sia già stato chiamato.
    """
    n = len(thresholds)
    # estrai n indici casuali (0 <= idx < len(imgs))
    sample_idxs = np.random.randint(0, len(imgs), size=n)

    fig, axs = plt.subplots(n, 2,
                            figsize=(2 * 4, n * 4),
                            squeeze=False)

    for row, thr in enumerate(thresholds):
        idx_img = sample_idxs[row]
        pred    = preds_list[row][idx_img].squeeze()

        # Colonna 1: immagine originale
        axs[row, 0].imshow(imgs[idx_img])
        axs[row, 0].set_title("Image")
        axs[row, 0].axis("off")

        # Colonna 2: maschera predetta
        axs[row, 1].imshow(pred)
        axs[row, 1].set_title(f"Prediction @ {thr}")
        axs[row, 1].axis("off")

    plt.tight_layout()
    plt.show()

# Esempio di chiamata:
show_threshold_pairs(X_test, Ptest, thresholds)
# 3b. validation sample

show_threshold_pairs(X_train,Ptrain,thresholds)
