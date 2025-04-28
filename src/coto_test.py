import os
import numpy as np
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from src.model import get_model
from src.utils.visualization import show_threshold_pairs_test
from src.predict import predict_masks
from skimage.io import imread, imshow
from tqdm import tqdm
import math
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

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

partition = {'train':[],'validation':[]}
labels = {}

class CocoDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 coco: COCO,
                 img_infos: list,
                 base_dir: str,
                 catIds: list,
                 list_IDs: list,
                 batch_size: int = 16,
                 patch_size: int = None,
                 shuffle: bool = True):
        """
        coco       : istanza COCO già caricata
        img_infos  : lista di dict restituita da coco.loadImgs(imgIds)
        base_dir   : cartella contenente le immagini JPEG
        catIds     : lista di category_id (es. [3] per 'car')
        list_IDs   : lista di indici in img_infos su cui iterare
        batch_size : dimensione del batch
        patch_size : se non None, estrae patch di lato patch_size
        shuffle    : mescola gli indici a ogni epoca
        """
        self.coco       = coco
        self.img_infos  = img_infos
        self.base_dir   = base_dir
        self.catIds     = catIds
        self.list_IDs   = list_IDs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        # quante batch per epoca
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        # seleziona quali elementi di list_IDs entrano in questa batch
        batch_idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_temp  = [self.list_IDs[i] for i in batch_idxs]
        # genera X e Y
        X, Y = self.__data_generation(list_temp)
        return X, Y

    def on_epoch_end(self):
        # aggiorna l’array di indici e mescola se richiesto
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_temp):
        # initialization
        # se patch_size è None, useremo intera immagine
        first_info = self.img_infos[list_temp[0]]
        H = first_info['height'] if self.patch_size is None else self.patch_size
        W = first_info['width']  if self.patch_size is None else self.patch_size

        X = np.empty((len(list_temp), H, W, 3),  dtype=np.float32)
        Y = np.empty((len(list_temp), H, W, 1), dtype=np.uint8)

        for i, idx in enumerate(list_temp):
            info = self.img_infos[idx]
            img_path = os.path.join(self.base_dir, info['file_name'])
            img = cv2.imread(img_path)[:, :, ::-1].astype(np.float32) / 255.0

            # costruisci maschera cumulativa
            mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
            annIds = self.coco.getAnnIds(imgIds=[info['id']],
                                          catIds=self.catIds,
                                          iscrowd=None)
            for ann in self.coco.loadAnns(annIds):
                mask |= self.coco.annToMask(ann).astype(np.uint8)
            mask = mask[..., None]

            # se serve patch, pad + crop
            if self.patch_size:
                ps = self.patch_size
                # calcola padding
                pad_h = math.ceil(info['height']/ps)*ps - info['height']
                pad_w = math.ceil(info['width'] /ps)*ps - info['width']
                if pad_h>0 or pad_w>0:
                    img  = np.pad(img,  ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
                    mask = np.pad(mask, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
                # per semplicità, prendi la patch in alto a sinistra
                img  = img [0:ps, 0:ps]
                mask = mask[0:ps, 0:ps]

            X[i] = img
            Y[i] = mask

        return X, Y

coco     = COCO('data/coco/annotations/instances_train2017.json')
catIds   = coco.getCatIds(catNms=['car'])
imgIds   = coco.getImgIds(catIds=catIds)
infos    = coco.loadImgs(imgIds)

# 2) split degli **indici** di infos (non degli ID COCO)
all_idxs = list(range(len(infos)))
train_idx, val_idx = train_test_split(all_idxs, test_size=0.1, random_state=42)

# 3) crea i generatori
train_gen = CocoDataGenerator(
    coco=coco,
    img_infos=infos,
    base_dir='data/coco/images/train2017',
    catIds=catIds,
    list_IDs=train_idx,
    batch_size=16,
    patch_size=256,
    shuffle=True
)
val_gen = CocoDataGenerator(
    coco=coco,
    img_infos=infos,
    base_dir='data/coco/images/train2017',
    catIds=catIds,
    list_IDs=val_idx,
    batch_size=16,
    patch_size=256,
    shuffle=False
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1
    )

    #Dynamic Learning rate
dynamic_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        min_lr=0.0001
    )
checkpoint = os.path.join('models','checkpoints','COCO_model.keras')
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint,
        save_best_only=True,
        verbose=1
    )

# 4) chiami il tuo train **senza modifiche**:
model = get_model()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,              # se vuoi specificare il numero di epoche
    steps_per_epoch=len(train_gen), # opzionale, ma esplicito
    validation_steps=len(val_gen),  # idem
    callbacks=[dynamic_lr, earlystop_cb,checkpoint_cb],                # se ne hai
    verbose=1
)
    
thresholds = [0.3, 0.4, 0.5,0.6]
prob_train = model.predict(train_gen, verbose=1)
prob_val   = model.predict(val_gen,   verbose=1)
# 2) Applica ogni soglia
Ptrain = [(prob_train > t).astype(np.uint8) for t in thresholds]
Pval   = [(prob_val   > t).astype(np.uint8) for t in thresholds]

X_Tbatch, Y_Tbatch = train_gen[0]
fig, axs = plt.subplots(1,2,figsize=(8,8))
axs[0].imshow(X_Tbatch[2])
axs[1].imshow(Y_Tbatch[2])
plt.show()
Ptrain_batch = [p[:X_Tbatch.shape[0]] for p in Ptrain]

X_Vbatch, Y_Vbatch = val_gen[0]
Pval_batch = [p[:X_Vbatch.shape[0]] for p in Ptrain]

show_threshold_pairs_test(X_Tbatch,Y_Tbatch,Ptrain,thresholds=thresholds)
show_threshold_pairs_test(X_Vbatch,Y_Vbatch,Pval_batch,thresholds=thresholds)