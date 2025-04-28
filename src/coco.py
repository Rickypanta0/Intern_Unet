import os
import numpy as np
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from src.train import train
from src.utils.visualization import show_threshold_pairs_test
from src.predict import predict_masks
from skimage.io import imread, imshow
from tqdm import tqdm
import math
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import random

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
# 1) Setup COCO & percorsi
base = os.path.join('data','coco')
anno = os.path.join(base, 'annotations', 'instances_train2017.json')
coco = COCO(anno)

# 2) Filtra le immagini con 'car'
catIds   = coco.getCatIds(catNms=['car'])
imgIds = coco.getImgIds(catIds=catIds)
img_info = coco.loadImgs(imgIds)
print(img_info)
annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
ann = coco.loadAnns(annIds)
idx = np.random.randint(0, len(img_info))
file_name = img_info[idx]['file_name']
img_id = img_info[idx]['id']
h = img_info[idx]['height']
w = img_info[idx]['width']
print(file_name)
path = os.path.join(base,'images','train2017',file_name)
img = cv2.imread(path)[:,:,::-1]
img = np.asarray(img)/255
mask = np.zeros((h,w),dtype=np.uint8)
annIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=None)
for ann in coco.loadAnns(annIds):
    mask |= coco.annToMask(ann).astype(np.uint8)

checkpoint = os.path.join('models','checkpoints','COCO_model.keras')
model = tf.keras.models.load_model(checkpoint,compile=False)

patch_size = 256
H, W = img.shape[:2]

# pad calcolato
pad_h = math.ceil(H/patch_size)*patch_size - H
pad_w = math.ceil(W/patch_size)*patch_size - W

# 1) pad sull’immagine (ha già 3 canali)
img_pad = np.pad(
    img,
    ((0, pad_h), (0, pad_w), (0, 0)),
    mode='constant',
    constant_values=0
)

# 2) pad sulla mask
#    se la tua mask è 2-D, aggiungi l’asse canale:
if mask.ndim == 2:
    mask = mask[..., np.newaxis]

mask_pad = np.pad(
    mask,
    ((0, pad_h), (0, pad_w), (0, 0)),
    mode='constant',
    constant_values=0
)

# 3) estrai patch non sovrapposte
X_patches, Y_patches = [], []
H_p, W_p = img_pad.shape[:2]
for y in range(0, H_p, patch_size):
    for x in range(0, W_p, patch_size):
        xp = img_pad[y:y+patch_size, x:x+patch_size]
        yp = mask_pad[y:y+patch_size, x:x+patch_size]
        X_patches.append(xp)
        Y_patches.append(yp)

X_patches = np.stack(X_patches, axis=0)
Y_patches = np.stack(Y_patches, axis=0)
print(X_patches.shape)
pred = (model.predict(X_patches,verbose=1)>0.5).astype(np.uint8)
print(pred.shape)
N, P = pred.shape[0], pred.shape[1]  
# calcolo griglia su cui ricomporre
n_rows = H_p // P
n_cols = W_p // P

# canvas finale (H_p, W_p), uint8
canvas = np.zeros((n_rows*P, n_cols*P), dtype=np.uint8)

idx = 0
for i in range(n_rows):
    for j in range(n_cols):
        # estrai il singolo patch (rimuovo il canale singleton)
        patch = pred[idx, :, :, 0]      # shape (P, P)
        # assegnalo alla posizione corretta sulla canvas
        canvas[i*P:(i+1)*P, j*P:(j+1)*P] = patch
        idx += 1

fig, axes = plt.subplots(1,3,figsize=(8,7))
axes[0].imshow(img)
axes[1].imshow(mask)
axes[2].imshow(canvas)
plt.show()
"""
# 2) campiona il subset di ID che ti serve
sampled_ids, _ = train_test_split(
    imgIds,
    test_size=0.92,
    random_state=42
)
print(len(imgIds), len(sampled_ids))

# 3) carica i metadata per quei soli ID
img_info = coco.loadImgs(sampled_ids)
print(len(img_info))
# 3) Prepara due dict utili:
#    - image_dims[image_id] = (height, width)
#    - image_files[image_id] = file_name
image_dims  = {img['id']: (img['height'], img['width']) for img in img_info}
image_files = {img['id']: img['file_name']               for img in img_info}

sum_h = sum(img['height'] for img in img_info)
sum_w = sum(img['width']  for img in img_info)

print(sum_h,sum_w)
avg_h = sum_h/len(image_dims)
avg_w = sum_w/len(image_dims)
DIV= 16
IMG_HEIGHT = int(math.ceil(avg_h / DIV) * DIV)
IMG_WIDTH  = int(math.ceil(avg_w / DIV) * DIV)
X, Y = [], []
for info in tqdm(img_info, desc="Carico immagini e maschere"):
    img_id    = info['id']
    file_name = info['file_name']
    # —————— A) carica e ridimensiona l'immagine ——————
    img_path = os.path.join(base, 'images', 'train2017', file_name)
    img      = cv2.imread(img_path)[:, :, ::-1]  # BGR → RGB
    img_res  = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X.append(img_res.astype(np.float32) / 255.)

    # —————— B) costruisci la maschera cumulativa ——————
    mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
    annIds = coco.getAnnIds(imgIds=[img_id], catIds=catIds, iscrowd=None)
    for ann in coco.loadAnns(annIds):
        mask |= coco.annToMask(ann).astype(np.uint8)
    # ridimensiona la maschera alla stessa shape di img_res
    mask_res = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Y.append(mask_res.astype(np.uint8)[..., None])

# 6) Trasforma in array: ora X[i] ↔ Y[i]
X = np.stack(X)   # shape (N, IMG_H, IMG_W, 3)
Y = np.stack(Y)   # shape (N, IMG_H, IMG_W, 1)

print("X.shape =", X.shape)
print("Y.shape =", Y.shape)
idx = np.random.randint(X.shape[0])
fig, axs = plt.subplots(2,1,figsize=(5,10))
axs[0].imshow(X[idx])
axs[1].imshow(Y[idx][...,0])
plt.show()
#controllare immagini ci sono alcune che in cui c'è una macchina ma la bm non è della macchina ma di altri soggetti
patch_size = 256
X_patches = []
Y_patches = []
for img, mask in zip(X, Y):
        H, W = img.shape[:2]
        # calcola padding necessario per rendere H, W multipli di patch_size
        pad_h = (math.ceil(H/patch_size)*patch_size - H)
        pad_w = (math.ceil(W/patch_size)*patch_size - W)
        
        # pad images e mask: img è (H,W,3), mask (H,W,1)
        img_pad  = np.pad(img,  ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
        mask_pad = np.pad(mask, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
        
        H_p, W_p = img_pad.shape[:2]
        # estrai patch non sovrapposte
        for y in range(0, H_p, patch_size):
            for x in range(0, W_p, patch_size):
                xp = img_pad [y:y+patch_size, x:x+patch_size]
                yp = mask_pad[y:y+patch_size, x:x+patch_size]
                # opzionale: scarta le patch bianche (tutte zero) se vuoi
                # if yp.sum() == 0: continue
                X_patches.append(xp)
                Y_patches.append(yp)
                
X_patches = np.stack(X_patches, axis=0)
Y_patches = np.stack(Y_patches, axis=0)

# 6) Split in train/test
X_train, X_test, Y_train, Y_test = train_test_split(
    X_patches, Y_patches, test_size=0.1, random_state=42
)

model, history= train(X_train, Y_train, X_test, Y_test,epochs=5)
    
threshold = [0.3, 0.4, 0.5,0.6]
Ptrain, Pval, Ptest = predict_masks(model=model,X_train=X_train, X_test=X_test, thresholds=threshold)
show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=threshold)
show_threshold_pairs_test(X_test,Y_test,Ptest,thresholds=threshold)
"""