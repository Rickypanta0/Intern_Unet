from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
from .predict import load_model,predict_masks
from .data_loader import load_folds
from sklearn.model_selection import train_test_split
from .utils.visualization import show_threshold_pairs_test
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SEED = 42
    np.random.seed = SEED
    base = os.path.join( 'data','raw','val')
    folds = [
        (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
         os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy'))
    ]
    
    imgs_list = []
    masks_list = []
    N = 12

    for i, (img_path, mask_path) in enumerate(folds):
        print(f"i: {i}")
        fold_seed = SEED + i
        rng = np.random.default_rng(fold_seed)

        imgs = np.load(img_path, mmap_mode='r')
        masks = np.load(mask_path, mmap_mode='r')

        assert len(imgs) == len(masks), "Mismatch tra immagini e maschere"
        assert len(imgs) >= N, f"Fold {i} ha solo {len(imgs)} immagini"

        idxs = rng.choice(len(imgs), size=N, replace=False)

        imgs_selected = imgs[idxs]
        masks_selected = masks[idxs]

        imgs_list.append(imgs_selected)
        masks_list.append(masks_selected)

        print(f"imgs_list: {len(imgs_list)}, masks_list: {len(masks_list)}")
    
    X = np.concatenate(imgs_list, axis=0) / 255 

    Xf = X.astype(np.float32)

    # 1) calcola la media per canale
    # se X ha shape (N,H,W,C):
    mean = Xf.mean(axis=(0,1,2), keepdims=True)
    # se X è singola immagine (H,W,C), usa mean = Xf.mean(axis=(0,1), keepdims=True)

    alpha = 1.4  # >1 aumenta il contrasto

    # 2) formula del contrast stretch
    Xc = mean + alpha * (Xf - mean)

    # 3) ritaglia al range originale
    if X.dtype == np.uint8:
        Xc = np.clip(Xc, 0, 255).astype(np.uint8)
    else:
        Xc = np.clip(Xc, 0.0, 1.0)

    Y = np.concatenate(masks_list, axis=0)
    print(len(X),len(Y))

    #fig, axs = plt.subplots(2,1)
    #axs[0].imshow(X[1])
    #axs[1].imshow(Y[1])
    #plt.show()

    checkp_base = os.path.join( 'models')
    checkp_folds = os.path.join(checkp_base, 'checkpoints', 'model_paper_v3.keras')
    check = os.path.join(checkp_base, 'checkpoints', 'model_backbone.keras')
    #X, Y = load_folds(folds=folds)
    final = load_model(checkpoint_path=checkp_folds)
    paper = load_model(checkpoint_path=check)
    #Y = np.expand_dims(Y, axis=-1)
    


    X_train, X_test, Y_train, Y_test = train_test_split(
        Xc, Y, test_size=0.1, random_state=SEED)
    
    thresholds = [0.3, 0.4, 0.5,0.6]
    output = final.predict(X_train)
    out = paper.predict(X_train)

    pred_min = []
    pred = []
    predB = []
    print(np.sum(output[0,:,:,0]),np.sum(output[0,:,:,1]),np.sum(output[0,:,:,2]))

    cell_prob = np.maximum(output[0,:,:,0], output[0,:,:,1])
    bg_prob   = output[0,:,:,1]

# normalizza
    total = cell_prob + bg_prob
    cell_prob /= total
    bg_prob   /= total

    binary_mask = (cell_prob > bg_prob).astype(np.uint8)

    cell_prob1 = np.maximum(out[0,:,:,0], out[0,:,:,1])
    bg_prob1   = out[0,:,:,1]

# normalizza
    total1 = cell_prob1 + bg_prob1
    cell_prob1 /= total1
    bg_prob1   /= total1

    binary_mask2 = (cell_prob1 > bg_prob1).astype(np.uint8)

    for i in range(output.shape[0]): 
        mask = (output[i,:,:,0]+output[i,:,:,2]>output[i,:,:,1])
        B = (output[i,:,:,0]>output[i,:,:,2]) & (output[i,:,:,0]>output[i,:,:,1])
        Border = (output[i,:,:,1]>output[i,:,:,0]) & (output[i,:,:,1]>output[i,:,:,2])
        bin = mask.astype(np.uint8)
        BM = B.astype(np.uint8)
        Bord = Border.astype(np.uint8)
        pred_min.append(bin)
        pred.append(BM)
        predB.append(Bord)

for i in range(X.shape[0]):
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    axs[0,0].imshow(X_train[i])
    axs[0,1].imshow(Y_train[i], cmap='gray')
    axs[0,1].set_title("GT")

    cell_prob = np.maximum(output[i,:,:,0], output[i,:,:,1])
    bg_prob   = output[i,:,:,1]

# normalizza
    total = cell_prob + bg_prob
    cell_prob /= total
    bg_prob   /= total

    binary_mask = (cell_prob > bg_prob).astype(np.uint8)

    cell_prob1 = np.maximum(out[i,:,:,0], out[i,:,:,1])
    bg_prob1   = out[i,:,:,1]

# normalizza
    total1 = cell_prob1 + bg_prob1
    cell_prob1 /= total1
    bg_prob1   /= total1

    binary_mask2 = (cell_prob1 > bg_prob1).astype(np.uint8)

    axs[1,0].imshow(binary_mask2, cmap='gray')
    axs[1,0].set_title("vn")
    axs[1,1].imshow(binary_mask,cmap='gray')
    axs[1,1].set_title("v2")
    plt.show()

    
    #morphological closing
    #show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=thresholds)
    #show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=thresholds)


    


