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
    types_list = []
    N = 13

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
    
    X = np.concatenate(imgs_list, axis=0) 
    X = X + 15
    X = X/255
    X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])
    # e ricrea un canale singleton (se il tuo modello accetta input (H,W,1))
    X = X_gray[..., np.newaxis].astype(np.float32)
    
    # adesso X ha shape (N,H,W,1)
    Xf = X.astype(np.float32)
    print(f"Xf: {Xf.shape}")
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

    #plt.imshow(Xc[0,:,:,:])
    #plt.show()
    Y = np.concatenate(masks_list, axis=0)

    checkp_base = os.path.join( 'models')
    check = os.path.join(checkp_base, 'checkpoints', 'model_grayscale_v3.keras')

    paper = load_model(checkpoint_path=check)
    

    X_train, X_test, Y_train, Y_test = train_test_split(
        Xc, Y, test_size=0.1, random_state=SEED)
    X_train_rgb = np.repeat(X_train, 3, axis=-1)
    thresholds = [0.3, 0.4, 0.5,0.6]

    out = paper.predict(X_train_rgb)

    pred_min = []
    pred = []
    predB = []
    
    print(np.sum(out[0,:,:,0]),np.sum(out[0,:,:,1]),np.sum(out[0,:,:,2]))


    cell_prob1 = np.maximum(out[0,:,:,0], out[0,:,:,1])
    bg_prob1   = out[0,:,:,1]

# normalizza
    total1 = cell_prob1 + bg_prob1
    cell_prob1 /= total1
    bg_prob1   /= total1

    binary_mask2 = (cell_prob1 > bg_prob1).astype(np.uint8)

#    for i in range(output.shape[0]): 
#        mask = (output[i,:,:,0]+output[i,:,:,2]>output[i,:,:,1])
#        B = (output[i,:,:,0]>output[i,:,:,2]) & (output[i,:,:,0]>output[i,:,:,1])
#        Border = (output[i,:,:,1]>output[i,:,:,0]) & (output[i,:,:,1]>output[i,:,:,2])
#        bin = mask.astype(np.uint8)
#        BM = B.astype(np.uint8)
#        Bord = Border.astype(np.uint8)
#        pred_min.append(bin)
#        pred.append(BM)
#        predB.append(Bord)

import cv2
for i in range(X.shape[0]):
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    axs[0,0].imshow(X_train_rgb[i])
    
    thresh = (Y_train[i] > 0).astype(np.uint8) * 255
    thresh_inv = 255 - thresh
    # find contours
    #label_img = img.copy()
    contour_img1 = Y_train[i].copy()
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    index = 1
    isolated_count = 0
    cluster_count = 0
    contour_img1 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for cntr in contours:
        if len(cntr) < 3:
        # non c'è abbastanza geometria per un hull
            continue
        area = cv2.contourArea(cntr)
        convex_hull = cv2.convexHull(cntr)
        convex_hull_area = cv2.contourArea(convex_hull)
        ratio = area / convex_hull_area

        if ratio < 0.91:
            # cluster contours in red
            cv2.drawContours(contour_img1, [cntr], 0, (0,0,255), 2)
            cluster_count = cluster_count + 1
        else:
            # isolated contours in green
            cv2.drawContours(contour_img1, [cntr], 0, (0,255,0), 2)
            isolated_count = isolated_count + 1
        index = index + 1

    print(f'GT: number_clusters: {cluster_count} -- number_isolated:{isolated_count}')
    axs[0,1].imshow(contour_img1, cmap='gray')
    axs[0,1].set_title("GT")

    cell_prob1 = np.maximum(out[i,:,:,0], out[i,:,:,1])
    bg_prob1   = out[i,:,:,1]

# normalizza
    total1 = cell_prob1 + bg_prob1
    cell_prob1 /= total1
    bg_prob1   /= total1

    binary_mask2 = (cell_prob1 > bg_prob1).astype(np.uint8)
    
    binary_mask = binary_mask2[1:-1,1:-1]
    # threshold to binary
    thresh = (binary_mask > 0).astype(np.uint8) * 255
    thresh_inv = 255 - thresh
    # find contours
    #label_img = img.copy()
    contour_img = binary_mask.copy()
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    index = 1
    isolated_count = 0
    cluster_count = 0
    contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for cntr in contours:
        if len(cntr) < 3:
        # non c'è abbastanza geometria per un hull
            continue
        area = cv2.contourArea(cntr)
        convex_hull = cv2.convexHull(cntr)
        convex_hull_area = cv2.contourArea(convex_hull)
        ratio = area / convex_hull_area

        if ratio < 0.91:
            # cluster contours in red
            cv2.drawContours(contour_img, [cntr], 0, (0,0,255), 2)
            cluster_count = cluster_count + 1
        else:
            # isolated contours in green
            cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
            isolated_count = isolated_count + 1
        index = index + 1

    print(f'Prediction: number_clusters: {cluster_count} -- number_isolated:{isolated_count}')
    axs[1,0].imshow(binary_mask2, cmap='gray')
    axs[1,0].set_title("Grayscale")
    axs[1,1].imshow(contour_img,  interpolation='nearest')
    axs[1,1].set_title("CV2")
    plt.show()




    


