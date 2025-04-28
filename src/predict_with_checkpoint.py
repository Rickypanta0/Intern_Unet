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
    base = os.path.join( 'data','raw')
    folds = [
        (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
         os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy'))
    ]
    
    imgs_list = []
    masks_list = []
    N = 15

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
    Y = np.concatenate(masks_list, axis=0)
    print(len(X),len(Y))

    #fig, axs = plt.subplots(2,1)
    #axs[0].imshow(X[1])
    #axs[1].imshow(Y[1])
    #plt.show()

    checkp_base = os.path.join( 'models')
    checkp_folds = os.path.join(checkp_base, 'checkpoints', 'model_for_nuclei_paper_3heads_v2.keras')
    check = os.path.join(checkp_base, 'checkpoints', 'model_for_nuclei_paper.keras')
    #X, Y = load_folds(folds=folds)
    final = load_model(checkpoint_path=checkp_folds)
    paper = load_model(checkpoint_path=check)
    #Y = np.expand_dims(Y, axis=-1)
    


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=SEED)
    
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

    fig, axs = plt.subplots(2,3,figsize=(8,8))
    axs[0,0].imshow(binary_mask,cmap='gray')
    axs[0,1].imshow(Y_train[0], cmap='gray')
    axs[0,1].set_title("GT")
    axs[0,2].imshow(output[0,:,:,0], cmap='gray')
    axs[1,0].imshow(output[0,:,:,1], cmap='gray')
    axs[1,1].imshow(output[0,:,:,2], cmap='gray')

    out_ = (out[:,:,:,:]>0.5).astype(np.uint8)
    axs[1,2].imshow(pred_min[0],cmap='gray')
    axs[1,2].set_title("No 3 heads")
    plt.show()

    
    #morphological closing
    #show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=thresholds)
    #show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=thresholds)


    


