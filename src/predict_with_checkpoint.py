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
        (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'))
    ]
    
    imgs_list = []
    masks_list = []
    N = 30

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

    fig, axs = plt.subplots(2,1)
    axs[0].imshow(X[1])
    axs[1].imshow(Y[1])
    plt.show()

    checkp_base = os.path.join( 'models')
    checkp_folds = [os.path.join(checkp_base, 'final', 'unet_final.keras'),
                    os.path.join(checkp_base, 'checkpoints','model_for_nuclei.keras')]
    
    #X, Y = load_folds(folds=folds)
    final = load_model(checkpoint_path=checkp_folds[0])
    checkpoints = load_model(checkpoint_path=checkp_folds[1])

    Y = np.expand_dims(Y, axis=-1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=SEED)
    thresholds = [0.3, 0.4, 0.5,0.6]
    Ptrain, Pval, Ptest = predict_masks(model=final,X_train=X_train,X_test=X_test)

    show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=thresholds)


    


