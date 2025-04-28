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
    base = os.path.join( 'data','raw')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'fold1', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'fold1', 'binary_masks.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'fold2', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'fold2', 'binary_masks.npy'))
    ]

    X, Y = load_folds(folds, normalize_images=True)
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
    Y_ = np.stack(Y_,axis=0)
    #from tensorflow.keras.datasets import mnist
    #(x_train, _), _ = mnist.load_data()
    #X = (x_train > 127).astype('uint8')[...,None]  # binarizzo
    #X = np.pad(X, pad_width=((0,0),(2,2),(2,2),(0,0)), mode='constant', constant_values=0)
    #Y = X.copy()
    #X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    #X_val,   X_test, Y_val,   Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=SEED)
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y_, test_size=0.1, random_state=SEED)
    #Y_train_oh = tf.keras.utils.to_categorical(Y_train, num_classes=3)
# ora ha shape (B,256,256,3)
    #Y_test_oh = tf.keras.utils.to_categorical(Y_test,   num_classes=3)
    model, history= train(X_train, Y_train, X_test, Y_test,epochs=50,batch_size=8)
    

    #threshold = [0.3, 0.4, 0.5,0.6]
    #Ptrain, Pval, Ptest = predict_masks(model=model,X_train=X_train, X_test=X_test, thresholds=threshold)
##
    #show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=threshold)
    #show_threshold_pairs_test(X_test,Y_test,Ptest,thresholds=threshold)

#loss: 0.5394 - mean_io_u: 0.0920 -> loss: 0.2117 - mean_io_u: 0.0929 - val_loss: 0.2345 - val_mean_io_u: 0.0913  25 esima epoca
#Test loss: 0.2372 — Test MeanIoU: 0.0903 


#loss: 0.1355 - mean_io_u: 0.4884 -> loss: 1.3205e-09 - mean_io_u: 1.0000 