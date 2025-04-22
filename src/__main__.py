from .train import train
from .data_loader import load_folds
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from .predict import predict_masks
from .utils.visualization import show_threshold_pairs_test

if __name__=="__main__":
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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=SEED)
    
    model, history= train(X_train, Y_train, X_test, Y_test)
    
    threshold = [0.3, 0.4, 0.5,0.6]
    Ptrain, Pval, Ptest = predict_masks(model=model,X_train=X_train, X_test=X_test, thresholds=threshold)

    show_threshold_pairs_test(X_train,Y_train,Ptrain,thresholds=threshold)
    show_threshold_pairs_test(X_test,Y_test,Ptest,thresholds=threshold)

#loss: 0.5394 - mean_io_u: 0.0920 -> loss: 0.2117 - mean_io_u: 0.0929 - val_loss: 0.2345 - val_mean_io_u: 0.0913  25 esima epoca
#Test loss: 0.2372 — Test MeanIoU: 0.0903 