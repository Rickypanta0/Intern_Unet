import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

base = os.path.join('data','raw')
folds = [os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'), 
         os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy')]

X = np.load(folds[0])/255
Y = np.load(folds[1])
IMG_SIZE = 256
IMG_SIZE_ = 200
img = X[0]
padded = tf.image.resize_with_crop_or_pad(img, IMG_SIZE+6, IMG_SIZE+6)

# 2) Fissiamo un seed iniziale
seed = tf.constant([1234, 5678], dtype=tf.int32)

# 3) Generiamo due seed “figli” diversi
seed1, seed2 = tf.random.experimental.stateless_split(seed, num=2)

# 4) Due crop diversi con i due seed
crop1 = tf.image.stateless_random_crop(padded, [IMG_SIZE_, IMG_SIZE_, 3], seed=seed1)
crop2 = tf.image.stateless_random_crop(padded, [IMG_SIZE_, IMG_SIZE_, 3], seed=seed2)

zoom_factor = IMG_SIZE_/IMG_SIZE
delta = IMG_SIZE - IMG_SIZE_
pad_top = delta//2
pad_bottom = delta//2
# Visualizziamo originale vs crop1 vs crop2
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(img);   axes[0].set_title('Originale')
axes[1].imshow(crop1); axes[1].set_title('Crop seed1')
axes[2].imshow(crop2); axes[2].set_title('Crop seed2')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()