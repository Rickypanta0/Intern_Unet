# src/model.py
import os
os.environ["KERAS_BACKEND"] = "torch"  # <<— fondamentale: prima di importare keras

import keras
from keras import layers, ops
from keras.utils import register_keras_serializable

# (opzionale) torch per le loss custom
import torch
import torch.nn.functional as F
"""
Definizione dell'architettura U-Net e compilazione del modello mantenendo lo stile esplicito originale.
"""
#import tensorflow as tf
#from keras.metrics import MeanIoU
##from src.losses import csca_binary_loss, bce_dice_loss,Lp,binary_iou,hover_mse_grad_loss,hover_loss_fixed
#from src.losses import bce_dice_loss,hover_loss_fixed,hovernet_hv_loss_tf
#from tensorflow.keras.utils import register_keras_serializable
"""
@register_keras_serializable()
class CellDice(tf.keras.metrics.Metric):
    #F1 (o Dice) su maschere cella = body ∪ border (canali 0 e 2).
    def __init__(self, name="cell_dice", smooth=1e-6, **kw):
        super().__init__(name=name, **kw)
        self.smooth = smooth
        self.intersection = self.add_weight(name="inter", initializer="zeros")
        self.union        = self.add_weight(name="union", initializer="zeros")
    def _bin_mask(self, y):
        body, bg, border = tf.unstack(y, axis=-1)
        return tf.cast(body + border > bg, tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_b = self._bin_mask(y_true)           # bool
        y_pred_b = self._bin_mask(y_pred)
        inter = tf.reduce_sum(y_true_b * y_pred_b)
        union = tf.reduce_sum(y_true_b) + tf.reduce_sum(y_pred_b)
        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)
"""
@register_keras_serializable()
class CellDice(keras.metrics.Metric):
    """Dice su maschera cella = body ∪ border (vs background)."""
    def __init__(self, name="cell_dice", smooth=1e-6, **kw):
        super().__init__(name=name, **kw)
        self.smooth = float(smooth)
        self.intersection = self.add_weight(name="inter", initializer="zeros", dtype="float32")
        self.union        = self.add_weight(name="union", initializer="zeros", dtype="float32")

    def _bin_mask(self, y):  # y: [...,3] = [body, bg, border]
        body   = y[..., 0]
        bg     = y[..., 1]
        border = y[..., 2]
        return ops.cast((body + border) > bg, "float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_b = self._bin_mask(y_true)
        y_pred_b = self._bin_mask(y_pred)
        inter = ops.sum(y_true_b * y_pred_b)
        union = ops.sum(y_true_b) + ops.sum(y_pred_b)
        # assegna valori Python/ops ai pesi
        self.intersection.assign_add(ops.cast(inter, "float32"))
        self.union.assign_add(ops.cast(union, "float32"))

    def result(self):
        s = ops.cast(self.smooth, "float32")
        return (2.0 * self.intersection + s) / (self.union + s)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

def get_model_paper(input_shape=(256, 256, 3)):
    inputs = keras.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(inputs)
    c1 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(c1)
    c1 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(p1)
    c2 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(c2)
    c2 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(p2)
    c3 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(c3)
    c3 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    c5 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(p3)
    c5 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = layers.Activation("relu")(c5)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(c5)
    c5 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = layers.Activation("relu")(c5)
    p5 = layers.MaxPooling2D(2)(c5)

    c6 = layers.Conv2D(512, 3, kernel_initializer="he_normal", padding="same")(p5)
    c6 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = layers.Activation("relu")(c6)
    c6 = layers.Dropout(0.3)(c6)
    c6 = layers.Conv2D(512, 3, kernel_initializer="he_normal", padding="same")(c6)
    c6 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = layers.Activation("relu")(c6)
    u1 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c6)

    # Decoder
    u6 = layers.Concatenate()([u1, c5])
    c7 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(u6)
    c7 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = layers.Activation("relu")(c7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, 3, kernel_initializer="he_normal", padding="same")(c7)
    c7 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = layers.Activation("relu")(c7)
    u2 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c7)

    u7 = layers.Concatenate()([u2, c3])
    c8 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(u7)
    c8 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = layers.Activation("relu")(c8)
    c8 = layers.Dropout(0.2)(c8)
    c8 = layers.Conv2D(128, 3, kernel_initializer="he_normal", padding="same")(c8)
    c8 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = layers.Activation("relu")(c8)
    u3 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c8)

    u8 = layers.Concatenate()([u3, c2])
    c9 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(u8)
    c9 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = layers.Activation("relu")(c9)
    c9 = layers.Dropout(0.05)(c9)
    c9 = layers.Conv2D(64, 3, kernel_initializer="he_normal", padding="same")(c9)
    c9 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = layers.Activation("relu")(c9)
    u4 = layers.UpSampling2D(size=(2,2), interpolation="nearest")(c9)

    u9 = layers.Concatenate()([u4, c1])
    c10 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(u9)
    c10 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = layers.Activation("relu")(c10)
    c10 = layers.Dropout(0.05)(c10)
    c10 = layers.Conv2D(32, 3, kernel_initializer="he_normal", padding="same")(c10)
    c10 = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = layers.Activation("relu")(c10)

    c11 = layers.Dropout(0.1)(c10)
    c11 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c11)
    c11 = layers.Dropout(0.05)(c11)
    c11 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c11)

    c12 = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c11)
    c12 = layers.Dropout(0.05)(c12)
    c12 = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c12)
    c12 = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c12)

    seg_head = layers.Conv2D(3, 1, activation="softmax", name="seg_head")(c12)

    c12_hv = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c11)
    c12_hv = layers.Dropout(0.05)(c12_hv)
    c12_hv = layers.Conv2D(16, 3, kernel_initializer="he_normal", padding="same")(c12_hv)
    c12_hv = layers.Conv2D(16, 1, kernel_initializer="he_normal", padding="same")(c12_hv)
    hv_head = layers.Conv2D(2, 1, activation="linear", name="hv_head")(c12_hv)

    model = keras.Model(inputs=inputs, outputs={"seg_head": seg_head, "hv_head": hv_head})
    return model
"""
def get_model(input_shape=(256, 256, 3), learning_rate=1e-4):
    #Costruisce e compila il modello U-Net con architettura hardcoded.
#
    #Args:
    #    input_shape: tupla shape dell'input (H, W, C)
    #    learning_rate: learning rate per Adam
    #Returns:
    #    modello compilato (tf.keras.Model)
    
    inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=Lp,
                  metrics=[binary_iou])
    return model
class OneHotMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true, y_pred: one-hot [..., num_classes]
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        # poi chiamo la classe base
        return super().update_state(y_true, y_pred, sample_weight)

def get_model_paper(input_shape=(256, 256, 3)):

    #Costruisce e compila il modello U-Net con architettura hardcoded.
#
    #Args:
    #    input_shape: tupla shape dell'input (H, W, C)
    #    learning_rate: learning rate per Adam
    #Returns:
    #    modello compilato (tf.keras.Model)
#
    #inputs = tf.keras.Input(input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
    c5 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)

    c6 = tf.keras.layers.Conv2D(512, (3,3), kernel_initializer='he_normal', padding='same')(p5)
    c6 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    c6 = tf.keras.layers.Dropout(0.3)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3,3), kernel_initializer='he_normal', padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    u1 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(c6)

    # Decoder
    #u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u1, c5])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
    c7 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    u2 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(c7)

    #u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u2,c3])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
    c8 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    u3 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(c8)

    #u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u3,c2])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
    c9 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    c9 = tf.keras.layers.Dropout(0.05)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    u4 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(c9)

    #u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u4, c1])
    c10 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
    c10 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = tf.keras.layers.Activation('relu')(c10)
    c10 = tf.keras.layers.Dropout(0.05)(c10)
    c10 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c10)
    c10 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-3)(c10)
    c10 = tf.keras.layers.Activation('relu')(c10)

    c11 = tf.keras.layers.Dropout(0.1)(c10)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c11)
    c11 = tf.keras.layers.Dropout(0.05)(c11)
    c11 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c11)

    c12 = tf.keras.layers.Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(c11)
    c12 = tf.keras.layers.Dropout(0.05)(c12)
    c12 = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c12)
    c12 = tf.keras.layers.Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(c12)
    
    seg_head = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), activation='softmax', name='seg_head')(c12)

    c12_hv = tf.keras.layers.Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(c11)
    c12_hv = tf.keras.layers.Dropout(0.05)(c12_hv)
    c12_hv = tf.keras.layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c12_hv)
    c12_hv = tf.keras.layers.Conv2D(16, (1, 1), kernel_initializer='he_normal', padding='same')(c12_hv)
    hv_head = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), activation='linear', name='hv_head')(c12_hv)

    #hv_head = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), activation='tanh', name='hv_head')(c12_hv)

    model = tf.keras.Model(inputs=inputs, outputs={
    'seg_head': seg_head,
    'hv_head': hv_head
    })
    
    return model
"""
"""
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import UpSampling2D, concatenate, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model


import tensorflow as tf
from tensorflow.keras.layers import (Input, UpSampling2D, Concatenate, Conv2D,
                                     BatchNormalization, Activation, Dropout)
from tensorflow.keras.models import Model

# blocchetto conv-bn-relu (+ opzionale dropout)
def CBR(x, f, k=3, dr=None):
    x = Conv2D(f, k, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_unet_resnet50(input_shape=(256,256,3), freeze_encoder=True):
    inp = Input(input_shape, name='image_in')

    # Preprocess sicuro (se lo usi)
    # x_in = ResNetPreprocess(name="resnet_preproc")(inp)
    # base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x_in)

    # Se preferisci preprocess fuori, usa direttamente inp:
    base = tf.keras.applications.ResNet50(include_top=False, 
                                          weights='imagenet',
                                          input_tensor=inp
                                          )

    if freeze_encoder:
        for L in base.layers: 
            L.trainable = False
    
    # Skip dal ResNet
    s1 = base.get_layer('conv1_relu').output          # 128x128
    s2 = base.get_layer('conv2_block3_out').output    # 64x64
    s3 = base.get_layer('conv3_block4_out').output    # 32x32
    s4 = base.get_layer('conv4_block6_out').output    # 16x16
    b  = base.get_layer('conv5_block3_out').output    # 8x8 (bottleneck)

    # Decoder (solo upsampling + concat con skip ResNet) — NIENTE MaxPooling qui
    x = UpSampling2D()(b);  x = Concatenate()([x, s4]); x = CBR(x, 256)
    x = Dropout(0.2)(x); x = CBR(x, 256)
    
    x = UpSampling2D()(x);  x = Concatenate()([x, s3])
    x = CBR(x, 128); x = Dropout(0.2)(x); x = CBR(x, 128)
    
    x = UpSampling2D()(x);  x = Concatenate()([x, s2])
    x = CBR(x,  64); x = Dropout(0.05)(x); x = CBR(x,  64)

    x = UpSampling2D()(x);  x = Concatenate()([x, s1])
    x = CBR(x,  32); x = Dropout(0.05)(x); x = CBR(x,  32)
    
    x = UpSampling2D()(x);  x = CBR(x, 32)  # fino a 256×256

    # Due head
    c11 = Dropout(0.1)(x); c11 = CBR(c11, 16); c11 = Dropout(0.05)(c11); c11 = CBR(c11, 16)
    c12 = Conv2D(16, 1, padding='same', kernel_initializer='he_normal')(c11)
    c12 = Dropout(0.05)(c12); c12 = CBR(c12, 16); c12 = Conv2D(16, 1, padding='same', kernel_initializer='he_normal')(c12)

    seg_head = Conv2D(3, 1, activation='softmax', name='seg_head')(c12)

    c12_hv = Conv2D(16, 1, padding='same', kernel_initializer='he_normal')(c11)
    c12_hv = Dropout(0.05)(c12_hv); c12_hv = CBR(c12_hv, 16)
    hv_head = tf.keras.layers.Conv2D(2, 1, activation='tanh', name='hv_head')(c12_hv)

    return Model(inp, {'seg_head': seg_head, 'hv_head': hv_head}), base


if __name__ == "__main__":
    # Test e summary
    unet_model = get_model()
    unet_model.summary()
"""