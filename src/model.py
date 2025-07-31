# src/model.py
"""
Definizione dell'architettura U-Net e compilazione del modello mantenendo lo stile esplicito originale.
"""
import tensorflow as tf
from keras.metrics import MeanIoU
#from src.losses import csca_binary_loss, bce_dice_loss,Lp,binary_iou,hover_mse_grad_loss,hover_loss_fixed
from src.losses import bce_dice_loss,hover_loss_fixed
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class CellDice(tf.keras.metrics.Metric):
    """F1 (o Dice) su maschere cella = body ∪ border (canali 0 e 2)."""
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

def get_model(input_shape=(256, 256, 3), learning_rate=1e-4):
    """
    Costruisce e compila il modello U-Net con architettura hardcoded.

    Args:
        input_shape: tupla shape dell'input (H, W, C)
        learning_rate: learning rate per Adam
    Returns:
        modello compilato (tf.keras.Model)
    """
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
def get_model_paper(input_shape=(256, 256, 3), learning_rate=1e-3):
    """
    Costruisce e compila il modello U-Net con architettura hardcoded.

    Args:
        input_shape: tupla shape dell'input (H, W, C)
        learning_rate: learning rate per Adam
    Returns:
        modello compilato (tf.keras.Model)
    """
    inputs = tf.keras.Input(input_shape)

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

    model = tf.keras.Model(inputs=inputs, outputs={
    'seg_head': seg_head,
    'hv_head': hv_head
    })

    #import tensorflow_addons as tfa
    #dice_metric = tfa.metrics.F1Score(
    #    num_classes=1,                  # binario
    #    average='micro',                # restituisce scalare, non rank‑0
    #    threshold=0.5
    #)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=25,   # 10 epoche
        t_mul=2.0,                                # lunghezza fase raddoppia
        m_mul=0.8,                                # ampiezza si riduce
        alpha=1e-5 / 1e-3                         # min_lr
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
    optimizer=optimizer,
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hover_loss_fixed  # <--- funzione non parametrica
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 2.0
    },
    metrics={'seg_head': [CellDice()],
             'hv_head' : [tf.keras.metrics.MeanSquaredError()]}
)
    
    return model

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import UpSampling2D, concatenate, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model



def model_paper_hover(input_shape=(256,256,3)):
    # 1) Input
    inputs = tf.keras.Input(shape=input_shape)

    inputs = tf.keras.Input(input_shape)

    # ---- Encoder + Bottleneck ----
    def conv_block(x, filters, dropout_rate):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    c1 = conv_block(inputs, 32, 0.1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64, 0.1)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128, 0.2)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c5 = conv_block(p3, 256, 0.3)
    p5 = tf.keras.layers.MaxPooling2D()(c5)

    c6 = conv_block(p5, 512, 0.3)
    u1 = tf.keras.layers.UpSampling2D()(c6)

    # ---- Decoder shared ----
    def up_block(u, skip, filters, dropout_rate):
        x = tf.keras.layers.concatenate([u, skip])
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return tf.keras.layers.UpSampling2D()(x)

    u2 = up_block(u1, c5, 256, 0.2)
    u3 = up_block(u2, c3, 128, 0.2)
    u4 = up_block(u3, c2, 64, 0.05)
    shared = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(u4)
    # ---- HoVerNet-style seg_head ----
    x = shared
    for filters in [64, 64, 32]:
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    seg_head = tf.keras.layers.Conv2D(3, 1, activation='softmax', name='seg_head')(x)

    # ---- HoVerNet-style hv_head ----
    x = shared
    for filters in [64, 64, 32]:
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    hv_head = tf.keras.layers.Conv2D(2, 1, activation='linear', name='hv_head')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs={'seg_head': seg_head, 'hv_head': hv_head})

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=25,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-5 / 1e-3
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss={
            'seg_head': bce_dice_loss,
            'hv_head': hover_loss_fixed
        },
        loss_weights={
            'seg_head': 1.0,
            'hv_head': 1.0
        },
        metrics={
            'seg_head': [CellDice()],
            'hv_head': [tf.keras.metrics.MeanSquaredError()]
        }
    )

    return model

if __name__ == "__main__":
    # Test e summary
    unet_model = get_model()
    unet_model.summary()