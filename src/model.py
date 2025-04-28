# src/model.py
"""
Definizione dell'architettura U-Net e compilazione del modello mantenendo lo stile esplicito originale.
"""
import tensorflow as tf
from keras.metrics import MeanIoU
from src.losses import csca_binary_loss, bce_dice_loss


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
                  loss=bce_dice_loss,
                  metrics=[MeanIoU(num_classes=2)])
    return model

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

    model = tf.keras.Model(inputs=inputs, outputs=seg_head)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = bce_dice_loss
# o una combo con dice multiclasse

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.MeanIoU(num_classes=3)]
    )
    
    return model

if __name__ == "__main__":
    # Test e summary
    unet_model = get_model()
    unet_model.summary()