
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from src.model import get_model_paper, model_paper_hover
import math 
from tensorflow import keras 
from matplotlib.cm import get_cmap
import tensorflow_io as tfio
#from src.test_augm import build_instance_map_valuewise, GenInstanceHV
import cv2

BACKBONE = 'resnet34'

"""
    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         hv_np : np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:

        #img_np: np.ndarray (H,W) o (H,W,1) o (H,W,3), valori [0..1] o [0..255]
        #mask_np: np.ndarray (H,W), valori {0,1}
        
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        # 1) Assicuriamoci di avere (H,W,1) o (H,W,3)
        if img_np.ndim == 2:
            img_np = img_np[...,None]
        # 2) Tensor float32
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)
        C_orig = img_t.shape[-1]
        # 3) Se mono-canale, duplichiamo su 3 canali
        if C_orig == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)
        # adesso img_t.shape == (H,W,3)
        # 4) Pad/center-crop per avere shape statica (IMG_SIZE×IMG_SIZE×3)
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE, IMG_SIZE)

        # 5) Preparo il seed TF
        seed_tf = tf.convert_to_tensor(seed, dtype=tf.int32)

        # 6) Random crop **dal padded**, taglio un quadrato zoom_size×zoom_size×3
        crop = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, 3],
            seed=seed_tf
        )
        # 7) Ridimensiono back a IMG_SIZE×IMG_SIZE
        xz = tf.image.resize(crop, [IMG_SIZE, IMG_SIZE])

        # 8) Stessa cosa per la mask (ha sempre 1 canale)
        mask_t = tf.convert_to_tensor(mask_np[...,None], dtype=tf.float32)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE, IMG_SIZE)
        crop_m = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, 1],
            seed=seed_tf
        )
        yz = tf.image.resize(crop_m, [IMG_SIZE, IMG_SIZE])

        # HV map (H,W,2)
        if hv_np.ndim != 3 or hv_np.shape[-1] != 2:
            raise ValueError(f"hv_np deve avere shape (H,W,2), ma ha {hv_np.shape}")

        # Separazione dei canali x e y
        hv_x = hv_np[..., 0][..., None]  # shape (H,W,1)
        hv_y = hv_np[..., 1][..., None]

        # Tensor conversione
        hv_x_t = tf.convert_to_tensor(hv_x, dtype=tf.float32)
        hv_y_t = tf.convert_to_tensor(hv_y, dtype=tf.float32)

        # Pad + crop + resize per x
        padded_x = tf.image.resize_with_crop_or_pad(hv_x_t, IMG_SIZE, IMG_SIZE)
        crop_x = tf.image.stateless_random_crop(padded_x, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cx = tf.image.resize(crop_x, [IMG_SIZE, IMG_SIZE])

        # Pad + crop + resize per y
        padded_y = tf.image.resize_with_crop_or_pad(hv_y_t, IMG_SIZE, IMG_SIZE)
        crop_y = tf.image.stateless_random_crop(padded_y, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cy = tf.image.resize(crop_y, [IMG_SIZE, IMG_SIZE])

        # 9) Torno a NumPy
        hv_cx_np = hv_cx.numpy().squeeze(-1).astype(np.float32)
        hv_cy_np = hv_cy.numpy().squeeze(-1).astype(np.float32)
        hv_np_f  = np.stack([hv_cx_np, hv_cy_np], axis=-1)

        xz_np = xz.numpy().astype(np.float32)
        yz_np = yz.numpy().astype(np.uint8)

        return xz_np, yz_np, hv_np_f
    """
from skimage.color import rgb2hed, hed2rgb
class DataGenerator(keras.utils.Sequence):
    def __init__(self, folds, list_IDs, batch_size=1, patch_size=None,
                 shuffle=True, augment=True, skip_empty=True):
        self.folds = folds  # lista di triple (img_path, mask_path, hv_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.augment = augment
        self.skip_empty = skip_empty
        self.preprocess_input = BACKBONE
        self.list_IDs = np.asarray(list_IDs)  # global indices
        self.ratio = 2

        self.sample_map = []
        for fold_idx, (img_path, _, _) in enumerate(folds):
            n_samples = np.load(img_path, mmap_mode='r').shape[0]
            for i in range(n_samples):
                self.sample_map.append((fold_idx, i))

        sample_img = np.load(folds[0][0], mmap_mode='r')[0]
        self.default_shape = sample_img.shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.sample_map[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def _make_target_3ch(self, m2):
        struct = np.ones((3, 3), dtype=bool)

        m2_ = 1-m2
        body_ = binary_erosion(m2_, structure=struct).astype(np.uint8)
        
        body = 1-body_

        border = (m2_ - body_).clip(0, 1).astype(np.uint8)
        background = (1 - m2).astype(np.uint8)

        return np.stack([body,background, border], axis=-1)

    def augm(self,
         seed: tuple[int,int],
         img_np: np.ndarray,
         mask_np: np.ndarray,
         hv_np : np.ndarray,
         zoom_size: int,
         IMG_SIZE: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        img_np: np.ndarray (H,W) o (H,W,1) o (H,W,3), valori [0..1] o [0..255]
        mask_np: np.ndarray (H,W), valori {0,1}
        """
        if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        # 1) Assicuriamoci di avere (H,W,1) o (H,W,3)
        if img_np.ndim == 2:
            img_np = img_np[...,None]
        # 2) Tensor float32
        img_t = tf.convert_to_tensor(img_np, dtype=tf.float32)
        C_orig = img_t.shape[-1]
        # 3) Se mono-canale, duplichiamo su 3 canali
        if C_orig == 1:
            img_t = tf.image.grayscale_to_rgb(img_t)
        # adesso img_t.shape == (H,W,3)
        # 4) Pad/center-crop per avere shape statica (IMG_SIZE×IMG_SIZE×3)
        padded = tf.image.resize_with_crop_or_pad(img_t, IMG_SIZE, IMG_SIZE)

        # 5) Preparo il seed TF
        seed_tf = seed

        # 6) Random crop **dal padded**, taglio un quadrato zoom_size×zoom_size×3
        crop = tf.image.stateless_random_crop(
            padded,
            size=[zoom_size, zoom_size, 3],
            seed=seed_tf
        )
        # 7) Ridimensiono back a IMG_SIZE×IMG_SIZE
        xz = tf.image.resize(crop, [IMG_SIZE, IMG_SIZE])

        # 8) Stessa cosa per la mask (ha sempre 1 canale)
        mask_t = tf.convert_to_tensor(mask_np[...,None], dtype=tf.float32)
        padded_m = tf.image.resize_with_crop_or_pad(mask_t, IMG_SIZE, IMG_SIZE)
        crop_m = tf.image.stateless_random_crop(
            padded_m,
            size=[zoom_size, zoom_size, 1],
            seed=seed_tf
        )
        yz = tf.image.resize(crop_m, [IMG_SIZE, IMG_SIZE])

        # HV map (H,W,2)
        if hv_np.ndim != 3 or hv_np.shape[-1] != 2:
            raise ValueError(f"hv_np deve avere shape (H,W,2), ma ha {hv_np.shape}")

        # Separazione dei canali x e y
        hv_x = hv_np[..., 0][..., None]  # shape (H,W,1)
        hv_y = hv_np[..., 1][..., None]

        # Tensor conversione
        hv_x_t = tf.convert_to_tensor(hv_x, dtype=tf.float32)
        hv_y_t = tf.convert_to_tensor(hv_y, dtype=tf.float32)

        # Pad + crop + resize per x
        padded_x = tf.image.resize_with_crop_or_pad(hv_x_t, IMG_SIZE, IMG_SIZE)
        crop_x = tf.image.stateless_random_crop(padded_x, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cx = tf.image.resize(crop_x, [IMG_SIZE, IMG_SIZE])

        # Pad + crop + resize per y
        padded_y = tf.image.resize_with_crop_or_pad(hv_y_t, IMG_SIZE, IMG_SIZE)
        crop_y = tf.image.stateless_random_crop(padded_y, [zoom_size, zoom_size, 1], seed=seed_tf)
        hv_cy = tf.image.resize(crop_y, [IMG_SIZE, IMG_SIZE])

        # 9) Torno a NumPy
        hv_cx_np = hv_cx.numpy().squeeze(-1).astype(np.float32)
        hv_cy_np = hv_cy.numpy().squeeze(-1).astype(np.float32)
        hv_np_f  = np.stack([hv_cx_np, hv_cy_np], axis=-1)

        xz_np = xz.numpy().astype(np.float32)
        yz_np = yz.numpy().astype(np.uint8)
        print("xz_np:", xz_np.shape)
        print("yz_np:", yz_np.shape)
        print("hv_np_f:", hv_np_f.shape)
        yz_np = np.squeeze(yz_np)
        if yz_np.ndim != 2:
            raise ValueError(f"La maschera deve essere 2D dopo lo squeeze, ma ha shape {yz_np.shape}")
        
        return xz_np, yz_np, hv_np_f

    def __data_generation(self, list_temp):
        if self.patch_size is None:
            H, W, _ = self.default_shape
        else:
            H = W = self.patch_size
        C = 3

        N = self.batch_size

        X = np.empty((N, H, W, C), dtype=np.float32)
        Y = np.empty((N, H, W, 3), dtype=np.float32)
        HV = np.empty((N, H, W, 2), dtype=np.float32)

        i=0
        #for fold_idx, local_idx in list_temp:
        #    img_path, mask_path, hv_path = self.folds[fold_idx]

        #    img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
        #    img = (img + 5) / 255.0
        #    img_gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        #    img_gray = img_gray[..., np.newaxis]
        #    img_rgb = np.repeat(img_gray, 3, axis=-1)

        #    mask = np.load(mask_path, mmap_mode='r')[local_idx]
        #    if mask.ndim == 3 and mask.shape[-1] == 1:
        #        mask = np.squeeze(mask, axis=-1)

        #    hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)

        for i, (fold_idx, local_idx) in enumerate(list_temp):
            img_path, mask_path, hv_path = self.folds[fold_idx]

            img = np.load(img_path, mmap_mode='r')[local_idx].astype(np.float32)
            img_rgb = (img + 15) / 255.0
            
            #HESOINE EXTRACTION
            
            #ihc_hed = rgb2hed(img_rgb)
#
            ## Create an RGB image for each of the stains
            #null = np.zeros_like(ihc_hed[:, :, 0])
            #img_rgb = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            
            #BLU CHANNEL
            
            #blu = img_rgb[...,2]
            #blu_enhanced = np.clip(blu + 0.05, 0, 1)
            #cmap = get_cmap('Blues')
            #img_rgb = cmap(blu_enhanced)[:, :, :3] 

            #GRAY SCALE
            #img_gray = np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140])
            #img_gray = img_gray[..., np.newaxis]
            #img_rgb = np.repeat(img_gray, 3, axis=-1)
            #
            mask = np.load(mask_path, mmap_mode='r')[local_idx]
        
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)

            hv = np.load(hv_path, mmap_mode='r')[local_idx].astype(np.float32)
            mask = self._make_target_3ch(mask)

            # Augmentation
            if self.augment:
                if np.random.rand() < 0.5:
                    img_rgb = np.fliplr(img_rgb)
                    mask   = np.fliplr(mask)
                    hv      = np.fliplr(hv)
                if np.random.rand() < 0.5:
                    img_rgb = np.flipud(img_rgb)
                    mask   = np.flipud(mask)
                    hv      = np.flipud(hv)
                k = np.random.randint(4) 
                #print(mask.shape)
                img_rgb = np.rot90(img_rgb, k, axes=(0, 1))
                mask   = np.rot90(mask,   k, axes=(0, 1))
                hv      = np.rot90(hv,      k, axes=(0, 1))

            X[i] = img_rgb
            Y[i] = mask
            HV[i] = hv

        return X, {'seg_head': Y, 'hv_head': HV}
    
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

if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # attiva memory growth **prima** di qualunque allocazione
        tf.config.experimental.set_memory_growth(gpus[0], True)
    SEED = 42
    np.random.seed = SEED
    base = os.path.join( 'data','raw_neoplastic')
    folds = [
        (os.path.join(base, 'Fold 1', 'images', 'images.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'binary_masks.npy'),
         os.path.join(base, 'Fold 1', 'masks', 'distance.npy')),
        (os.path.join(base, 'Fold 2', 'images', 'images.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'binary_masks.npy'),
         os.path.join(base, 'Fold 2', 'masks', 'distance.npy'))
    ]


    num_all = sum(np.load(f[0], mmap_mode='r').shape[0] for f in folds)

    all_IDs = np.arange(num_all)
    # split 90/10
    split = int(0.9 * num_all)
    train_IDs = all_IDs[:split]
    val_IDs   = all_IDs[split:]
    # generatori
    #print(val_IDs)
    train_gen = DataGenerator(folds, train_IDs, batch_size=4, shuffle=True, augment=True)
    val_gen = DataGenerator(folds, val_IDs, batch_size=4, shuffle=False, augment=False)

    monitor_metric = 'val_seg_head_cell_dice'

    #Dynamic Learning rate
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        mode="max",
        verbose=1
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs','fit'),
            histogram_freq=1,
            write_images=True
        )
    checkpoint_path='models/checkpoints/neo/model_BB_RGB_v2.keras'
        # Checkpoint
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=monitor_metric,
            mode="max",
            verbose=1
        )

        # EarlyStopping
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=5,
            restore_best_weights=True
        )
    Xb, Yb = train_gen[0]
    k = min(4, Xb.shape[0])                                 # quante immagini mostrare
    high = max(1, Xb.shape[0] - (k - 1))                    # ultimo indice di partenza valido + 1
    idx = np.random.randint(0, high)    
    
    print(Xb.shape, Yb['seg_head'].shape, Yb['hv_head'].shape)

    hv_batch = Yb['hv_head']
    #print(f"min: {np.min(hv_batch)}, max {np.max(hv_batch)}, mean {np.mean(hv_batch)}")
    """"""
    for i in range(idx,idx + 4):
        fig, axs = plt.subplots(2, 2, figsize=(6,6))
        image = Xb[i]          # (H, W, 3) – RGB image (non usata da GenInstanceHV)
        mask_3ch = Yb['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
        body_mask = mask_3ch[..., 0].astype(np.uint8)  # binaria
        border_mask = mask_3ch[..., 2].astype(np.uint8)
        background = mask_3ch[..., 1].astype(np.uint8)

        C = np.logical_or(body_mask, border_mask).astype(np.uint8)
        axs[0,0].imshow(Xb[i],cmap='gray')                         # immagine normalizzata [0,1]
        axs[0,1].imshow(body_mask, cmap='gray')  # body
        axs[1,0].imshow(background, cmap='gray')  # background
        axs[1,1].imshow(border_mask, cmap='gray')  # border
    
        plt.tight_layout()
        plt.show()
    """
    model = get_model_paper()
    
    #model_ = model_paper_hover()
    history = model.fit(train_gen,
            validation_data=val_gen, 
            epochs=60, 
            callbacks=[
                earlystop_cb,
                checkpoint_cb,
                tensorboard_cb,
            ],
            verbose=1)
    """
    
    #SECONDO ROUND
    
    from src.losses import bce_dice_loss,hover_loss_fixed
    from tensorflow.keras.utils import register_keras_serializable
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=10 * len(train_gen),   # es.: 10 epoche
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-5 / 1e-3
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    best = tf.keras.models.load_model(
        "models/checkpoints/neo/model_BB_RGB.keras",
        custom_objects={"CellDice": CellDice,
                        "bce_dice_loss": bce_dice_loss,
                        "hover_loss_fixed": hover_loss_fixed}
    )

    # 2. Ricompila con loss_weights nuovi e/o LR schedule diverso
    best.compile(
        optimizer=optimizer,
        loss={'seg_head': bce_dice_loss,
              'hv_head' : hover_loss_fixed},
        loss_weights={'seg_head': 0.25, 'hv_head': 5},
        metrics={'seg_head': [CellDice()], 'hv_head': []}
    )

    # 3. Continua l’addestramento
    history = best.fit(train_gen,
                       validation_data=val_gen,
                       epochs=20,             # numero epoche aggiuntive
                       callbacks=[checkpoint_cb, earlystop_cb],
                       verbose=1)
    
    """
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-4,
        first_decay_steps=10 * len(train_gen),   # es.: 10 epoche
        alpha=0.5
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = tf.keras.models.load_model(
        "models/checkpoints/model_RGB_HV_v2.keras",
        custom_objects={"CellDice": CellDice,
                        "bce_dice_loss": bce_dice_loss,
                        "hover_loss_fixed": hover_loss_fixed}
    )
    def ancestors(layer):
        seen = set()
        stack = [layer]
        while stack:
            L = stack.pop()
            if L in seen:
                continue
            seen.add(L)
            # percorri i nodi in ingresso al layer
            for node in getattr(L, "_inbound_nodes", []):
                for t in getattr(node, "input_tensors", []):
                    # ogni tensor porta al layer sorgente tramite _keras_history
                    src = getattr(getattr(t, "_keras_history", None), "layer", None)
                    if src is not None:
                        stack.append(src)
        return seen

    hv_layer  = model.get_layer("hv_head")
    seg_layer = model.get_layer("seg_head")

    hv_anc  = ancestors(hv_layer)
    seg_anc = ancestors(seg_layer)

    unique_hv_layers = hv_anc - seg_anc   # solo i layer della branchetta HV
    unique_seg_layers = seg_anc - hv_anc

    # Opzionale: togli InputLayer se presente
    unique_hv_layers = {L for L in unique_hv_layers if L.__class__.__name__ != "InputLayer"}
    unique_seg_layers = {L for L in unique_seg_layers if L.__class__.__name__ != "InputLayer"}

    for L in unique_seg_layers:
        L.trainable = False
    # 2. Ricompila con loss_weights nuovi e/o LR schedule diverso
    model.compile(
        optimizer=optimizer,
        loss={'seg_head': bce_dice_loss,
              'hv_head' : hover_loss_fixed},
        loss_weights={'seg_head': 1.0, 'hv_head': 2.0},
        metrics={'seg_head': [CellDice()], 'hv_head': []}
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_seg_head_cell_dice',
        mode='max',
        patience=4,
        restore_best_weights=True
    )
    # 3. Continua l’addestramento
    history = model.fit(train_gen,
                       validation_data=val_gen,
                       epochs=20,             # numero epoche aggiuntive
                       callbacks=[checkpoint_cb, earlystop_cb],
                       verbose=1)
"""
"""
from src.losses import bce_dice_loss,hover_loss_fixed
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

def build_unet_resnet34(input_shape=(256, 256, 3), freeze_encoder=True):
    # Encoder: ResNet34 senza top
    base_model = tf.keras.applications.ResNet50(
        include_top=False,  # no classification head
        weights='imagenet',
        input_shape=input_shape
    )
    
    if freeze_encoder:
        for layer in base_model.layers:
            layer.trainable = False

    # Estraggo feature maps da diversi blocchi
    skips = [
        base_model.get_layer('conv1_relu').output,   # 128x128
        base_model.get_layer('conv2_block3_out').output, # 64x64
        base_model.get_layer('conv3_block4_out').output, # 32x32
        base_model.get_layer('conv4_block6_out').output  # 16x16
    ]
    x = base_model.get_layer('conv5_block3_out').output  # 8x8

    # Decoder: upsampling + concatenazione con skip connection
    for skip in reversed(skips):
        x = UpSampling2D()(x)
        x = Concatenate()([x, skip])
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)

    # Output finale decoder: 256x256
    x = UpSampling2D()(x)

    return base_model.input, x

inputs, decoder_output = build_unet_resnet34(input_shape=(256,256,3), freeze_encoder=True)
tf.config.run_functions_eagerly(True)
# Head segmentazione
seg_head = Conv2D(16, (3, 3), padding='same', activation='relu')(decoder_output)
seg_head = Conv2D(3, (1, 1), activation='softmax', name='seg_head')(seg_head)

# Head HoVer (regressione)
hv_head = Conv2D(16, (3, 3), padding='same', activation='relu')(decoder_output)
hv_head = Conv2D(2, (1, 1), activation='linear', name='hv_head')(hv_head)

# Modello finale
model = Model(inputs=inputs, outputs={'seg_head': seg_head, 'hv_head': hv_head})

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=25,   # 10 epoche
        t_mul=2.0,                                # lunghezza fase raddoppia
        m_mul=0.8,                                # ampiezza si riduce
        alpha=1e-5 / 1e-3                         # min_lr
    )
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

metric = F1Score(threshold=0.5)
model.compile(
    optimizer=optimizer,
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hover_loss_fixed
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 1.5
    },
    metrics={'seg_head': [CellDice()],
             'hv_head' : [tf.keras.metrics.MeanSquaredError()]}
)
model.fit(train_gen,
          validation_data=val_gen,
          epochs=1,
          callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
          verbose=1)

# Recupera una batch
X_batch, y_batch = next(iter(val_gen))
y_pred = model.predict(X_batch, verbose=0)['seg_head']

# Applica la logica _bin_mask a mano (argmax-based)
labels_true = tf.argmax(y_batch['seg_head'], axis=-1)
labels_pred = tf.argmax(y_pred, axis=-1)

cell_true = tf.cast((labels_true == 0) | (labels_true == 2), tf.float32)
cell_pred = tf.cast((labels_pred == 0) | (labels_pred == 2), tf.float32)

intersection = tf.reduce_sum(cell_true * cell_pred)
union = tf.reduce_sum(cell_true) + tf.reduce_sum(cell_pred)
print("cell_true mean:", tf.reduce_mean(cell_true).numpy())
print("cell_pred mean:", tf.reduce_mean(cell_pred).numpy())
dice = (2 * intersection) / (union + 1e-6)
print(f"Manual CellDice (batch): {dice.numpy():.4f}")


i = 0  # indice della prima immagine nella batch
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cell_true[i], cmap='gray')
plt.title("GT (body ∪ border)")

plt.subplot(1, 3, 2)
plt.imshow(cell_pred[i], cmap='gray')
plt.title("Pred (body ∪ border)")

plt.subplot(2, 3, 1)
plt.imshow(cell_true[i] * cell_pred[i], cmap='gray')
plt.title("Intersection")

plt.subplot(2, 3, 2)
plt.imshow(labels_pred[i], cmap='nipy_spectral')
plt.title("Intersection")

plt.tight_layout()
plt.show()

# Sblocco encoder
for layer in model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss={
        'seg_head': bce_dice_loss,
        'hv_head': hover_loss_fixed
    },
    loss_weights={
        'seg_head': 1.0,
        'hv_head': 1.5
    },
    metrics={'seg_head': [CellDice()],
             'hv_head' : [tf.keras.metrics.MeanSquaredError()]}
)
# Allenamento completo
history = model.fit(train_gen,
          validation_data=val_gen,
          epochs=50,
          callbacks=[earlystop_cb, checkpoint_cb, tensorboard_cb, reduce_lr_cb],
          verbose=1)
"""
#OVERFITTING ?
hist = history.history           # dizionario {metric_name: [e1, e2, ...]}
plt.figure(figsize=(6, 4))
plt.plot(hist['loss'],     label='train loss')
plt.plot(hist['val_loss'], label='val loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss curves'); plt.legend(); plt.tight_layout()
plt.show()
    
    