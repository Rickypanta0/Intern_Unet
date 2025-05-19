import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from src.losses import  hover_loss_fixed

# Load data
base = os.path.join('data', 'raw', 'val')
folds = [
    (os.path.join(base, 'Fold 3', 'images', 'fold3', 'images.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'binary_masks.npy'),
     os.path.join(base, 'Fold 3', 'masks', 'fold3', 'distance.npy'))
]

N = 13
imgs_list, masks_list, dmaps_list = [], [], []
SEED = 42

for i, (img_path, mask_path, dist_path) in enumerate(folds):
    rng = np.random.default_rng(SEED + i)
    imgs = np.load(img_path, mmap_mode='r')
    masks = np.load(mask_path, mmap_mode='r')
    dmaps = np.load(dist_path, mmap_mode='r')
    
    idxs = rng.choice(len(imgs), size=N, replace=False)
    imgs_list.append(imgs[idxs])
    masks_list.append(masks[idxs])
    dmaps_list.append(dmaps[idxs])

X = np.concatenate(imgs_list, axis=0)
Y = np.concatenate(masks_list, axis=0)
HV = np.concatenate(dmaps_list, axis=0)

# Preprocess input images
X = X + 15
X = X / 255
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
mean = X_gray.mean(axis=(0, 1, 2), keepdims=True)
Xc = np.clip(mean + 1.4 * (X_gray - mean), 0.0, 1.0)

# Split
X_train, _, Y_train, _ = train_test_split(Xc, Y, test_size=0.1, random_state=SEED)
X_train_rgb = np.repeat(X_train, 3, axis=-1)
from src.losses import bce_dice_loss
# Load model
checkpoint_path = 'models/checkpoints/model_grayscale_HV.keras'

model = load_model(
    checkpoint_path,
    custom_objects={
        'bce_dice_loss': bce_dice_loss,
        'hover_loss_fixed': hover_loss_fixed
    }
)

# Predict
preds = model.predict(X_train_rgb)
seg_preds = preds['seg_head']
hv_preds  = preds['hv_head']  # (batch, H, W, 2)

# Watershed + HV map visualization and segmentation
for i in range(X_train.shape[0]):
    img = X_train[i, ..., 0]
    gt = seg_preds[i]
    seg = seg_preds[i]
    hv = hv_preds[i]

    # Binary mask from seg (body+border vs background)
    body_prob = seg[..., 0]
    border_prob = seg[..., 2]
    bg_prob = seg[..., 1]
    
    binary_mask = ((body_prob + border_prob) > bg_prob).astype(np.uint8)
    
    hv = hv.astype(np.float32)

    grad_x = cv2.Sobel(hv[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(hv[..., 1], cv2.CV_32F, 0, 1, ksize=3)

    # Sm = max(|∇x(px)|, |∇y(py)|)
    grad = np.maximum(np.abs(grad_x), np.abs(grad_y))
    #print("grad range:", grad.min(), grad.max(), grad.mean())
#
    #p1, p99 = np.percentile(grad, [1, 99])
    #grad_stretch = np.clip((grad - p1) / (p99 - p1 + 1e-6), 0, 1)
#
    ## Converti in immagine 8 bit
    #grad_vis = (grad_stretch * 255).astype(np.uint8)
    #plt.imshow(grad_vis, cmap='hot')
    #plt.title("Gradient HV Map (Sm)")
    #plt.colorbar()
    #plt.axis('off')
    #plt.show()
    # === Parametri ottimali trovati nel paper ===
    h = 0.4  # threshold per q (nuclei prob)
    k = 0.1  # threshold per Sm (contorni HV)

    # 1. Soglia mappa q
    tau_q_h = (body_prob > h).astype(np.uint8)  # τ(q,h)

    # 2. Soglia mappa Sm
    tau_Sm_k = (grad > k).astype(np.uint8)    # τ(Sm,k)

    # 3. Marker M = σ(τ(q,h) − τ(Sm,k))
    M = tau_q_h.astype(np.int32) - tau_Sm_k.astype(np.int32)
    M[M < 0] = 0  # ReLU

    # 4. Energy landscape E = [1 − τ(Sm,k)] * τ(q,h)
    E = (1 - tau_Sm_k) * tau_q_h  # pixel con alta q ma fuori dai bordi

    # 5. Distance transform sull’energia
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    distance = ndi.distance_transform_edt(E)

    # 6. Etichette marker
    markers, _ = ndi.label(M)

    # 7. Segmentazione finale
    labels_ws = watershed(-distance, markers, mask=tau_q_h)
    
    # 6. Visualizza il risultato
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Maschera nuclei (seg_pred)")
    
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title("Contorni HV (Sm > soglia)")
    
    plt.subplot(1, 3, 3)
    plt.imshow(labels_ws, cmap='nipy_spectral')
    plt.title("Segmentazione finale (Watershed)")
    plt.show()
