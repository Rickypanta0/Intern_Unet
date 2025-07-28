
import os
import numpy as np
import matplotlib.pyplot as plt
from src.losses import bce_dice_loss, hover_loss_fixed
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
from matplotlib.cm import get_cmap
from src.utils.metrics import dice_loss
import pandas as pd
import gc
from keras import backend as K

def HE_deconv(img):
    ihc_hed = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    
    return ihc_h, ihc_e, ihc_d 

def loadmodel(checkpoint_path, weight_seg_head=1.0, weight_hv_head=1.0):
    model = load_model(
        checkpoint_path,
        custom_objects={
            "BCEDiceLoss": bce_dice_loss,   # stesso nome salvato
            "HVLoss": hover_loss_fixed
        },
        compile=False                    # ← evita la ricompilazione automatica
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"seg_head": bce_dice_loss, "hv_head": hover_loss_fixed},
        loss_weights={"seg_head": weight_seg_head, "hv_head": weight_hv_head},
    )
    return model
from torchmetrics.classification import JaccardIndex
import torch
from torcheval.metrics.functional import binary_f1_score

def score_testSet(preds, Y):
    H = 256
    W = 256
    f1_list = []
    dice_list = []
    iou_list = []
    seg_preds = preds['seg_head']
    jaccard = JaccardIndex(task='binary')
    for i in range(Y.shape[0]):
        seg = seg_preds[i]
        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]
        y_pred_bin = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)
        y_true_bin = Y[i]
        """"""
        plt.subplot(1,3,1); plt.imshow(Y[i], cmap='gray'); plt.title("GT")
        plt.subplot(1,3,2); plt.imshow(y_pred_bin.reshape(Y[i].shape), cmap='gray'); plt.title("Pred")
        plt.subplot(1,3,3); plt.imshow((Y[i].squeeze() == y_pred_bin).astype(int), cmap='gray'); plt.title("Match")
        plt.show()
        
        y_pred_flat = y_pred_bin.flatten()
        y_true_flat = y_true_bin.flatten()

        if y_true_flat.sum() == 0 and y_pred_flat.sum() == 0:
            continue
        pred_tensor = torch.tensor(y_pred_flat, dtype=torch.long)
        true_tensor = torch.tensor(y_true_flat, dtype=torch.long)

        pred_tensor_ = torch.tensor(y_pred_bin.reshape(1, 1, H, W), dtype=torch.float32)
        true_tensor_ = torch.tensor(y_true_bin.reshape(1, 1, H, W), dtype=torch.float32)

        iou_val = jaccard(pred_tensor, true_tensor).item()
        f1_val = binary_f1_score(pred_tensor, true_tensor).item()
        dice_val = 1 - dice_loss(pred_tensor_, true_tensor_).item()

        #print(f"IoU[{i}]: {iou_val:.4f}, F1[{i}]: {f1_val:.4f}, Dice: {dice_val:.4f}")

        f1_list.append(f1_val)
        iou_list.append(iou_val)
        dice_list.append(dice_val)

    f1 = np.mean(f1_list)
    dice = np.mean(dice_list)
    iou = np.mean(iou_list)
    return f1, dice, iou

def print_labels_with_plot(scores_F1, scores_Dice, scores_IoU):
    # Crea DataFrame
    df = pd.DataFrame(index=["RGB", "Hesoine", "Blue Channel", "Gray"],
                      columns=["F1", "Dice", "IoU"])
    df['F1'] = scores_F1
    df['Dice'] = scores_Dice
    df['IoU'] = scores_IoU

    # Plot a barre
    ax = df.plot(kind='bar', figsize=(8, 5), rot=0)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Confronto delle metriche per preprocessing')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Tabella con highlight
    styled = df.style.highlight_max(axis=1, color='lightgreen')
    print(df)
    return styled

def labels_scores(checkpoint_paths, X_orig, Y):
    f1_score = []
    dice_score = []
    iou_score = []
    for i, checkpoint in enumerate(checkpoint_paths):
        X = np.array(X_orig, copy=True)
        if 'RGB' in checkpoint:
            X = np.clip(X + 15, 0, 255)
            X = X / 255.0

        elif 'HE' in checkpoint:
            X = X / 255.0
            X_ = []
            for f in X:
                ihc_h, _, _ = HE_deconv(f)
                X_.append(ihc_h)
            X = np.stack(X_, axis=0)

        elif 'Blu' in checkpoint:
            X = X / 255.0
            X_ = []
            cmap = get_cmap('Blues')
            for f in X:
                blu = f[..., 2]
                blu_enhanced = np.clip(blu + 0.05, 0, 1)
                img_rgb = cmap(blu_enhanced)[..., :3]
                X_.append(img_rgb)
            X = np.stack(X_, axis=0)
        elif 'Gray' in checkpoint:
            X = X + 15
            X = X / 255
            X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
            X = np.repeat(X_gray, 3, axis=-1).astype(np.float32)
        else:
            raise ValueError(f"Checkpoint non riconosciuto: {checkpoint}")

        model = loadmodel(checkpoint_path=checkpoint_paths[i], weight_hv_head=1.0)
        preds = model.predict(X)

        f1, dice, iou = score_testSet(preds, Y)
        
        f1_score.append(f1)
        dice_score.append(dice)
        iou_score.append(iou)
    
        del model, preds, X
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()

        print("checkpoint: ", checkpoint)
        print("F1:", f1)
        print("Dice:", dice)
        print("IoU:", iou)
    #print_labels(f1_score,dice_score,iou_score)


from src.postprocessing import __proc_np_hv
from scipy.stats import pearsonr
def grafo_diffusione(test_set_img, test_set_gt, test_set_hv, preds):
    N_gt = []
    N_pred = []

    seg_preds = preds['seg_head']
    hv_preds  = preds['hv_head']  # (batch, H, W, 2)
    
    for i in tqdm(range(test_set_img.shape[0])):
        mask_gt = test_set_gt[i]
        #costruzione maschera binaria
        seg = seg_preds[i]
        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]
        mask_pred = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.float32)

        #determinazione labels pred
        hv = hv_preds[i]
        pred = np.stack([mask_pred, hv[..., 0], hv[..., 1]], axis=-1)
        # segmentazione con watershed guidata da HV map
        label_pred = __proc_np_hv(pred,trhld=0.55)

        #determinazione labels GT
        hv_t = test_set_hv[i]
        pred_ = np.stack([mask_gt.squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
        label_gt = __proc_np_hv(pred_,GT=True)

        N_pred.append(np.max(np.unique(label_pred)))
        N_gt.append(np.max(np.unique(label_gt)))
    N_gt   = np.array(N_gt)
    N_pred = np.array(N_pred)
    
    # 2. Scatter + linea y=x
    plt.figure(figsize=(4,4))
    plt.scatter(N_gt, N_pred, s=15, alpha=0.7)
    plt.plot([0, N_gt.max()], [0, N_gt.max()], 'k--')
    plt.xlabel('Nuclei GT'); plt.ylabel('Nuclei predetti')

    # 3. Correlazione e MAE
    r, _ = pearsonr(N_gt, N_pred)
    mae  = np.mean(np.abs(N_gt - N_pred))
    plt.title(f'ρ = {r:.3f}, MAE = {mae:.1f}')
    plt.tight_layout(); plt.show()
    
    return N_gt, N_pred
from dataclasses import dataclass
from typing import Dict, Literal, Optional
from scipy import stats as st


@dataclass
class BiasTestResult:
    n: int
    mean_diff: float
    std_diff: float
    ci95: tuple
    shapiro_p: float
    t_stat: float
    t_p: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]


def test_count_bias(
    gt: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.05
) -> BiasTestResult:
    """
    Verifica se il modello è privo di bias di conteggio:
    H0: mean(pred - gt) = 0  (nessun bias)

    Restituisce sia il t-test one-sample sia Wilcoxon (robusto).
    """
    gt = np.asarray(gt).astype(float)
    pred = np.asarray(pred).astype(float)
    mask = np.isfinite(gt) & np.isfinite(pred)
    d = pred[mask] - gt[mask]                      # differenze (bias per immagine)
    n = d.size
    if n < 3:
        raise ValueError("Servono almeno 3 immagini per un test affidabile.")

    mean_d = d.mean()
    std_d = d.std(ddof=1)

    # Shapiro-Wilk per normalità delle differenze
    shapiro_stat, shapiro_p = st.shapiro(d) if n <= 5000 else (np.nan, np.nan)

    # t-test one-sample contro 0
    t_stat, t_p = st.ttest_1samp(d, popmean=0.0)

    # IC 95% per la media
    se = std_d / np.sqrt(n)
    t_crit = st.t.ppf(1 - alpha / 2, df=n - 1)
    ci95 = (mean_d - t_crit * se, mean_d + t_crit * se)

    # Wilcoxon signed-rank (saltato se tutte le diff sono 0)
    w_stat = w_p = None
    if np.any(d != 0):
        w_stat, w_p = st.wilcoxon(d, zero_method="wilcox", alternative="two-sided", correction=False)

    return BiasTestResult(
        n=n, mean_diff=mean_d, std_diff=std_d, ci95=ci95,
        shapiro_p=shapiro_p, t_stat=t_stat, t_p=t_p,
        wilcoxon_stat=w_stat, wilcoxon_p=w_p
    )



base = os.path.join('data', 'raw_neoplastic')
folds = [
    (os.path.join(base,'Fold 3', 'images', 'images.npy'),
     os.path.join(base,'Fold 3', 'masks', 'binary_masks.npy'),
     os.path.join(base, 'Fold 3','masks', 'distance.npy'))
]

N = 14
imgs_list, masks_list, dmaps_list = [], [], []
SEED = 42

for i, (img_path, mask_path, dist_path) in enumerate(folds):
    rng = np.random.default_rng(SEED + i)
    imgs = np.load(img_path, mmap_mode='r')
    masks = np.load(mask_path, mmap_mode='r')
    dmaps = np.load(dist_path, mmap_mode='r')

    #idxs = rng.choice(len(imgs), size=N, replace=False)
    imgs_list.append(imgs)
    masks_list.append(masks)
    dmaps_list.append(dmaps)

X = np.concatenate(imgs_list, axis=0)
Y = np.concatenate(masks_list, axis=0)
HV = np.concatenate(dmaps_list, axis=0)

# Preprocess input images
X = X + 15
X = X / 255
k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 5))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)  



#for i in range(idx,idx+5):
#    ihc_h, _, _ = HE_deconv(X[i])
#    
#    blu = X[i,...,2]
#    cmap = get_cmap('Blues')
#    channel3_rgb = cmap(blu)[:, :, :3] 
#    
#    fig, axs = plt.subplots(1,3,figsize=(8,8))
#    axs[0].imshow(X[i])
#    axs[1].imshow(channel3_rgb)
#    axs[2].imshow(channel3_rgb)
#    plt.tight_layout()
#    plt.show()
#X_ = []
#for f in X:
#    ihc_h, _, _ = HE_deconv(f)
#    X_.append(ihc_h)
#X_ = np.stack(X_, axis=0)
#
#
#X_ = []
#for f in X:
#    blu = f[...,2]
#    blu_enhanced = np.clip(blu + 0.05, 0, 1)
#    cmap = get_cmap('Blues')
#    img_rgb = cmap(blu_enhanced)[:, :, :3] 
#    X_.append(img_rgb)
#X_ = np.stack(X_,axis=0)
#
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
X_ = np.repeat(X_gray, 3, axis=-1).astype(np.float32)
checkpoint_Blu='models/checkpoints/neo/model_Blu.keras'
checkpoint_HE = 'models/checkpoints/neo/model_HE.keras'
checkpoint_RGB = 'models/checkpoints/neo/model_RGB.keras'
checkpoint_Gray = 'models/checkpoints/neo/model_Gray.keras'

checkpoint_paths=[checkpoint_RGB]
#
model = loadmodel(checkpoint_path='models/checkpoints/neo/model_Gray.keras', weight_hv_head=1.0)
preds = model.predict(X_)
N_gt, N_pred_rgb = grafo_diffusione(X_,Y,HV,preds)
#
#labels_scores(checkpoint_paths, X, Y)

scores_F1 = [0.9647045533373284, 0.9624335286067217, 0.9472365862827455, 0.959334461216888]
scores_Dice = [0.9647049096294574, 0.962433908088167, 0.9472371079219255, 0.9593348696247608]
scores_IoU = [0.9331706415897428, 0.9290504189732318, 0.9020675153832152, 0.923387367286549]

# Mostra tabella
#print_labels_with_plot(scores_F1, scores_Dice, scores_IoU)   

res_bias = test_count_bias(N_gt, N_pred_rgb)

def plot_bias_histogram(N_gt, N_pred, res, bins=30, title="Bias di conteggio"):
    diff = np.asarray(N_pred, float) - np.asarray(N_gt, float)
    diff = diff[np.isfinite(diff)]
    plt.figure(figsize=(6,4))
    plt.hist(diff, bins=bins, alpha=0.85)
    # linee: media e intervallo di confidenza
    plt.axvline(res.mean_diff, color='tab:orange', lw=2, label=f"media = {res.mean_diff:.2f}")
    plt.axvline(res.ci95[0],   color='tab:red', ls='--', lw=1.5, label=f"IC95%: [{res.ci95[0]:.2f}, {res.ci95[1]:.2f}]")
    plt.axvline(res.ci95[1],   color='tab:red', ls='--', lw=1.5)
    plt.axvline(0, color='k', lw=1)
    plt.xlabel("Differenza (Pred − GT)")
    plt.ylabel("Frequenza")
    plt.title(title)
    txt = f"t = {res.t_stat:.2f}, p = {res.t_p:.3e}\n" \
          f"Wilcoxon p = {res.wilcoxon_p:.3e}" if res.wilcoxon_p is not None else \
          f"t = {res.t_stat:.2f}, p = {res.t_p:.3e}"
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

plot_bias_histogram(N_gt, N_pred_rgb, res_bias, title="Bias conteggio – modello Blu")
print(res_bias)
