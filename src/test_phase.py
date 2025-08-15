
import os
import numpy as np
import matplotlib.pyplot as plt
from src.losses import bce_dice_loss, hover_loss_fixed
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
from matplotlib.cm import get_cmap
from src.utils.metrics import get_fast_dice_2, get_fast_aji,get_fast_pq
import pandas as pd
import gc
from keras import backend as K
class DiceScore(tf.keras.metrics.Metric):
    """
    Dice coefficient (a.k.a. F1 per segmentazione binaria).
    Se y_pred è una mappa di probabilità applica la soglia desiderata,
    altrimenti accetta già tensori {0,1}.
    """
    def __init__(self, threshold=0.5, name="dice", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union        = self.add_weight(name="union",        initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast a float32 (evita overflow con somma su batch grandi)
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred > self.threshold, self.dtype)   # binarizza

        inter = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        smooth = tf.constant(1e-6, dtype=self.dtype)
        return (2.0 * self.intersection + smooth) / (self.union + smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)
    
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

from tensorflow.keras.metrics import F1Score, IoU
def score_testSet(preds, Y, HV, X):
    H = 256
    W = 256
    I_f1_list = []
    I_dice_list = []
    I_iou_list = []
    seg_preds = preds['seg_head']
    hv_preds  = preds['hv_head']

    f1_metric = F1Score(average='macro',threshold=0.5, name='f1')
    iou_metric = IoU(num_classes=2, target_class_ids=[1], name='iou')
    
    for i in tqdm(range(Y.shape[0])):
        seg = seg_preds[i]
        hv = hv_preds[i]
        hv_t = HV[i]
        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]
        #plt.subplot(1,3,1); plt.imshow(body_prob, cmap='gray'); plt.title("body")
        #plt.subplot(1,3,2); plt.imshow(border_prob, cmap='gray'); plt.title("border")
        #plt.subplot(1,3,3); plt.imshow(bg_prob, cmap='gray'); plt.title("background")
        #plt.show()
        y_pred_bin = (body_prob + border_prob > bg_prob).clip(0, 1).astype(np.int32)
        y_true_bin = Y[i].astype(np.int32)

        #plt.subplot(1,3,1); plt.imshow(Y[i], cmap='gray'); plt.title("GT")
        #plt.subplot(1,3,2); plt.imshow(y_pred_bin.reshape(Y[i].shape), cmap='gray'); plt.title("Pred")
        #plt.subplot(1,3,3); plt.imshow((Y[i].squeeze() == y_pred_bin).astype(int), cmap='gray'); plt.title("Match")
        #plt.show()

        iou_metric.update_state(y_true_bin.squeeze(), y_pred_bin)
        f1_metric.update_state(y_true_bin.squeeze(), y_pred_bin)

        if y_true_bin.sum() == 0 and y_pred_bin.sum() == 0:
            print("NNNNN")
            continue

        pred = np.stack([y_pred_bin, hv[..., 0], hv[..., 1]], axis=-1)
        label_pred = __proc_np_hv(pred,trhld=0.43)

        pred_GT = np.stack([Y[i].squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
        label_true = __proc_np_hv(pred_GT, GT=True)
 
        #print(np.unique(label_pred), np.unique(label_true))
        
        iou_val = get_fast_aji(label_true, label_pred, pred_GT, pred, seg, X[i])
        outcome,_ = get_fast_pq(label_true, label_pred)
        f1, sq, pq = outcome
        dice_val = get_fast_dice_2(label_true, label_pred)
        #print(f"ISTANCE: IoU[{i}]: {iou_val:.4f}, F1[{i}]: {outcome[0]:.4f}, Dice: {dice_val:.4f}")

        I_f1_list.append(f1)
        I_iou_list.append(iou_val)
        I_dice_list.append(dice_val)

    f1r = np.mean(I_f1_list)
    dicer = np.mean(I_dice_list)
    iour = np.mean(I_iou_list)

    iou = float(iou_metric.result())
    f1 = float(f1_metric.result())
    
    return [f1r, dicer, iour], [iou,f1]

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

def hema_rgb( img_rgb01):  # img in [0,1]
        ihc = rgb2hed(np.clip(img_rgb01, 0, 1))
        H = ihc[..., 0]
        hema_only = np.stack([H, np.zeros_like(H), np.zeros_like(H)], axis=-1)
        img_h = hed2rgb(hema_only).astype(np.float32)  # torna in RGB
        return np.clip(img_h, 0, 1) 

def pre_proces(checkpoint, X, weight_hv_head=1.0):
    if 'RGB' in checkpoint:
        X = np.clip(X + 15, 0, 255)
        X = X / 255.0
    elif 'HE' in checkpoint:
        X = X / 255.0
        X_ = []
        print("RRRR")
        for f in X:
            ihc_h = hema_rgb(f)
            #ihc_h, _, _ = HE_deconv(f)
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
    
    model = loadmodel(checkpoint_path=checkpoint, weight_hv_head=weight_hv_head)
    preds = model.predict(X)
    return preds, X

def labels_scores(checkpoint, Y, HV):
    f1_score = []
    dice_score = []
    iou_score = []

    f1, dice, iou = score_testSet(preds, Y, HV)
    
    f1_score.append(f1)
    dice_score.append(dice)
    iou_score.append(iou)

    print("checkpoint: ", checkpoint)
    print("F1:", f1)
    print("Dice:", dice)
    print("IoU:", iou)
    #print_labels(f1_score,dice_score,iou_score)


from src.postprocessing import __proc_np_hv
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
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
        label_pred = __proc_np_hv(pred,trhld=0.43)

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
    r2 = r2_score(N_gt, N_pred) #https://arxiv.org/pdf/2409.04175#page=1.98 [69]
    mae  = np.mean(np.abs(N_gt - N_pred))
    plt.title(f'ρ = {r:.3f}, MAE = {mae:.1f}, R2 = {r2}')
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

    idxs = rng.choice(len(imgs), size=N, replace=False)
    imgs_list.append(imgs)
    masks_list.append(masks)
    dmaps_list.append(dmaps)

X = np.concatenate(imgs_list, axis=0)
Y = np.concatenate(masks_list, axis=0)
HV = np.concatenate(dmaps_list, axis=0)

# Preprocess input images

k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 5))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)  


checkpoint_Blu='models/checkpoints/neo/model_BB_Blu.keras'
checkpoint_HE = 'models/checkpoints/neo/model_HE.keras'
checkpoint_RGB = 'models/checkpoints/neo/model_RGB.keras'
checkpoint_Gray = 'models/checkpoints/neo/model_Gray.keras'

checkpoint_paths=[checkpoint_RGB]
#
#model = loadmodel(checkpoint_path=checkpoint_RGB, weight_hv_head=1.0)
#preds = model.predict(X_)
preds, X = pre_proces(checkpoint_RGB, X)
k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 1))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)    

#print(f"min: {np.min(hv_batch)}, max {np.max(hv_batch)}, mean {np.mean(hv_batch)}")
""""""
from matplotlib.colors import hsv_to_rgb
for i in range(idx,idx + 4):
    fig, axs = plt.subplots(2, 3, figsize=(6,6))
    image = X[i]          # (H, W, 3) – RGB image (non usata da GenInstanceHV)
    mask_3ch = Y[i]       # (H, W, 3) – bodycell, background, bordercell
    hv = HV[i]
    mask_3chp = preds['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
    hvp = preds['hv_head'][i]
    clip_pct=99
    hx = hvp[..., 0]
    hy = hvp[..., 1]
    ang = np.arctan2(hy, hx)                 # [-pi, pi]
    hue = (ang + np.pi) / (2*np.pi)          # [0,1] -> direzione

    mag = np.sqrt(hx*hx + hy*hy)
    mmax = np.percentile(mag, clip_pct) + 1e-8
    val = np.clip(mag / mmax, 0, 1)          # [0,1] -> intensità

    sat = np.ones_like(val)                  # saturazione piena
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv).astype(np.float32)
    body, bg, border = tf.unstack(mask_3chp, axis=-1)
    Cp = tf.cast(body + border > bg, tf.float32)
    axs[0,0].imshow(X[i])                         # immagine normalizzata [0,1]
    axs[0,1].imshow(Cp, cmap='gray')  # body
    axs[0,2].imshow(hx)  # background
    axs[1,0].imshow(mask_3ch, cmap='gray')  # border
    axs[1,1].imshow(hv[...,0])  # body
    axs[1,2].imshow(hv[...,1])  # background

    plt.tight_layout()
    plt.show()
#N_gt, N_pred_rgb = grafo_diffusione(X,Y,HV,preds)

v, v_ = score_testSet(preds, Y, HV, X)
print(v,v_)
print(f"checkpoint: {checkpoint_RGB}\n")
print(f"F1: {v[0]}, Dice: {v[1]}, IoU: {v[2]}")
#RGB, HE, Blu, Gray Label
L_scores_F1 = [0.6494954673267342, 0.9624335286067217, 0.9472365862827455, 0.959334461216888]
L_scores_Dice = [0.6938941265614244, 0.962433908088167, 0.9472371079219255, 0.9593348696247608]
L_scores_IoU = [0.5750426737055911, 0.9290504189732318, 0.9020675153832152, 0.923387367286549]

#RGB, HE, Blu BinaryMask
scores_F1 = [0.8431848287582397]
scores_IoU = [0.7296468019485474]

# Mostra tabella
#print_labels_with_plot(scores_F1, scores_Dice, scores_IoU)   

#res_bias = test_count_bias(N_gt, N_pred_rgb)

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

#plot_bias_histogram(N_gt, N_pred_rgb, res_bias, title="Bias conteggio – modello Blu")
#print(res_bias)
