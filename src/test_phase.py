
import os
os.environ["KERAS_BACKEND"] = "torch" 
import numpy as np
from keras import ops
import matplotlib.pyplot as plt
#from src.losses import bce_dice_loss, hover_loss_fixed,hovernet_hv_loss_tf,hv_keras_loss
from keras.models import load_model
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
from matplotlib.cm import get_cmap
from src.utils.metrics import get_fast_dice_2, get_fast_aji,get_fast_pq
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from src.losses import HVLossMonaiTorch, bce_dice_loss_ops
from keras import backend as K
from keras.utils import register_keras_serializable
#from tensorflow.keras.applications.resnet import preprocess_input
"""
class DiceScore(tf.keras.metrics.Metric):
    
    #Dice coefficient (a.k.a. F1 per segmentazione binaria).
    #Se y_pred è una mappa di probabilità applica la soglia desiderata,
    #altrimenti accetta già tensori {0,1}.
    
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
"""
@register_keras_serializable()
class CellDice(keras.metrics.Metric):
    """F1 (Dice) su maschera cella = body ∪ border (vs background)."""
    def __init__(self, name="cell_dice", smooth=1e-6, **kw):
        super().__init__(name=name, **kw)
        self.smooth = float(smooth)
        self.intersection = self.add_weight(name="inter", initializer="zeros", dtype="float32")
        self.union        = self.add_weight(name="union", initializer="zeros", dtype="float32")

    def _bin_mask(self, y):
        body   = y[..., 0]
        bg     = y[..., 1]
        border = y[..., 2]
        return ops.cast((body + border) > bg, "float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_b = self._bin_mask(y_true)
        y_pred_b = self._bin_mask(y_pred)
        inter = ops.sum(y_true_b * y_pred_b)
        union = ops.sum(y_true_b) + ops.sum(y_pred_b)
        self.intersection.assign_add(ops.cast(inter, "float32"))
        self.union.assign_add(ops.cast(union, "float32"))

    def result(self):
        s = ops.cast(self.smooth, "float32")
        return (2.0 * self.intersection + s) / (self.union + s)

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

def loadmodel(checkpoint_path, weight_seg_head=0.25, weight_hv_head=1.0):
    model = load_model(
        checkpoint_path,
        custom_objects={
            "BCEDiceLoss": bce_dice_loss_ops,   # stesso nome salvato
            "HVLoss": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=3.0, ksize=5)
        },
        compile=False                    # ← evita la ricompilazione automatica
    )

    model.compile(
        optimizer= keras.optimizers.Adam(1e-3),
        loss={"seg_head": bce_dice_loss_ops,
               "hv_head": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=3.0, ksize=5)},
        loss_weights={"seg_head": weight_seg_head, "hv_head": weight_hv_head},
    )
    return model
import time
from keras.metrics import F1Score, IoU
def score_testSet(preds, Y, HV, X,trhld=0.43, min_area=10):
    H = 256
    W = 256
    I_f1_list = []
    I_dice_list = []
    I_iou_list = []
    seg_preds = preds['seg_head']
    hv_preds  = preds['hv_head']

    y_pred_all = (seg_preds[..., 0] + seg_preds[..., 2] > seg_preds[..., 1])    # bool
    y_true_all = np.squeeze(Y).astype(bool)

    #f1_metric = F1Score(average='macro',threshold=0.5, name='f1')
    #iou_metric = IoU(num_classes=2, target_class_ids=[1], name='iou')
    tp = np.logical_and(y_true_all,  y_pred_all).sum(dtype=np.int64)
    tn = np.logical_and(~y_true_all, ~y_pred_all).sum(dtype=np.int64)
    fp = np.logical_and(~y_true_all,  y_pred_all).sum(dtype=np.int64)
    fn = np.logical_and( y_true_all, ~y_pred_all).sum(dtype=np.int64)

    iou_pixel = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    f1_pos    = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 1.0
    f1_neg    = 2*tn / (2*tn + fp + fn) if (2*tn + fp + fn) else 1.0
    f1_pixel  = float((f1_pos + f1_neg) / 2)

    #t_hv = t_aji = t_pq = t_dice = t_misc = 0.0
    for i in tqdm(range(Y.shape[0])):
        t0 = time.perf_counter()
        seg = seg_preds[i]
        hv = hv_preds[i]
        hv_t = HV[i]

        #plt.subplot(1,3,1); plt.imshow(body_prob, cmap='gray'); plt.title("body")
        #plt.subplot(1,3,2); plt.imshow(border_prob, cmap='gray'); plt.title("border")
        #plt.subplot(1,3,3); plt.imshow(bg_prob, cmap='gray'); plt.title("background")
        #plt.show()

        #plt.subplot(1,3,1); plt.imshow(Y[i], cmap='gray'); plt.title("GT")
        #plt.subplot(1,3,2); plt.imshow(y_pred_bin.reshape(Y[i].shape), cmap='gray'); plt.title("Pred")
        #plt.subplot(1,3,3); plt.imshow((Y[i].squeeze() == y_pred_bin).astype(int), cmap='gray'); plt.title("Match")
        #plt.show()
        y_pred_bin = y_pred_all[i]                  # bool
        y_true_bin = y_true_all[i]  
        #iou_metric.update_state(y_true_bin.squeeze(), y_pred_bin)
        #f1_metric.update_state(y_true_bin.squeeze(), y_pred_bin)

        #t_misc += time.perf_counter() - t0
        
        if y_true_bin.sum() == 0 and y_pred_bin.sum() == 0:
            print("NNNNN")
            continue
        
        t0 = time.perf_counter()
        
        pred = np.stack([y_pred_bin, hv[..., 0], hv[..., 1]], axis=-1)
        label_pred = __proc_np_hv(pred,trhld=trhld, min_area=min_area)
        
        pred_GT = np.stack([Y[i].squeeze(), hv_t[..., 0], hv_t[..., 1]], axis=-1)
        label_true = __proc_np_hv(pred_GT, GT=True)
 
        #t_hv += time.perf_counter() - t0
        #t0 = time.perf_counter()#print(np.unique(label_pred), np.unique(label_true))
        
        iou_val = get_fast_aji(label_true, label_pred, pred_GT, pred, seg, X[i])
        #t_aji += time.perf_counter() - t0
        
        #t0 = time.perf_counter()
        outcome,_ = get_fast_pq(label_true, label_pred)
        f1, sq, pq = outcome
        #t_pq += time.perf_counter() - t0
        #t0 = time.perf_counter()
        dice_val = get_fast_dice_2(label_true, label_pred)
        #print(f"ISTANCE: IoU[{i}]: {iou_val:.4f}, F1[{i}]: {outcome[0]:.4f}, Dice: {dice_val:.4f}")
        #t_dice += time.perf_counter() - t0
        #print(f"HV:{t_hv:.2f}s  AJI:{t_aji:.2f}s  PQ:{t_pq:.2f}s  Dice:{t_dice:.2f}s  Misc:{t_misc:.2f}s")
        I_f1_list.append(f1)
        I_iou_list.append(iou_val)
        I_dice_list.append(dice_val)

    f1r = np.mean(I_f1_list)
    dicer = np.mean(I_dice_list)
    iour = np.mean(I_iou_list)

    return [f1r, dicer, iour], [iou_pixel, f1_pixel]

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

def pre_proces(checkpoint, X, weight_hv_head=2.0):
    if 'RGB' in checkpoint:
        print("RGB - Pre processing")
        X_ = []
        for f in X:
            f_ = (f+5)/255
            #f_ = preprocess_input(f)
            X_.append(f_)
        X = np.stack(X_, axis=0)
    elif 'HE' in checkpoint:
        X = X / 255.0
        X_ = []
        print("HE - Pre processing")
        for f in X:
            ihc_h = hema_rgb(f)
            #ihc_h, _, _ = HE_deconv(f)
            X_.append(ihc_h)
        X = np.stack(X_, axis=0)
    elif 'Blu' in checkpoint:
        print("Blu - Pre processing")
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
        print("Gray - Pre processing")
        X = X + 15
        X = X / 255
        X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis].astype(np.float32)
        X = np.repeat(X_gray, 3, axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Checkpoint non riconosciuto: {checkpoint}")
    
    model = loadmodel(checkpoint_path=checkpoint, weight_hv_head=weight_hv_head)
    preds = model.predict(X, batch_size=4)
    preds['hv_head'] = np.tanh(preds['hv_head'])
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
def grafo_diffusione(test_set_img, test_set_gt, test_set_hv, preds, trhld=0.43, min_area=10):
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
        label_pred = __proc_np_hv(pred,trhld=trhld, min_area=min_area)

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

def tune_params(X_val, Y_val, HV_val, seg_val, hv_val, t_fg_grid, min_area_grid):
    # 1) Precompute GT UNA VOLTA, fisso
    n_gt_list = []
    lg_list = []
    for i in range(len(X_val)):
        # union body∪border dal GT (niente threshold “float”: è già one-hot)
        gt_fg = Y_val[i].astype(np.float32)
        gt_stack = np.stack([gt_fg.squeeze(), HV_val[i,...,0], HV_val[i,...,1]], axis=-1)
        Lg = __proc_np_hv(gt_stack, GT=True)  # ← fisso!
        n_gt_list.append(int(Lg.max()))
        lg_list.append(Lg)

    best = (float("-inf"), float("-inf"))
    best_params = None

    for t_fg in tqdm(t_fg_grid):
        for min_area in min_area_grid:
            abs_err, smape, iou_list =[], [], []
            for i in range(len(X_val)):

                gt_fg = Y_val[i].astype(np.float32)
                gt_stack = np.stack([gt_fg.squeeze(), HV_val[i,...,0], HV_val[i,...,1]], axis=-1)    

                P = seg_val[i]  
                fg_prob = ((P[...,0]+P[...,2]) > P[...,1]).astype(np.float32)             # = P_body + P_border se ben calibrato
                pred_stack = np.stack([fg_prob, hv_val[i,...,0], hv_val[i,...,1]], axis=-1)

                Lp = __proc_np_hv(pred_stack, GT=False, trhld=t_fg, min_area=min_area)

                n_pred = int(Lp.max())
                n_gt   = n_gt_list[i]

                iou_val = get_fast_aji(lg_list[i], Lp, Y[i], gt_stack, P, X_val[i])
                iou_list.append(iou_val)

                abs_err.append(abs(n_pred - n_gt))
                #smape.append(2*abs(n_pred-n_gt)/(n_pred+n_gt+1e-6))

            mae = float(np.mean(abs_err))
            iou = float(np.mean(iou_list))
            if (iou > best[1]):  # tie-break corretto
                best = (mae, iou)
                best_params = (t_fg, min_area, mae, iou)
            print(f"t_fg={t_fg:.2f}, min_area={min_area}, MAE={mae:.2f}, iou={iou:.2f}")
            print(best_params)

    return best_params
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist

def mean_min_dist(centroids):
    if len(centroids) < 2:
        return np.inf
    return np.mean(pdist(centroids))

def analyze_nuclei(labeled):
    #labeled = label(mask_bin)
    props = regionprops(labeled)
    areas = [p.area for p in props]
    centroids = [p.centroid for p in props]
    num_nuclei = len(props)
    mean_area = np.mean(areas) if areas else 0
    return num_nuclei, mean_area, centroids

def categorize_images(instance_masks):
    all_areas = []
    all_dists = []
    stats = []

    for i in tqdm(range(len(instance_masks))):
        istance_mask = instance_masks[i]
        num_nuclei, mean_area, centroids = analyze_nuclei(istance_mask)
        mean_dist = mean_min_dist(centroids)
        stats.append((num_nuclei, mean_area, mean_dist))
        all_areas.append(mean_area)
        all_dists.append(mean_dist)

    area_thresh = np.median(all_areas)
    dist_thresh = np.median(all_dists)

    categories = []
    for i, (n, a, d) in enumerate(stats):
        area_cat = "piccoli" if a < area_thresh else "grandi"
        density_cat = "alta" if d < dist_thresh else "bassa"
        categories.append((area_cat, density_cat))
    
    return categories  # lista di tuple (area_cat, density_cat)

def count_nuclei(mask_bin):
    labeled = label(mask_bin)
    return np.max(labeled)

def test_model_on_categories(preds, HV, Y, categories, trhld, min_area):
    from keras import backend as K

    seg_preds = preds['seg_head']
    hv_preds = preds['hv_head']

    results = {
        ("piccoli", "alta"): [],
        ("piccoli", "bassa"): [],
        ("grandi", "alta"): [],
        ("grandi", "bassa"): []
    }

    for i in range(Y.shape[0]):
        gt_fg = Y[i].astype(np.float32)
        gt_stack = np.stack([gt_fg.squeeze(), HV[i,...,0], HV[i,...,1]], axis=-1)

        n_true = __proc_np_hv(gt_stack, GT=True)

        seg = seg_preds[i]
        body_prob = seg[..., 0]
        border_prob = seg[..., 2]
        bg_prob = seg[..., 1]

        pred_mask = ((body_prob + border_prob) > bg_prob).astype(np.uint8)
        pred_stack = np.stack([pred_mask, hv_preds[i,...,0], hv_preds[i,...,1]], axis=-1)

        n_pred = __proc_np_hv(pred_stack, GT=False, trhld=trhld, min_area=min_area)

        error = abs(n_true - n_pred)

        key = categories[i]
        results[key].append(error)

    # Calcola media errore per categoria
    results_mean = {k: np.mean(v) if v else 0 for k, v in results.items()}

    return results

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

k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 5))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)  


checkpoint_Blu='models/checkpoints/neo/model_Blu.keras'
checkpoint_HE = 'models/checkpoints/neo/model_HE.keras'
checkpoint_RGB = 'models/checkpoints/neo/model_RGB.keras'
checkpoint_Gray = 'models/checkpoints/neo/model_Gray.keras'

checkpoint_paths=[checkpoint_HE]
#
#model = loadmodel(checkpoint_path=checkpoint_RGB, weight_hv_head=1.0)
#preds = model.predict(X_)
preds, X = pre_proces(checkpoint_HE, X)

print("shape:", HV.shape, "dtype:", HV.dtype)     # atteso: (N,H,W,2), float32
print("GT per-channel min/max:",
      HV[...,0].min(), HV[...,0].max(),
      HV[...,1].min(), HV[...,1].max())
print("channels equal? mean|x-y| =", np.mean(np.abs(HV[...,0]-HV[...,1])))

seg_val_preds = preds['seg_head']
hv_val_preds  = preds['hv_head']
#print("pred per-channel min/max:",
#      hv_val_preds[...,0].min(), hv_val_preds[...,0].max(),
#      hv_val_preds[...,1].min(), hv_val_preds[...,1].max())
k = min(4, X.shape[0])                                 # quante immagini mostrare
high = max(1, X.shape[0] - (k - 1))                    # ultimo indice di partenza valido + 1
idx = np.random.randint(0, high)    

from matplotlib.colors import hsv_to_rgb
for i in range(idx,idx + 5):
    fig, axs = plt.subplots(2, 3, figsize=(6,6))
    image = X[i]          # (H, W, 3) – RGB image (non usata da GenInstanceHV)
    mask_3ch = Y[i]       # (H, W, 3) – bodycell, background, bordercell
    hv = HV[i]
    mask_3chp = preds['seg_head'][i]       # (H, W, 3) – bodycell, background, bordercell
    hvp = preds['hv_head'][i]
    clip_pct=99
    hx = hvp[..., 0]
    hy = hvp[..., 1]
    body = mask_3chp[...,0]
    bg = mask_3chp[...,1] 
    border = mask_3chp[...,2]

    Cp = ((body + border) > bg).astype(np.float32) 
    axs[0,0].imshow(Cp, cmap='gray')                         # immagine normalizzata [0,1]
    axs[0,1].imshow(hx)  # body
    axs[0,2].imshow(hy)  # background
    axs[1,0].imshow(Y[i], cmap='gray')  # border
    axs[1,1].imshow(hv[...,0])  # body
    axs[1,2].imshow(hv[...,1])  # background

    plt.tight_layout()
    plt.show()

t_fg_grid    = np.round(np.arange(0.35, 0.61, 0.02), 2)
#t_seed_grid  = [0.55, 0.60, 0.65]
min_area_grid= [10, 20, 30]

X_val, _, Y_val, _, HV_val, _ = train_test_split(
    X, Y, HV, test_size=0.1, random_state=SEED
)

best = tune_params(X, Y, HV, seg_val_preds, hv_val_preds,t_fg_grid, min_area_grid)
best_trshld =  best[0]
best_min_area = best[1]
print(f"Migliori: t_fg={best[0]}, min_area={best[1]} | MAE={best[2]:.2f}, sMAPE={best[3]:.3f}")

N_gt, N_pred_rgb = grafo_diffusione(X,Y,HV,preds, trhld=best_trshld, min_area=best_min_area)

v, v_ = score_testSet(preds, Y, HV, X, trhld=best_trshld, min_area=best_min_area)

print(v,v_)
print(f"checkpoint: {checkpoint_HE}\n")
print(f"F1: {v[0]}, Dice: {v[1]}, IoU: {v[2]}")
#RGB, HE, Blu, Gray Label
L_scores_F1 = [0.638529049884507, 0.6217882334816339, 0.9472365862827455, 0.959334461216888]
L_scores_Dice = [0.6069365145261811, 0.5990955220675084, 0.9472371079219255, 0.9593348696247608]
L_scores_IoU = [0.5296632884537072, 0.5147541559041431, 0.9020675153832152, 0.923387367286549]

#RGB, HE, Blu BinaryMask
scores_F1 = [0.8700626496266222, 0.8625804271868238]
scores_IoU = [0.655638917837067, 0.6399422174400162]

res_bias = test_count_bias(N_gt, N_pred_rgb)

plot_bias_histogram(N_gt, N_pred_rgb, res_bias, title="Bias conteggio – modello RGB")

from src.hv_GT_generator import build_instance_map_valuewise
from matplotlib.patches import Patch

def label_img_calc_error(checkpoints, X, HV, Y, best_trshld, best_min_area):
    labeled_path = os.path.join('data', 'raw_neoplastic','Fold 3', 'masks',  'masks.npy')
    label_M = np.load(labeled_path, mmap_mode='r')
    label_mask = [build_instance_map_valuewise(i) for i in label_M]

    categories = categorize_images(label_mask)

    # Ordine categorie (gruppi sull'asse x) e modelli (box dentro ogni gruppo)
    order  = [("piccoli","alta"), ("piccoli","bassa"), ("grandi","alta"), ("grandi","bassa")]
    labels = ['RGB', 'Hesoine', 'Blu', 'Gray']

    # Contenitore: per ogni categoria, una lista di 4 liste (una per modello)
    errs_by_cat = {cat: [ [] for _ in labels ] for cat in order}

    def _model_label_from_ckpt(path: str) -> str:
        if 'RGB' in path: return 'RGB'
        if 'HE' in path or 'Hesoine' in path: return 'Hesoine'
        if 'Blu' in path or 'Blue' in path: return 'Blu'
        if 'Gray' in path or 'Grey' in path: return 'Gray'
        return labels[0]  # fallback

    for check in checkpoints:
        model = load_model(
            check,
            custom_objects={
                "BCEDiceLoss": bce_dice_loss_ops,
                "HVLoss": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=3.0, ksize=5)
            },
            compile=False
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss={"seg_head": bce_dice_loss_ops, "hv_head": HVLossMonaiTorch(lambda_mse=2.0, lambda_grad=3.0, ksize=5)},
            loss_weights={"seg_head": 1.0, "hv_head": 2.0},
        )

        preds = model.compile(X, batch=4)  # (lasciato invariato come richiesto)
        results = test_model_on_categories(preds, HV, Y, categories, best_trshld, best_min_area)
        # `results` deve essere: {("piccoli","alta"): [err_i...], ...}

        mlabel = _model_label_from_ckpt(check)
        m_idx = labels.index(mlabel)

        # Accumula gli errori grezzi per categoria e modello
        for cat in order:
            errs_by_cat[cat][m_idx].extend(results.get(cat, []))

    # Unico grafico raggruppato
    _box_plot_grouped(errs_by_cat, labels, order)


def _box_plot_grouped(errs_by_cat, model_labels, cat_order):
    # Impostazioni layout
    n_groups  = len(cat_order)          # 4 categorie
    n_models  = len(model_labels)       # 4 modelli
    box_w     = 0.6
    dx        = 0.8                     # distanza tra box nello stesso gruppo
    gap       = 1.5                     # gap tra gruppi
    colors    = ['red', 'violet', 'blue', 'gray']  # 'blue' (non 'blu')

    # Prepara dati e posizioni
    data = []
    positions = []
    group_centers = []

    base = 1.0
    for g, cat in enumerate(cat_order):
        # posizioni delle 4 box del gruppo g
        group_pos = [base + m*dx for m in range(n_models)]
        positions.extend(group_pos)

        # dati: per ogni modello, la lista errori (se vuota, metti NaN per evitare errori)
        for m in range(n_models):
            vals = errs_by_cat[cat][m]
            data.append(vals if len(vals) else [np.nan])

        # centro gruppo per xtick
        group_centers.append(np.mean(group_pos))
        # avanza base al prossimo gruppo
        base += n_models*dx + gap

    fig, ax = plt.subplots(figsize=(12, 5))
    bplots = ax.boxplot(
        data,
        positions=positions,
        widths=box_w,
        patch_artist=True,
        manage_ticks=False
    )

    # Colora le box per modello (ripetendo il pattern colori)
    for i, patch in enumerate(bplots['boxes']):
        m = i % n_models
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.8)

    # Asse x: nomi categorie
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{a}/{d}" for (a, d) in cat_order])
    ax.set_ylabel("Errore di conteggio")
    ax.set_title("Errore di conteggio per categoria (boxplot raggruppato)")
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Legenda modelli
    legend_patches = [Patch(facecolor=colors[m], label=model_labels[m]) for m in range(n_models)]
    ax.legend(handles=legend_patches, title="Modello", loc="upper right")

    plt.tight_layout()
    plt.show()
#for k, err in results.items():
#    print(f"{k[0]} nuclei - {k[1]} densità → errore medio: {err:.2f}")
