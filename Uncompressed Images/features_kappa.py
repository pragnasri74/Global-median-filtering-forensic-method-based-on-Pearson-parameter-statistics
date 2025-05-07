import numpy as np
import cv2
from scipy.stats import skew, kurtosis

def features_kappa(img):
    """
    Compute a 23-D feature vector:
      - 18-bin histogram of valid κ values
      - 4 statistical moments of κ
      - 1 SRV count (number of dropped blocks)
    """

    # 1) Ensure grayscale
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # 2) Ensure uint8 for medianBlur
    if img_gray.dtype != np.uint8:
        # scale to [0,255] and cast
        img_u8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img_gray

    # 3) Median filter on 8-bit image
    med_u8 = cv2.medianBlur(img_u8, 3)

    # 4) Convert to float64 for residual computation
    img_f      = img_u8.astype(np.float64)
    med_f      = med_u8.astype(np.float64)
    residual   = img_f - med_f

    # 5) Sliding-window skew & kurtosis
    win_sz = 3
    step   = 2
    sko_list, kuo_list = [], []
    H, W = residual.shape

    for i in range(0, H - win_sz + 1, step):
        for j in range(0, W - win_sz + 1, step):
            block = residual[i:i+win_sz, j:j+win_sz].ravel()
            sko_list.append(skew(block))
            kuo_list.append(kurtosis(block, fisher=False))

    sko = np.array(sko_list)
    kuo = np.array(kuo_list)
    total_blocks = len(sko)

    # 6) Drop NaN/Inf from skew/kurtosis
    valid_sk = np.isfinite(sko) & np.isfinite(kuo)
    sko_n, kuo_n = sko[valid_sk], kuo[valid_sk]

    # 7) Compute κ = [S² (K+3)²] / [4(4K−3S²)(2K−3S²−6)]
    num = (sko_n*2) * (kuo_n + 3)*2
    den = 4 * (4*kuo_n - 3*sko_n*2) * (2*kuo_n - 3*sko_n*2 - 6)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = num / den

    # 8) Keep only finite κ in (−2,1)
    valid_k = np.isfinite(kappa) & (kappa > -2) & (kappa < 1)
    kappa_valid = kappa[valid_k]

    # 9) SRV count = how many dropped
    srv_count = total_blocks - len(kappa_valid)

    # 10) If nothing left, return zeros
    if len(kappa_valid) == 0:
        return np.zeros(23, dtype=np.float64)

    # 11) Build the 23-D feature vector
    hist, _ = np.histogram(kappa_valid, bins=18)
    f1 = np.mean(kappa_valid)
    f2 = np.var(kappa_valid)
    f3 = skew(kappa_valid)
    f4 = kurtosis(kappa_valid, fisher=False)

    return np.hstack([hist, [f1, f2, f3, f4, srv_count]])
