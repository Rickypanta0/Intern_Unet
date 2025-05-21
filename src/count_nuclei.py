from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2
import numpy as np

img_norm = cv2.normalize(binary_mask, None, 0,255, cv2.NORM_MINMAX)
img_8u = img_norm.astype(np.uint8)
ret, thresh = cv2.threshold(img_8u, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
image = opening
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(
            distance,
            footprint=np.ones((3,3)),
            labels=image,
            min_distance=1
            )
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)
contour_img = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
isolated_count = 0
cluster_count = 0
for label in np.unique(labels):
    if label<=0:
        continue
    
    single_mask = (labels == label).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(single_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts or len(cnts[0])<3:
    # non c'è abbastanza geometria per un hull
        continue
    cntr = cnts[0]
    area = cv2.contourArea(cntr)
    convex_hull = cv2.convexHull(cntr)
    convex_hull_area = cv2.contourArea(convex_hull)
    ratio = area / convex_hull_area
    if ratio < 0.91:
        # cluster contours in red
        cv2.drawContours(contour_img, [cntr], 0, (0,0,255), 2)
        cluster_count = cluster_count + 1
    else:
        # isolated contours in green
        cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
        isolated_count = isolated_count + 1
    index = index + 1
    #cv2.drawContours(contour_img, [cntr], -1, color, 2)
