import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import radiomics
from mitk import nii_resample,nii_resize
from skimage import measure, morphology
import skimage.measure
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize
import surface_distance as surfdist

### 统计通过diameter_FP.py和diameter_missing.py统计出来的list（missing_diameter_list，FP_diameter_list，missing_class_list， missing_info_list）

# 统计遗漏肿瘤的长径 长径: 0.5-1, 1-2, 2-3, 3-5, >5, <0.5
missing_diameter_list = [
    1, 1, 0, 5, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
    1, 0, 0, 1, 0, 5, 5, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 5, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0,
    0, 2, 5, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 5, 0, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
    0, 1, 0, 0, 5, 0, 5, 0, 5, 5, 5, 0, 1, 0, 0, 0, 5, 5, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 5,
    5, 1, 5, 0, 5, 5, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 5, 5
]
cout_missing_list = [0, 0, 0, 0, 0, 0]
for i in missing_diameter_list:
    if i == 0:
        cout_missing_list[0] += 1
    elif i == 1:
        cout_missing_list[1] += 1
    elif i == 2:
        cout_missing_list[2] += 1
    elif i == 3:
        cout_missing_list[3] += 1
    elif i == 4:
        cout_missing_list[4] += 1
    elif i == 5:
        cout_missing_list[5] += 1
print('cout_missing_list', cout_missing_list)

# 统计假阳肿瘤的长径
FP_diameter_list = [
    0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 5, 0, 0, 0, 1, 1, 0, 1, 5, 0, 1, 1, 1, 0, 1, 1, 2, 0, 0,
    1, 1, 0, 1, 1, 0, 0, 1, 0, 5, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 5, 1, 2, 2, 0, 0, 0, 0, 5, 0, 1, 1, 0,
    0, 0, 5, 1, 0, 0, 1, 0, 0, 5, 1, 0, 0, 0, 5, 0, 0, 5, 1, 0, 0, 1, 1, 5, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0
]
cout_FP_list = [0, 0, 0, 0, 0, 0]
for i in FP_diameter_list:
    if i == 0:
        cout_FP_list[0] += 1
    elif i == 1:
        cout_FP_list[1] += 1
    elif i == 2:
        cout_FP_list[2] += 1
    elif i == 3:
        cout_FP_list[3] += 1
    elif i == 4:
        cout_FP_list[4] += 1
    elif i == 5:
        cout_FP_list[5] += 1
print('cout_FP_list', cout_FP_list)

# 统计遗漏肿瘤的种类
missing_class_list = [
    5, 5, 5, 3, 4, 4, 4, 2, 1, 5, 2, 2, 2, 2, 2, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 1, 4, 5, 5, 5, 5, 5, 5, 1, 1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 4, 3,
    4, 5, 3, 3, 3, 3, 4, 5, 4, 4, 4, 3, 4, 4, 3, 4, 3, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5,
    5, 5, 4, 4, 5, 5, 5, 5, 4, 5, 1, 3, 3, 4, 2, 2, 3, 1
]
cout_missing_class_list = [0, 0, 0, 0, 0]
for i in missing_class_list:
    if i == 1:
        cout_missing_class_list[0] += 1
    elif i == 2:
        cout_missing_class_list[1] += 1
    elif i == 3:
        cout_missing_class_list[2] += 1
    elif i == 4:
        cout_missing_class_list[3] += 1
    elif i == 5:
        cout_missing_class_list[4] += 1
print('cout_missing_class_list', cout_missing_class_list)

# cout_missing_list=[80, 65, 2, 1, 0, 22]
# cout_FP_list=[67, 76, 3, 0, 0, 12]
# cout_missing_class_list=[6, 73, 25, 41, 25]

# 统计遗漏的肿瘤所有信息
#################### 长径: 0.5-1, 1-2, 2-3, 3-5, >5
#################### 类别: 肝细胞肝癌HCC，肝脏转移瘤，胆管细胞癌，肝囊肿，肝血管瘤
###################### 文件名, 长径, 类别
missing_info_list = [['predict.nii.gz', 1, 5], ['predict.nii.gz', 1, 5], ['predict.nii.gz', 0, 5],
                     ['predict.nii.gz', 5, 3], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 3, 1],
                     ['predict.nii.gz', 5, 5], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 5],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 2, 5], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 4],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 4],
                     ['predict.nii.gz', 0, 5], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 1, 4], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 4],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 1, 3], ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 3],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 3],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 5, 3], ['predict.nii.gz', 5, 3],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 1],
                     ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 5], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 0, 5], ['predict.nii.gz', 1, 5], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 1, 5], ['predict.nii.gz', 1, 1], ['predict.nii.gz', 1, 1],
                     ['predict.nii.gz', 1, 4], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 5, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 1, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 2, 5],
                     ['predict.nii.gz', 5, 3], ['predict.nii.gz', 1, 3], ['predict.nii.gz', 1, 3],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 5, 3], ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 3],
                     ['predict.nii.gz', 1, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 4],
                     ['predict.nii.gz', 5, 3], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 5, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 5, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 5, 2],
                     ['predict.nii.gz', 5, 2], ['predict.nii.gz', 5, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 5, 2], ['predict.nii.gz', 5, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2], ['predict.nii.gz', 0, 2],
                     ['predict.nii.gz', 1, 5], ['predict.nii.gz', 5, 5], ['predict.nii.gz', 5, 5],
                     ['predict.nii.gz', 1, 5], ['predict.nii.gz', 5, 4], ['predict.nii.gz', 0, 4],
                     ['predict.nii.gz', 5, 5], ['predict.nii.gz', 5, 5], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 1, 5], ['predict.nii.gz', 1, 4], ['predict.nii.gz', 1, 5],
                     ['predict.nii.gz', 0, 1], ['predict.nii.gz', 0, 3], ['predict.nii.gz', 1, 3],
                     ['predict.nii.gz', 0, 4], ['predict.nii.gz', 1, 2], ['predict.nii.gz', 1, 2],
                     ['predict.nii.gz', 5, 3], ['predict.nii.gz', 5, 1]]
# [[长径0.5-1], [1-2], [2-3], [3-5], [>5]]
# [[肝细胞肝癌HCC，肝脏转移瘤，胆管细胞癌，肝囊肿，肝血管瘤], [,,,,], [,,,,], [,,,,], [,,,,]]
total_list = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
for li in missing_info_list:
    dia, cla = li[1], li[2]
    if dia == 0:
        if cla == 1:
            total_list[0][0] += 1
        elif cla == 2:
            total_list[0][1] += 1
        elif cla == 3:
            total_list[0][2] += 1
        elif cla == 4:
            total_list[0][3] += 1
        elif cla == 5:
            total_list[0][4] += 1
    elif dia == 1:
        if cla == 1:
            total_list[1][0] += 1
        elif cla == 2:
            total_list[1][1] += 1
        elif cla == 3:
            total_list[1][2] += 1
        elif cla == 4:
            total_list[1][3] += 1
        elif cla == 5:
            total_list[1][4] += 1
    elif dia == 2:
        if cla == 1:
            total_list[2][0] += 1
        elif cla == 2:
            total_list[2][1] += 1
        elif cla == 3:
            total_list[2][2] += 1
        elif cla == 4:
            total_list[2][3] += 1
        elif cla == 5:
            total_list[2][4] += 1
    elif dia == 3:
        if cla == 1:
            total_list[3][0] += 1
        elif cla == 2:
            total_list[3][1] += 1
        elif cla == 3:
            total_list[3][2] += 1
        elif cla == 4:
            total_list[3][3] += 1
        elif cla == 5:
            total_list[3][4] += 1
    elif dia == 4:
        if cla == 1:
            total_list[4][0] += 1
        elif cla == 2:
            total_list[4][1] += 1
        elif cla == 3:
            total_list[4][2] += 1
        elif cla == 4:
            total_list[4][3] += 1
        elif cla == 5:
            total_list[4][4] += 1

print(total_list)

total_list = [[1, 42, 9, 24, 4], [3, 23, 9, 16, 14], [0, 0, 0, 0, 2], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
