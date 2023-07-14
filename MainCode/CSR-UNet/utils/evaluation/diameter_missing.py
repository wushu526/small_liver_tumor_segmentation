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

### 计算统计预测遗漏的数据，占位和对应长径列表

################## Computing Dice###################################
from radiomics import featureextractor

###########
l_label0, l_predict0 = [], []
l_label1, l_predict1 = [], []
l_label2, l_predict2 = [], []
l_label3, l_predict3 = [], []
l_label4, l_predict4 = [], []
mp_dice0, mp_dice1, mp_dice2, mp_dice3, mp_dice4 = [], [], [], [], []
mp_iou0, mp_iou1, mp_iou2, mp_iou3, mp_iou4 = [], [], [], [], []
mp_kappa0, mp_kappa1, mp_kappa2, mp_kappa3, mp_kappa4 = [], [], [], [], []
mp_precision0, mp_precision1, mp_precision2, mp_precision3, mp_precision4 = [], [], [], [], []
mp_recall0, mp_recall1, mp_recall2, mp_recall3, mp_recall4 = [], [], [], [], []
mp_f10, mp_f11, mp_f12, mp_f13, mp_f14 = [], [], [], [], []
mp_acc0, mp_acc1, mp_acc2, mp_acc3, mp_acc4 = [], [], [], [], []
mp_auc0, mp_auc1, mp_auc2, mp_auc3, mp_auc4 = [], [], [], [], []
mp_hd0, mp_hd1, mp_hd2, mp_hd3, mp_hd4 = [], [], [], [], []
mp_vs0, mp_vs1, mp_vs2, mp_vs3, mp_vs4 = [], [], [], [], []
l0, l1, l2, l3, l4 = 0, 0, 0, 0, 0
missing_diameter_list = []
missing_class_list = []
missing_info_list = []
missing_name_list = []

settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None
settings['interpolator'] = 'sitkBSpline'
settings['correctMask'] = True
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)


def func_dice(mask1, mask2):
    inter = float(np.sum((mask1 > 0) * (mask2 > 0)))
    summ = float(np.sum(mask1 > 0) + np.sum(mask2 > 0))
    dice = 2 * inter / summ
    return dice


def func_iou(mask1, mask2):
    inter = float(np.sum((mask1 > 0) * (mask2 > 0)))
    summ = float(np.sum((mask1 > 0) + (mask2 > 0)))
    iou = inter / summ
    return iou


def func_kappa(mask1, mask2):
    ##用于一致性检验，衡量分类精度
    mask1.tolist(), mask2.tolist()
    img, target = np.array(mask1).flatten(), np.array(mask2).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa


def func_position_metrics(l_label, l_predict):
    l_label.append(0)
    l_predict.append(0)
    l_label = np.array(l_label)
    l_predict = np.array(l_predict)
    l_label = l_label.flatten()
    l_predict = l_predict.flatten()
    l_label = label_binarize(l_label, classes=[i for i in range(2)])
    l_predict = label_binarize(l_predict, classes=[i for i in range(2)])
    m_precision_ = precision_score(l_label, l_predict, average='weighted')
    m_recall_ = recall_score(l_label, l_predict, average='weighted')
    m_f1_ = f1_score(l_label, l_predict, average='weighted')
    m_acc_ = accuracy_score(l_label, l_predict)
    m_auc_ = roc_auc_score(l_label, l_predict, multi_class='ovo')
    tn, fp, fn, tp = confusion_matrix(l_label, l_predict).ravel()
    m_vs_ = 1 - (abs(fn - fp) / (2 * tp + fp + fn))
    return m_recall_, m_f1_, m_acc_, m_auc_, m_vs_


def func_maximumdiameter(MaxDiameter):
    if MaxDiameter >= 0.5 and MaxDiameter <= 1:
        return 0
    elif MaxDiameter > 1 and MaxDiameter <= 2:
        return 1
    elif MaxDiameter > 2 and MaxDiameter <= 3:
        return 2
    elif MaxDiameter > 3 and MaxDiameter <= 5:
        return 3
    elif MaxDiameter > 5:
        return 4
    else:
        return 5


def func_MaximumDiameter_multi_test(path_img_file1, path_lab_file1, path_predict_file1):
    global l0, l1, l2, l3, l4
    # 预测
    predict_nii = nib.load(path_predict_file1)
    predict = predict_nii.get_fdata()
    spacing = predict_nii.header.get_zooms()
    spa_ = np.max([spacing[0], spacing[1]])
    img_nii = nib.load(path_img_file1)
    sitk_img = sitk.ReadImage(path_img_file1)
    img = img_nii.get_data()
    # 标签
    mask_nii = nib.load(path_lab_file1)
    mask = mask_nii.get_data()
    mask[img == 0] = 0
    [data_tumor_label, num] = skimage.measure.label(mask, connectivity=1, return_num=True)
    affine = nib.load(path_lab_file1).affine
    # 先循环标签的每个占位
    for i in range(num):
        inter = 0
        volume = np.sum(data_tumor_label == i + 1)
        dat_t = np.zeros_like(data_tumor_label)
        dat_t[data_tumor_label == i + 1] = 1
        if volume >= 5:
            new_nii = nib.Nifti1Image(dat_t, affine)
            nib.save(new_nii, 'tumor053miss.nii.gz')
            lab = sitk.ReadImage('tumor053miss.nii.gz')
            radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=sitk_img, inputMask=lab)
            MaximumDiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
        else:
            MaximumDiameter = volume * spa_ / 10
        maxx = func_maximumdiameter(MaximumDiameter)
        # 计算每个占位与预测有无交集
        predict[predict>1] = 1
        ecide = (predict * dat_t).astype(int)
        if 1 in np.unique(ecide):
            inter = 1
        if inter == 0:
            missing_diameter_list.append(maxx)
            li = np.where(dat_t == 1)
            # 值的类型
            missing_class_list.append(mask[li[0][0], li[1][0], li[2][0]])
            missing_info_list.append([path_predict_file1.split('/')[-2], maxx, mask[li[0][0], li[1][0], li[2][0]]])
            print('没有交集', path_predict_file1.split('/')[-2], maxx, mask[li[0][0], li[1][0], li[2][0]])
            if path_predict_file1 not in missing_name_list:
                missing_name_list.append(path_predict_file1)

# path_文件夹下需要有delay.nii.gz，lesion_corrected.nii.gz，liver_corrected.nii.gz，predict.nii.gz
path_ = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/for_pred/resize_testing/'

files = os.listdir(path_)
# files.remove('plans.pkl')
print(len(files))
for file in files:
    path_img_file = os.path.join(path_, file, 'delay.nii.gz')
    path_lesion_corrected = os.path.join(path_, file, 'lesion_corrected.nii.gz')
    path_predict_file = os.path.join(path_, file, 'predict.nii.gz')
    print(file)
    func_MaximumDiameter_multi_test(path_img_file, path_lesion_corrected, path_predict_file)


print('missing_diameter_list', missing_diameter_list)
print('missing_class_list', missing_class_list)
print('missing_info_list', missing_info_list)
print('missing_name_list', missing_name_list)

# 结果

#################### 长径:  0.5-1, 1-2, 2-3， 3-5, >5
#################### 类别: 肝细胞肝癌HCC，肝脏转移瘤，胆管细胞癌，肝囊肿，肝血管瘤
###################### 文件名, 长径, 类别

missing_info_list = []
missing_diameter_list = []
missing_class_list = []
