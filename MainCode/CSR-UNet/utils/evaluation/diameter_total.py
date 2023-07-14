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

### 计算统计预测结果和标签在肿瘤所有数据的分割指标和检测指标

################## Computing Dice###################################
from radiomics import featureextractor


############################计算dice指标############################
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



# path_文件夹下需要有lesion_corrected.nii.gz，predict.nii.gz
path_ = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/for_pred/resize_testing/'
files = os.listdir(path_)
print(len(files))
m_dice, m_iou, m_acc, m_recall, m_kappa, m_f1, m_precision, m_auc, m_hd, m_vs = [], [], [], [], [], [], [], [], [], []

for file in files:
    print(file)
    path_lab_file = os.path.join(path_, file, 'lesion_corrected.nii.gz')
    path_predict_file = os.path.join(path_, file, 'predict.nii.gz')
    label = nib.load(path_lab_file).get_data()
    predict = nib.load(path_predict_file).get_data()
    spacing = nib.load(path_predict_file).header.get_zooms()
    label[label>0]=1
    predict[predict>0]=1
    m_dice.append(func_dice(label, predict))
    m_iou.append(func_iou(label, predict))
    m_kappa.append(func_kappa(label, predict))
    surf_dists = surfdist.compute_surface_distances(label > 0, predict > 0, spacing)
    m_hd_ = surfdist.compute_robust_hausdorff(surf_dists, 95)
    if np.isinf(float(m_hd_)):
        m_hd_ = 189
    m_hd.append(m_hd_)
    data_tumor_label, numbers = measure.label(label, return_num=True, connectivity=1)
    l_label = []
    l_predict = []
    for i in range(1, numbers + 1):
        dat = data_tumor_label == i
        decide = (dat * predict).astype(int)
        if len(np.unique(decide)) == 2:
            l_predict.append(1)
        else:
            l_predict.append(0)
        l_label.append(1)
    l_label.append(0)
    l_predict.append(0)
    l_label = np.array(l_label)
    l_predict = np.array(l_predict)
    l_label = l_label.flatten()
    l_predict = l_predict.flatten()
    l_label = label_binarize(l_label, classes=[i for i in range(2)])
    l_predict = label_binarize(l_predict, classes=[i for i in range(2)])
    m_precision.append(precision_score(l_label, l_predict, average='weighted'))
    m_recall.append(recall_score(l_label, l_predict, average='weighted'))
    m_f1.append(f1_score(l_label, l_predict, average='weighted'))
    m_acc.append(accuracy_score(l_label, l_predict))
    m_auc.append(roc_auc_score(l_label, l_predict, multi_class='ovo'))
    tn, fp, fn, tp = confusion_matrix(l_label, l_predict).ravel()
    m_vs_ = 1 - (abs(fn - fp) / (2 * tp + fp + fn))
    m_vs.append(m_vs_)
    print(func_dice(label, predict), func_iou(label, predict), func_kappa(label, predict), m_hd_,
          precision_score(l_label, l_predict, average='weighted'), recall_score(l_label, l_predict, average='weighted'),
          f1_score(l_label, l_predict, average='weighted'), accuracy_score(l_label, l_predict),
          roc_auc_score(l_label, l_predict, multi_class='ovo'), m_vs_)

mean_dice = np.mean(m_dice)
s_dice = np.std(m_dice)
mean_iou = np.mean(m_iou)
s_iou = np.std(m_iou)
mean_kappa = np.mean(m_kappa)
s_kappa = np.std(m_kappa)
mean_precision = np.mean(m_precision)
s_precision = np.std(m_precision)
mean_recall = np.mean(m_recall)
s_recall = np.std(m_recall)
mean_f1 = np.mean(m_f1)
s_f1 = np.std(m_f1)
mean_acc = np.mean(m_acc)
s_acc = np.std(m_acc)
mean_auc = np.mean(m_auc)
s_auc = np.std(m_auc)
mean_hd = np.mean(m_hd)
s_hd = np.std(m_hd)
mean_vs = np.mean(m_vs)
s_vs = np.std(m_vs)

print(
    'mean:  Dice: %.3f, IOU: %.3f, Kappa: %.3f, Precision: %.3f, Recall: %.3f, F1-score: %.3f, Acc: %.3f, AUC: %.3f, HD: %.3f, VS: %.3f'
    % (mean_dice, mean_iou, mean_kappa, mean_precision, mean_recall, mean_f1, mean_acc, mean_auc, mean_hd, mean_vs))
print(
    'standard:  Dice: %.3f, IOU: %.3f, Kappa: %.3f, Precision: %.3f, Recall: %.3f, F1-score: %.3f, Acc: %.3f, AUC: %.3f, HD: %.3f, VS: %.3f'
    % (s_dice, s_iou, s_kappa, s_precision, s_recall, s_f1, s_acc, s_auc, s_hd, s_vs))
