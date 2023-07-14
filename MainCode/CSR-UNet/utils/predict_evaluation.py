import os
import nibabel as nib
import numpy as np
from numpy.core.fromnumeric import mean
import torch
import skimage.morphology
from  skimage import measure
import mitk
from mitk import npy_regionprops_denoise
from torch.autograd.grad_mode import F
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize

def calculate(inputs, targets):
    """
    return:
    TP, FP, FN, TN
    """
    TP = (inputs * targets)
    FP = ((1 - targets) * inputs)
    FN = (targets * (1 - inputs))
    TN = (inputs * (1 - inputs))
    return TP, FP, FN, TN


def dice_equation(mask1, mask2):
    inter = float(np.sum((mask1 > 0) * (mask2 > 0)))
    summ = float(np.sum(mask1 > 0) + np.sum(mask2 > 0))
    if summ == 0:
        return 0.0
    dice = 2 * inter / summ
    return dice


def iou_equation(mask1, mask2):
    inter = float(np.sum((mask1 > 0) * (mask2 > 0)))
    summ = float(np.sum(mask1 > 0) + np.sum(mask2 > 0))
    if (summ - inter) == 0:
        return 0.0
    iou = inter / (summ - inter)
    return iou

def func_acc(mask1, mask2):
    matrix = confusion_matrix(y_true=np.array(mask1).flatten(),y_pred=np.array(mask2).flatten())
    acc = np.diag(matrix).sum()/matrix.sum()
    return acc

def func_recall(mask1, mask2):
    matrix = confusion_matrix(y_true=np.array(mask1).flatten(),y_pred=np.array(mask2).flatten())
    recall = np.diag(matrix)/matrix.sum(axis=0)
    return recall

def func_kappa(mask1, mask2):
    mask1.tolist(),mask2.tolist()
    img, target= np.array(mask1).flatten(), np.array(mask2).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa

def cal_case_(pred_path, ground_path):
    ground = nib.load(ground_path)
    reorient_ground_nii = mitk.nii_reorientation(ground, end_ornt=["L", "A", "S"])
    ground_data = reorient_ground_nii.get_data().astype('int64')
    ground_data = np.array(ground_data)
    pred = nib.load(pred_path)
    reorient_pred_nii = mitk.nii_reorientation(pred, end_ornt=["L", "A", "S"])
    pred_data = reorient_pred_nii.get_data().astype('int64')
    pred_data = np.array(pred_data)
    if pred_data.shape == ground_data.shape:
        dice1 = dice_equation(ground_data, pred_data)
        iou = iou_equation(ground_data, pred_data)
        acc = func_acc(ground_data, pred_data)
        recall = func_recall(ground_data, pred_data)
        kappa = func_kappa(ground_data, pred_data)
        print('DICE:', dice1, ', iou:', iou, ', acc', acc, ', recall', recall, ', kappa', kappa)
        return dice1, iou, acc, recall, kappa


# tumor
# path_gt_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task70_TumorSeg/labelsTs/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task70_TumorSeg/pred_tumor_25D/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task70_TumorSeg/predict_nnunet_3d_fullres_adam_reduce_dicece/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task70_TumorSeg/predict_nnunet_3d_fullres_sgd_focalloss/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task70_TumorSeg/predict_tumor_3D/'

# liver
# path_gt_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task68_Liver/labelsTs/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task68_Liver/predict_liver_3D/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task68_Liver/predict_nnunet_3d_fullres_adam_reduce_dicece/'
# path_pred_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task68_Liver/predict_nnunet_3d_fullres_sgd_focalloss/'


# ##### liver_profile
# path_gt_nii = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task69_LiverSeg/labelsTs/'
# # path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_profile_testing/'
# # path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_profile_testing20/'
# # path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_profile_testing20_epoch162/'
# # path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_profile_testing20_epoch189/'
# path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_profile_testing20_epoch249/'

##### liver_couinaud
# path_gt_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/sort/liver_couinaud/labelsTs/'
# path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_couinaud_testing20_338/'
# path_pred_nii = '/home3/HWGroup/wushu/liver_tumor/data/v1.0.0/pred/pred_liver_couinaud_testing20_448/'

# dice_list = []
# iou_list = []
# acc_list = []
# recall_list = []
# kappa_list = []

# for pat_name in os.listdir(path_gt_nii):
#     print(pat_name)
#     model_path = os.path.join(path_pred_nii, pat_name,)
#     label_path = os.path.join(path_gt_nii, pat_name)
#     if os.path.exists(model_path) and os.path.exists(label_path):
#         dice, iou, acc, recall, kappa = cal_case_(model_path, label_path)
#         if dice != None:
#             dice_list.append(dice)
#             iou_list.append(iou)
#             acc_list.append(acc)
#             recall_list.append(recall)
#             kappa_list.append(kappa)

# dice_array = np.array(dice_list)
# print('mean dice:', np.mean(dice_array))
# print('min dice:', np.min(dice_array))
# print('max dice:', np.max(dice_array))
# print('std dice:', np.std(dice_array))

# iou_array = np.array(iou_list)
# print('mean iou:', np.mean(iou_array))
# print('min iou:', np.min(iou_array))
# print('max iou:', np.max(iou_array))
# print('std iou:', np.std(iou_array))

# acc_array = np.array(acc_list)
# print('mean acc:', np.mean(acc_array))
# print('min acc:', np.min(acc_array))
# print('max acc:', np.max(acc_array))
# print('std acc:', np.std(acc_array))

# recall_array = np.array(recall_list)
# print('mean recall:', np.mean(recall_array))
# print('min recall:', np.min(recall_array))
# print('max recall:', np.max(recall_array))
# print('std recall:', np.std(recall_array))

# kappa_array = np.array(kappa_list)
# print('mean kappa:', np.mean(kappa_array))
# print('min kappa:', np.min(kappa_array))
# print('max kappa:', np.max(kappa_array))
# print('std kappa:', np.std(kappa_array))


########################################################################3

# 计算目标检测指标
def cal_case_detection(pred_path, ground_path):
    ground = nib.load(ground_path)
    reorient_ground_nii = mitk.nii_reorientation(ground, end_ornt=["L", "A", "S"])
    ground_data = reorient_ground_nii.get_data()
    print(np.unique(ground_data))
    ground_data[ground_data>=1] = 1
    ground_data = ground_data.astype('uint16')
    pred = nib.load(pred_path)
    reorient_pred_nii = mitk.nii_reorientation(pred, end_ornt=["L", "A", "S"])
    pred_data = reorient_pred_nii.get_data()
    pred_data[pred_data>=1] = 1
    pred_data = pred_data.astype('uint16')
    print(pred_data.shape == ground_data.shape)
    if pred_data.shape == ground_data.shape:
        m_precision, m_recall, m_f1, m_acc, m_auc = multimetrics(ground_data, pred_data)
        print('precision:', m_precision, ', recall:', m_recall, ', f1: ', m_f1, ', acc:', m_acc, ', auc:', m_auc)
        return m_precision, m_recall, m_f1, m_acc, m_auc

def multimetrics(label, predict):
    data_tumor_label,numbers = measure.label(label, return_num = True, connectivity=1)
    l_label = []
    l_predict = []
    for i in range(1,numbers+1):
        dat = data_tumor_label==i
        decide = (dat*predict).astype(int)
        if len(np.unique(decide)) ==2:
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
    m_precision = precision_score(l_label, l_predict, average='weighted')
    m_recall = recall_score(l_label, l_predict, average='weighted')
    m_f1 = f1_score(l_label, l_predict, average='weighted')
    m_acc = accuracy_score(l_label, l_predict)
    m_auc = roc_auc_score(l_label, l_predict, multi_class='ovo')
    return m_precision, m_recall, m_f1, m_acc, m_auc

# 小肝癌
# path_gt_nii = '/home3/HWGroup/wushu/LUNA16/Task115_LiverTumorSmall/raw_splitted/labelsTs/'
# path_pred_nii = '/home3/HWGroup/wushu/LUNA16/models/Task115_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii/'
path_gt_nii = '/home3/HWGroup/wushu/LUNA16/Task117_LiverTumorSmall/raw_splitted/labelsTr_Ts_total/'

# 117
# path_pred_nii = '/home3/HWGroup/wushu/LUNA16/models/Task117_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii/'
# path_pred_nii = '/home3/HWGroup/wushu/LUNA16/models/Task117_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_postprepcess1/'
# 116
# path_pred_nii = '/home3/HWGroup/wushu/LUNA16/models/Task116_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_postprepcess1/'
# 118
path_pred_nii = '/home3/HWGroup/wushu/LUNA16/models/Task118_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_postprepcess1/'
precision_list = []
recall_list = []
f1_list = []
acc_list = []
auc_list = []
for pat_name in os.listdir(path_pred_nii):
    # if pat_name.endswith('nii.gz'):
    #     pat_name_no_box = pat_name.split('_boxes')[0] + '.nii.gz'
    if pat_name.endswith('_postprep.nii.gz'):
        pat_name_no_box = pat_name.split('_postprep')[0] + '.nii.gz'
        print(pat_name, pat_name_no_box)
        model_path = os.path.join(path_pred_nii, pat_name)
        label_path = os.path.join(path_gt_nii, pat_name_no_box)
        print(model_path, label_path)
        if os.path.exists(model_path) and os.path.exists(label_path):
            precision, recall, f1, acc, auc = cal_case_detection(model_path, label_path)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            acc_list.append(acc)
            auc_list.append(auc)
precision_array = np.array(precision_list)
print('mean precision:', np.mean(precision_array))
print('std precision:', np.std(precision_array))
recall_array = np.array(recall_list)
print('mean recall:', np.mean(recall_array))
print('std recall:', np.std(recall_array))
f1_array = np.array(f1_list)
print('mean f1:', np.mean(f1_array))
print('std f1:', np.std(f1_array))
acc_array = np.array(acc_list)
print('mean acc:', np.mean(acc_array))
print('std acc:', np.std(acc_array))
auc_array = np.array(auc_list)
print('mean auc:', np.mean(auc_array))
print('std auc:', np.std(auc_array))