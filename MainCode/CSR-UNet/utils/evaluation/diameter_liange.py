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

### 计算统计预测结果和标签在良性和恶性分类下对应的分割指标和检测指标，并且统计假阳性数据和对应长径列表

################## Computing Dice###################################
from radiomics import featureextractor

################
mp_dice1, mp_dice2 = [], []
mp_iou1, mp_iou2 = [], []
mp_kappa1, mp_kappa2 = [], []
mp_precision1, mp_precision2 = [], []
mp_recall1, mp_recall2 = [], []
mp_f11, mp_f12 = [], []
mp_acc1, mp_acc2 = [], []
mp_auc1, mp_auc2 = [], []
mp_hd1, mp_hd2 = [], [] 
mp_vs1, mp_vs2 = [], []
l_label1, l_label2 = [], []
l_predict1, l_predict2 = [], []
################## Computing metrics###################################

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
    mask1.tolist(),mask2.tolist()
    img, target= np.array(mask1).flatten(), np.array(mask2).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa

def multimetrics(label_, predict_, spacing_, mark_):
    [data_tumor_label, nums] = skimage.measure.label(label_, connectivity=1, return_num=True)
    zero_1, zero_2 = 0,0
    for i in range(nums):
        inter = 0
        dat_tl = data_tumor_label == i+1
        data_predict,ns = measure.label(predict_, return_num = True, connectivity=1)
        for i in range(ns):
            dat_tp = data_predict == i+1
            ecide = (dat_tp*dat_tl).astype(int)
            if len(np.unique(ecide)) == 2:
                inter=1
                m_dice_ = func_dice(dat_tl, dat_tp)
                m_iou_ = func_iou(dat_tl, dat_tp)
                m_kappa_ = func_kappa(dat_tl, dat_tp)
                surf_dists = surfdist.compute_surface_distances(dat_tl > 0, dat_tp > 0, spacing_)
                m_hd_ = surfdist.compute_robust_hausdorff(surf_dists, 95)
                if np.isinf(float(m_hd_)):
                    m_hd_ = 189
                print(m_dice_,m_iou_,m_kappa_,m_hd_)
                if mark_ == 1:
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        mp_dice1.append(m_dice_)
                        mp_iou1.append(m_iou_)
                        mp_kappa1.append(m_kappa_) 
                        mp_hd1.append(m_hd_)
                    l_label1.append(1)
                    l_predict1.append(1)
                elif mark_ == 2:
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        mp_dice2.append(m_dice_)
                        mp_iou2.append(m_iou_)
                        mp_kappa2.append(m_kappa_) 
                        mp_hd2.append(m_hd_)
                    l_label2.append(1)
                    l_predict2.append(1)
            else:
                continue
        if inter==0:
            if mark_ == 1:
                zero_1 = zero_1+1
            elif mark_ == 2:
                zero_2 = zero_2+1
    return nums, zero_1, zero_2

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
    m_vs_ = 1-(abs(fn-fp)/(2*tp+fp+fn))
    return m_recall_, m_f1_, m_acc_, m_auc_, m_vs_

# path_文件夹下需要有lesion_corrected.nii.gz，liver_corrected.nii.gz，predict.nii.gz
path_ = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/for_pred/resize_testing/'
files = os.listdir(path_)
if 'plans.pkl' in files:
    files.remove('plans.pkl')
l1,l2,l3,l4,l5 = 0,0,0,0,0

for file in files:
    path_file = os.path.join(path_, file)
    path_predict_file = os.path.join(path_, file, 'predict.nii.gz')
    predict = nib.load(path_predict_file).get_data()
    spacing = nib.load(path_predict_file).header.get_zooms()
    liver = nib.load(os.path.join(path_file, 'liver_corrected.nii.gz')).get_data()
    dat = nib.load(os.path.join(path_file, 'lesion_corrected.nii.gz')).get_data()
    dat[liver==0] = 0
    lists = np.unique(dat)[1:]
    print(lists,file)
    mask1 = np.zeros_like(dat)   # 恶性
    mask2 = np.zeros_like(dat)   # 良性
    l_label1, l_predict1, l_label2, l_predict2 = [], [], [], []
    for ii in lists:
        if ii in [1,2,3]:
            mask1[dat==ii] = 1
        else:
            mask2[dat==ii] = 1
    zero_11_, zero_22_ = 0,0
    if np.sum(mask1) >0:
        nums, zero_11, zero_22 = multimetrics(mask1, predict, spacing, 1)

        zero_11_ = zero_11_ + zero_11
        zero_22_ = zero_22_ + zero_22
        l1 = l1+nums
    if np.sum(mask2) >0:
        nums, zero_11, zero_22 = multimetrics(mask2, predict, spacing, 2)
        
        zero_11_ = zero_11_ + zero_11
        zero_22_ = zero_22_ + zero_22
        l2 = l2+nums

    if zero_11_ >0:
        mp_dice1.append(0)
        mp_iou1.append(0)
        mp_kappa1.append(0)
        mp_hd1.append(189)
        l_label1.append(1)
        l_predict1.append(0)
    if zero_22_ >0:
        mp_dice2.append(0)
        mp_iou2.append(0)
        mp_kappa2.append(0)
        mp_hd2.append(189)
        l_label2.append(1)
        l_predict2.append(0)

    if l_label1 != [] and l_predict1 != []:
        m_recall1, m_f11, m_acc1, m_auc1, m_vs1 = func_position_metrics(l_label1, l_predict1)
        mp_recall1.append(m_recall1)
        mp_f11.append(m_f11)
        mp_acc1.append(m_acc1)
        mp_auc1.append(m_auc1)
        mp_vs1.append(m_vs1)
    if l_label2 != [] and l_predict2 != []:
        m_recall2, m_f12, m_acc2, m_auc2, m_vs2 = func_position_metrics(l_label2, l_predict2)
        mp_recall2.append(m_recall2)
        mp_f12.append(m_f12)
        mp_acc2.append(m_acc2)
        mp_auc2.append(m_auc2)
        mp_vs2.append(m_vs2)
    print(l_label1, l_predict1, l_label2, l_predict2)

print(l1,l2)
print("################# 恶性 #########################")
print(np.mean(mp_dice1),np.std(mp_dice1),np.mean(mp_iou1),np.std(mp_iou1),np.mean(mp_kappa1),np.std(mp_kappa1))
print("################# 良性 #########################")
print(np.mean(mp_dice2),np.std(mp_dice2),np.mean(mp_iou2),np.std(mp_iou2),np.mean(mp_kappa2),np.std(mp_kappa2))

print(mp_recall1)
print(mp_recall2)
print("################# 恶性 #########################")
print(np.mean(mp_recall1),np.std(mp_recall1),np.mean(mp_f11),np.std(mp_f11),np.mean(mp_acc1),np.std(mp_acc1),np.mean(mp_auc1),np.std(mp_auc1),np.mean(mp_hd1),np.std(mp_hd1),np.mean(mp_vs1),np.std(mp_vs1))
print("################# 良性 #########################")
print(np.mean(mp_recall2),np.std(mp_recall2),np.mean(mp_f12),np.std(mp_f12),np.mean(mp_acc2),np.std(mp_acc2),np.mean(mp_auc2),np.std(mp_auc2),np.mean(mp_hd2),np.std(mp_hd2),np.mean(mp_vs2),np.std(mp_vs2))

import xlwt
task = 'liange'
method = '053_143'
path_save_excel = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/{}_{}.xls'.format(task, method)
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet(task)
worksheet.write(0, 0, label = '指标')
worksheet.write(0, 1, label = '恶性')
worksheet.write(0, 2, label = '良性')
LISTs = ['Dice','IOU','Recall','f1-measure','Accuracy','AUC','HD','VS']
Value1 = [mp_dice1, mp_iou1, mp_recall1, mp_f11, mp_acc1, mp_auc1, mp_hd1, mp_vs1]
Value2 = [mp_dice2, mp_iou2, mp_recall2, mp_f12, mp_acc2, mp_auc2, mp_hd2, mp_vs2]
for i in range(1,9):
    worksheet.write(i, 0, label = LISTs[i-1])
    worksheet.write(i, 1, label = str(round(np.mean(Value1[i-1]),3))+'±'+str(round(np.std(Value1[i-1]),3)))
    worksheet.write(i, 2, label = str(round(np.mean(Value2[i-1]),3))+'±'+str(round(np.std(Value2[i-1]),3)))
workbook.save(path_save_excel)
