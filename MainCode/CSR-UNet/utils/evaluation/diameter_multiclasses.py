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

### 计算统计预测结果和标签在肿瘤不同分类下对应的分割指标和检测指标

################## Computing Dice###################################
from radiomics import featureextractor

###################
mp_dice1, mp_dice2, mp_dice3, mp_dice4, mp_dice5 = [], [], [], [], []
mp_iou1, mp_iou2, mp_iou3, mp_iou4, mp_iou5 = [], [], [], [], []
mp_kappa1, mp_kappa2, mp_kappa3, mp_kappa4, mp_kappa5 = [], [], [], [], []
mp_precision1, mp_precision2, mp_precision3, mp_precision4, mp_precision5 = [], [], [], [], []
mp_recall1, mp_recall2, mp_recall3, mp_recall4, mp_recall5 = [], [], [], [], []
mp_f11, mp_f12, mp_f13, mp_f14, mp_f15 = [], [], [], [], []
mp_acc1, mp_acc2, mp_acc3, mp_acc4, mp_acc5 = [], [], [], [], []
mp_auc1, mp_auc2, mp_auc3, mp_auc4, mp_auc5 = [], [], [], [], []
mp_hd1, mp_hd2, mp_hd3, mp_hd4, mp_hd5 = [], [], [], [], [] 
mp_vs1, mp_vs2, mp_vs3, mp_vs4, mp_vs5 = [], [], [], [], [] 
l_label1, l_label2, l_label3, l_label4, l_label5 = [], [], [], [], [] 
l_predict1, l_predict2, l_predict3, l_predict4, l_predict5 = [], [], [], [], [] 
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
    zero_1, zero_2, zero_3, zero_4, zero_5 = 0,0,0,0,0
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
                if mark_ == 1:
                    mp_dice1.append(m_dice_)
                    mp_iou1.append(m_iou_)
                    mp_kappa1.append(m_kappa_) 
                    mp_hd1.append(m_hd_)
                    l_label1.append(1)
                    l_predict1.append(1)
                elif mark_ == 2:
                    mp_dice2.append(m_dice_)
                    mp_iou2.append(m_iou_)
                    mp_kappa2.append(m_kappa_) 
                    mp_hd2.append(m_hd_)
                    l_label2.append(1)
                    l_predict2.append(1)
                elif mark_ == 3:
                    mp_dice3.append(m_dice_)
                    mp_iou3.append(m_iou_)
                    mp_kappa3.append(m_kappa_) 
                    mp_hd3.append(m_hd_)
                    l_label3.append(1)
                    l_predict3.append(1)
                elif mark_ == 4:
                    mp_dice4.append(m_dice_)
                    mp_iou4.append(m_iou_)
                    mp_kappa4.append(m_kappa_) 
                    mp_hd4.append(m_hd_)
                    l_label4.append(1)
                    l_predict4.append(1)
                elif mark_ == 5:
                    mp_dice5.append(m_dice_)
                    mp_iou5.append(m_iou_)
                    mp_kappa5.append(m_kappa_) 
                    mp_hd5.append(m_hd_)
                    l_label5.append(1)
                    l_predict5.append(1)
            else:
                continue
        if inter==0:
            if mark_ == 1:
                zero_1 = zero_1+1
            elif mark_ == 2:
                zero_2 = zero_2+1
            elif mark_ == 3:
                zero_3 = zero_3+1
            elif mark_ == 4:
                zero_4 = zero_4+1
            elif mark_ == 5:
                zero_5 = zero_5+1
    return nums, zero_1, zero_2, zero_3, zero_4, zero_5

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
    path_file = os.path.join(path_, file.split('.')[0])
    path_predict_file = os.path.join(path_, file, 'predict.nii.gz')
    predict = nib.load(path_predict_file).get_data()
    spacing = nib.load(path_predict_file).header.get_zooms()
    liver = nib.load(os.path.join(path_file, 'liver_corrected.nii.gz')).get_data()
    dat = nib.load(os.path.join(path_file, 'lesion_corrected.nii.gz')).get_data()
    dat[liver==0] = 0
    lists = np.unique(dat)
    print(lists,file)
    zero_11_, zero_22_, zero_33_, zero_44_, zero_55_ = 0,0,0,0,0
    l_label1, l_predict1, l_label2, l_predict2, l_label3, l_predict3, l_label4, l_predict4, l_label5, l_predict5 = [], [], [], [], [], [], [], [], [], []
    for ii in lists:
        if ii==0:
            pass
        else:
            mask = np.zeros_like(dat)
            mask[dat==ii]=1
            # nums, zero_11, zero_22, zero_33, zero_44, zero_55 = multimetrics(mask, predict, spacing, ii)
            
            
            [data_tumor_label, nums] = skimage.measure.label(mask, connectivity=1, return_num=True)
            zero_1, zero_2, zero_3, zero_4, zero_5 = 0,0,0,0,0
            for i in range(nums):
                inter = 0
                dat_tl = data_tumor_label == i+1
                data_predict,ns = measure.label(predict, return_num = True, connectivity=1)
                for i in range(ns):
                    dat_tp = data_predict == i+1
                    ecide = (dat_tp*dat_tl).astype(int)
                    if len(np.unique(ecide)) == 2:
                        inter=1
                        m_dice_ = func_dice(dat_tl, dat_tp)
                        m_iou_ = func_iou(dat_tl, dat_tp)
                        m_kappa_ = func_kappa(dat_tl, dat_tp)
                        surf_dists = surfdist.compute_surface_distances(dat_tl > 0, dat_tp > 0, spacing)
                        m_hd_ = surfdist.compute_robust_hausdorff(surf_dists, 95)
                        if np.isinf(float(m_hd_)):
                            m_hd_ = 189
                        print(m_dice_,m_iou_,m_kappa_,m_hd_)
                        if ii == 1:
                            if m_dice_ > 0.1:
                                mp_dice1.append(m_dice_)
                                mp_iou1.append(m_iou_)
                                mp_kappa1.append(m_kappa_)
                                mp_hd1.append(m_hd_)
                            l_label1.append(1)
                            l_predict1.append(1)
                        elif ii == 2:
                            if m_dice_ > 0.1:
                                mp_dice2.append(m_dice_)
                                mp_iou2.append(m_iou_)
                                mp_kappa2.append(m_kappa_)
                                mp_hd2.append(m_hd_)
                            l_label2.append(1)
                            l_predict2.append(1)
                        elif ii == 3:
                            if m_dice_ > 0.1:
                                mp_dice3.append(m_dice_)
                                mp_iou3.append(m_iou_)
                                mp_kappa3.append(m_kappa_)
                                mp_hd3.append(m_hd_)
                            l_label3.append(1)
                            l_predict3.append(1)
                        elif ii == 4:
                            if m_dice_ > 0.1:
                                mp_dice4.append(m_dice_)
                                mp_iou4.append(m_iou_)
                                mp_kappa4.append(m_kappa_)
                                mp_hd4.append(m_hd_)
                            l_label4.append(1)
                            l_predict4.append(1)
                        elif ii == 5:
                            if m_dice_ > 0.1:
                                mp_dice5.append(m_dice_)
                                mp_iou5.append(m_iou_)
                                mp_kappa5.append(m_kappa_)
                                mp_hd5.append(m_hd_)
                            l_label5.append(1)
                            l_predict5.append(1)
                    else:
                        continue
                if inter==0:
                    if ii == 1:
                        zero_1 = zero_1+1
                    elif ii == 2:
                        zero_2 = zero_2+1
                    elif ii == 3:
                        zero_3 = zero_3+1
                    elif ii == 4:
                        zero_4 = zero_4+1
                    elif ii == 5:
                        zero_5 = zero_5+1
            
            
            zero_11_ = zero_11_ + zero_1
            zero_22_ = zero_22_ + zero_2
            zero_33_ = zero_33_ + zero_3
            zero_44_ = zero_44_ + zero_4
            zero_55_ = zero_55_ + zero_5
            if ii==1:
                l1 = l1+nums
            elif ii==2:
                l2 = l2+nums
            elif ii==3:
                l3 = l3+nums
            elif ii==4:
                l4 = l4+nums
            elif ii==5:
                l5 = l5+nums
    if zero_11_>0:
        mp_dice1.append(0)
        mp_iou1.append(0)
        mp_kappa1.append(0)
        mp_hd1.append(189)
        l_label1.append(1)
        l_predict1.append(0)
    if zero_22_>0:
        mp_dice2.append(0)
        mp_iou2.append(0)
        mp_kappa2.append(0)
        mp_hd2.append(189)
        l_label2.append(1)
        l_predict2.append(0)
    if zero_33_ >0:
        mp_dice3.append(0)
        mp_iou3.append(0)
        mp_kappa3.append(0)
        mp_hd3.append(189)
        l_label3.append(1)
        l_predict3.append(0)
    if zero_44_ >0:
        mp_dice4.append(0)
        mp_iou4.append(0)
        mp_kappa4.append(0)
        mp_hd4.append(189)
        l_label4.append(1)
        l_predict4.append(0)
    if zero_55_>0:
        mp_dice5.append(0)
        mp_iou5.append(0)
        mp_kappa5.append(0)
        mp_hd5.append(189)
        l_label5.append(1)
        l_predict5.append(0)

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
    if l_label3 != [] and l_predict3 != []:
        m_recall3, m_f13, m_acc3, m_auc3, m_vs3 = func_position_metrics(l_label3, l_predict3)
        mp_recall3.append(m_recall3)
        mp_f13.append(m_f13)
        mp_acc3.append(m_acc3)
        mp_auc3.append(m_auc3)
        mp_vs3.append(m_vs3)
    if l_label4 != [] and l_predict4 != []:
        m_recall4, m_f14, m_acc4, m_auc4, m_vs4 = func_position_metrics(l_label4, l_predict4)
        mp_recall4.append(m_recall4)
        mp_f14.append(m_f14)
        mp_acc4.append(m_acc4)
        mp_auc4.append(m_auc4)
        mp_vs4.append(m_vs4)
    if l_label5 != [] and l_predict5 != []:
        m_recall5, m_f15, m_acc5, m_auc5, m_vs5 = func_position_metrics(l_label5, l_predict5)
        mp_recall5.append(m_recall5)
        mp_f15.append(m_f15)
        mp_acc5.append(m_acc5)
        mp_auc5.append(m_auc5)
        mp_vs5.append(m_vs5)

print(mp_recall1)
print(mp_recall2)
print(mp_recall3)
print(mp_recall4)
print(mp_recall5)
print(l1,l2,l3,l4,l5)
print("################# label1 #########################")
print(np.mean(mp_dice1),np.std(mp_dice1),np.mean(mp_iou1),np.std(mp_iou1),np.mean(mp_kappa1),np.std(mp_kappa1))
print("################# label2 #########################")
print(np.mean(mp_dice2),np.std(mp_dice2),np.mean(mp_iou2),np.std(mp_iou2),np.mean(mp_kappa2),np.std(mp_kappa2))
print("################# label3 #########################")
print(np.mean(mp_dice3),np.std(mp_dice3),np.mean(mp_iou3),np.std(mp_iou3),np.mean(mp_kappa3),np.std(mp_kappa3))
print("################# label4 #########################")
print(np.mean(mp_dice4),np.std(mp_dice4),np.mean(mp_iou4),np.std(mp_iou4),np.mean(mp_kappa4),np.std(mp_kappa4))
print("################# label5 #########################")
print(np.mean(mp_dice5),np.std(mp_dice5),np.mean(mp_iou5),np.std(mp_iou5),np.mean(mp_kappa5),np.std(mp_kappa5))

print("################# label1 #########################")
print(np.mean(mp_recall1),np.std(mp_recall1),np.mean(mp_f11),np.std(mp_f11),np.mean(mp_acc1),np.std(mp_acc1),np.mean(mp_auc1),np.std(mp_auc1),np.mean(mp_hd1),np.std(mp_hd1),np.mean(mp_vs1),np.std(mp_vs1))
print("################# label2 #########################")
print(np.mean(mp_recall2),np.std(mp_recall2),np.mean(mp_f12),np.std(mp_f12),np.mean(mp_acc2),np.std(mp_acc2),np.mean(mp_auc2),np.std(mp_auc2),np.mean(mp_hd2),np.std(mp_hd2),np.mean(mp_vs2),np.std(mp_vs2))
print("################# label3 #########################")
print(np.mean(mp_recall3),np.std(mp_recall3),np.mean(mp_f13),np.std(mp_f13),np.mean(mp_acc3),np.std(mp_acc3),np.mean(mp_auc3),np.std(mp_auc3),np.mean(mp_hd3),np.std(mp_hd3),np.mean(mp_vs3),np.std(mp_vs3))
print("################# label4 #########################")
print(np.mean(mp_recall4),np.std(mp_recall4),np.mean(mp_f14),np.std(mp_f14),np.mean(mp_acc4),np.std(mp_acc4),np.mean(mp_auc4),np.std(mp_auc4),np.mean(mp_hd4),np.std(mp_hd4),np.mean(mp_vs4),np.std(mp_vs4))
print("################# label5 #########################")
print(np.mean(mp_recall5),np.std(mp_recall5),np.mean(mp_f15),np.std(mp_f15),np.mean(mp_acc5),np.std(mp_acc5),np.mean(mp_auc5),np.std(mp_auc5),np.mean(mp_hd5),np.std(mp_hd5),np.mean(mp_vs5),np.std(mp_vs5))

import xlwt
task = '5classes'
method = '053_143'
path_save_excel = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/{}_{}.xls'.format(task, method)
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet(task)
worksheet.write(0, 0, label = '指标')
worksheet.write(0, 1, label = '1肝细胞肝癌HCC')
worksheet.write(0, 2, label = '2肝脏转移瘤')
worksheet.write(0, 3, label = '3胆管细胞癌ICC')
worksheet.write(0, 4, label = '4肝囊肿')
worksheet.write(0, 5, label = '5肝血管瘤')
LISTs = ['Dice','IOU','Recall','f1-measure','Accuracy','AUC','HD','VS']
Value1 = [mp_dice1, mp_iou1, mp_recall1, mp_f11, mp_acc1, mp_auc1, mp_hd1, mp_vs1]
Value2 = [mp_dice2, mp_iou2, mp_recall2, mp_f12, mp_acc2, mp_auc2, mp_hd2, mp_vs2]
Value3 = [mp_dice3, mp_iou3, mp_recall3, mp_f13, mp_acc3, mp_auc3, mp_hd3, mp_vs3]
Value4 = [mp_dice4, mp_iou4, mp_recall4, mp_f14, mp_acc4, mp_auc4, mp_hd4, mp_vs4]
Value5 = [mp_dice5, mp_iou5, mp_recall5, mp_f15, mp_acc5, mp_auc5, mp_hd5, mp_vs5]
for i in range(1,9):
    worksheet.write(i, 0, label = LISTs[i-1])
    worksheet.write(i, 1, label = str(round(np.mean(Value1[i-1]),3))+'±'+str(round(np.std(Value1[i-1]),3)))
    worksheet.write(i, 2, label = str(round(np.mean(Value2[i-1]),3))+'±'+str(round(np.std(Value2[i-1]),3)))
    worksheet.write(i, 3, label = str(round(np.mean(Value3[i-1]),3))+'±'+str(round(np.std(Value3[i-1]),3)))
    worksheet.write(i, 4, label = str(round(np.mean(Value4[i-1]),3))+'±'+str(round(np.std(Value4[i-1]),3)))
    worksheet.write(i, 5, label = str(round(np.mean(Value5[i-1]),3))+'±'+str(round(np.std(Value5[i-1]),3)))
workbook.save(path_save_excel)