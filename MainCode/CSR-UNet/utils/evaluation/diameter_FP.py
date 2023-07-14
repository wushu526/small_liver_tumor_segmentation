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

### 计算统计预测结果和标签在不同长径下对应的分割指标和检测指标，并且统计假阳性数据和对应长径列表

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
FP_diameter_list = []
FP_name_list = []

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
    mask = mask_nii.get_fdata()
    data_predict, ns = measure.label(predict, return_num=True, connectivity=1)
    mask[img == 0] = 0
    [data_tumor_label, num] = skimage.measure.label(mask, connectivity=1, return_num=True)
    affine = nib.load(path_lab_file1).affine
    # 先循环预测的每个占位
    mark_0,mark_1,mark_2,mark_3,mark_4 = 0,0,0,0,0
    l_label0, l_predict0, l_label1, l_predict1, l_label2, l_predict2, l_label3, l_predict3, l_label4, l_predict4 = [], [], [], [], [], [], [], [], [], []
    for i in range(ns):
        inter = 0
        volume = np.sum(data_predict == i + 1)
        dat_tp = np.zeros_like(data_predict)
        dat_tp[data_predict == i + 1] = 1
        # print(volume)
        if volume >= 10:
            new_nii = nib.Nifti1Image(dat_tp, affine)
            nib.save(new_nii, 'tumor053fp.nii.gz')
            lab = sitk.ReadImage('tumor053fp.nii.gz')
            radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=sitk_img, inputMask=lab)
            MaximumDiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
        else:
            MaximumDiameter = volume * spa_ / 10
        maxx = func_maximumdiameter(MaximumDiameter)

        print('MaximumDiameter',MaximumDiameter)
        # 再循环标签的每个占位，有无交集
        for i in range(num):
            dat_t = data_tumor_label == i + 1
            ecide = (dat_tp * dat_t).astype(int)
            if len(np.unique(ecide)) == 2:
                inter = 1
                m_dice_ = func_dice(dat_t, dat_tp)
                m_iou_ = func_iou(dat_t, dat_tp)
                m_kappa_ = func_kappa(dat_t, dat_tp)
                surf_dists = surfdist.compute_surface_distances(dat_t > 0, dat_tp > 0, spacing)
                m_hd_ = surfdist.compute_robust_hausdorff(surf_dists, 95)
                print(m_dice_,m_iou_,m_kappa_,m_hd_)
                if np.isinf(float(m_hd_)):
                    m_hd_ = 189
                if maxx == 0:
                    l_label0.append(1)
                    l_predict0.append(1)
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        l0 = l0 + 1
                        mp_dice0.append(m_dice_)
                        mp_iou0.append(m_iou_)
                        mp_kappa0.append(m_kappa_)
                        mp_hd0.append(m_hd_)
                elif maxx == 1:
                    l_label1.append(1)
                    l_predict1.append(1)
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        l1 = l1 + 1
                        mp_dice1.append(m_dice_)
                        mp_iou1.append(m_iou_)
                        mp_kappa1.append(m_kappa_)
                        mp_hd1.append(m_hd_)
                elif maxx == 2:
                    l_label2.append(1)
                    l_predict2.append(1)
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        l2 = l2 + 1
                        mp_dice2.append(m_dice_)
                        mp_iou2.append(m_iou_)
                        mp_kappa2.append(m_kappa_)
                        mp_hd2.append(m_hd_)
                elif maxx == 3:
                    l_label3.append(1)
                    l_predict3.append(1)
                    if m_dice_ > 0.1: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        l3 = l3 + 1
                        mp_dice3.append(m_dice_)
                        mp_iou3.append(m_iou_)
                        mp_kappa3.append(m_kappa_)
                        mp_hd3.append(m_hd_)
                elif maxx == 4:
                    l_label4.append(1)
                    l_predict4.append(1)
                    if m_dice_ > 0.7: # 可能会有部分小占位与大占位相连的情况，计算出来的dice特别低，需要剔除
                        l4 = l4 + 1
                        mp_dice4.append(m_dice_)
                        mp_iou4.append(m_iou_)
                        mp_kappa4.append(m_kappa_)
                        mp_hd4.append(m_hd_)
            else:
                continue
        if inter == 0:
            FP_diameter_list.append(maxx)
            if path_predict_file1 not in FP_name_list:
                FP_name_list.append(path_predict_file1)
            if maxx == 0:
                l0 = l0 + 1
                mark_0 = mark_0+1
            elif maxx == 1:
                l1 = l1 + 1
                mark_1 = mark_1+1
            elif maxx == 2:
                l2 = l2 + 1
                mark_2 = mark_2+1
            elif maxx == 3:
                l3 = l3 + 1
                mark_3 = mark_3+1
            elif maxx == 4:
                l4 = l4 + 1
                mark_4 = mark_4+1

    if mark_0>0:
        mp_dice0.append(0)
        mp_iou0.append(0)
        mp_kappa0.append(0)
        mp_hd0.append(189)
        l_label0.append(1)
        l_predict0.append(0)
    if mark_1 >0:
        mp_dice1.append(0)
        mp_iou1.append(0)
        mp_kappa1.append(0)
        mp_hd1.append(189)
        l_label1.append(1)
        l_predict1.append(0)
    if mark_2>0:
        mp_dice2.append(0)
        mp_iou2.append(0)
        mp_kappa2.append(0)
        mp_hd2.append(189)
        l_label2.append(1)
        l_predict2.append(0)
    if mark_3 >0:
        mp_dice3.append(0)
        mp_iou3.append(0)
        mp_kappa3.append(0)
        mp_hd3.append(189)
        l_label3.append(1)
        l_predict3.append(0)
    if mark_4 >0:
        mp_dice4.append(0)
        mp_iou4.append(0)
        mp_kappa4.append(0)
        mp_hd4.append(189)
        l_label4.append(1)
        l_predict4.append(0)

    if l_label0 != [] and l_predict0 != []:
        m_recall0, m_f10, m_acc0, m_auc0, m_vs0 = func_position_metrics(l_label0, l_predict0)
        mp_recall0.append(m_recall0)
        mp_f10.append(m_f10)
        mp_acc0.append(m_acc0)
        mp_auc0.append(m_auc0)
        mp_vs0.append(m_vs0)
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

# path_文件夹下需要有原图delay.nii.gz，lesion_corrected.nii.gz，predict.nii.gz
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

print(l0, l1, l2, l3, l4)
np_saving_path = '/home3/HWGroup/wushu/liver_tumor/code/utils/evaluation_gongwei_lcp_202304/053_053_143/'
# save dice
print('mp_dice0',mp_dice0)
np.save(os.path.join(np_saving_path,'mp_dice0.npy'),mp_dice0)
print('mp_dice1',mp_dice1)
np.save(os.path.join(np_saving_path,'mp_dice1.npy'),mp_dice1)
print('mp_dice2',mp_dice2)
np.save(os.path.join(np_saving_path,'mp_dice2.npy'),mp_dice2)
print('mp_dice3',mp_dice3)
np.save(os.path.join(np_saving_path,'mp_dice3.npy'),mp_dice3)
print('mp_dice4',mp_dice4)
np.save(os.path.join(np_saving_path,'mp_dice4.npy'),mp_dice4)
# save iou
print('mp_iou0',mp_iou0)
np.save(os.path.join(np_saving_path,'mp_iou0.npy'),mp_iou0)
print('mp_iou1',mp_iou1)
np.save(os.path.join(np_saving_path,'mp_iou1.npy'),mp_iou1)
print('mp_iou2',mp_iou2)
np.save(os.path.join(np_saving_path,'mp_iou2.npy'),mp_iou2)
print('mp_iou3',mp_iou3)
np.save(os.path.join(np_saving_path,'mp_iou3.npy'),mp_iou3)
print('mp_iou4',mp_iou4)
np.save(os.path.join(np_saving_path,'mp_iou4.npy'),mp_iou4)
# save kappa
print('mp_kappa0',mp_kappa0)
np.save(os.path.join(np_saving_path,'mp_kappa0.npy'),mp_kappa0)
print('mp_kappa1',mp_kappa1)
np.save(os.path.join(np_saving_path,'mp_kappa1.npy'),mp_kappa1)
print('mp_kappa2',mp_kappa2)
np.save(os.path.join(np_saving_path,'mp_kappa2.npy'),mp_kappa2)
print('mp_kappa3',mp_kappa3)
np.save(os.path.join(np_saving_path,'mp_kappa3.npy'),mp_kappa3)
print('mp_kappa4',mp_kappa4)
np.save(os.path.join(np_saving_path,'mp_kappa4.npy'),mp_kappa4)

print("################# MaximumDiameter 0.5-1 #########################")
print(np.mean(mp_dice0), np.std(mp_dice0), np.mean(mp_iou0), np.std(mp_iou0), np.mean(mp_kappa0), np.std(mp_kappa0))
print("################# MaximumDiameter 1-2 #########################")
print(np.mean(mp_dice1), np.std(mp_dice1), np.mean(mp_iou1), np.std(mp_iou1), np.mean(mp_kappa1), np.std(mp_kappa1))
print("################# MaximumDiameter 2-3 #########################")
print(np.mean(mp_dice2), np.std(mp_dice2), np.mean(mp_iou2), np.std(mp_iou2), np.mean(mp_kappa2), np.std(mp_kappa2))
print("################# MaximumDiameter 3-5 #########################")
print(np.mean(mp_dice3), np.std(mp_dice3), np.mean(mp_iou3), np.std(mp_iou3), np.mean(mp_kappa3), np.std(mp_kappa3))
print("################# MaximumDiameter >5 #########################")
print(np.mean(mp_dice4), np.std(mp_dice4), np.mean(mp_iou4), np.std(mp_iou4), np.mean(mp_kappa4), np.std(mp_kappa4))

print("******************* MaximumDiameter 0.5-1 *******************")
print(np.mean(mp_recall0), np.std(mp_recall0), np.mean(mp_f10), np.std(mp_f10), np.mean(mp_acc0), np.std(mp_acc0),
      np.mean(mp_auc0), np.std(mp_auc0), np.mean(mp_hd0), np.std(mp_hd0), np.mean(mp_vs0), np.std(mp_vs0))
print("******************* MaximumDiameter 1-2 *******************")
print(np.mean(mp_recall1), np.std(mp_recall1), np.mean(mp_f11), np.std(mp_f11), np.mean(mp_acc1), np.std(mp_acc1),
      np.mean(mp_auc1), np.std(mp_auc1), np.mean(mp_hd1), np.std(mp_hd1), np.mean(mp_vs1), np.std(mp_vs1))
print("******************* MaximumDiameter 2-3 *******************")
print(np.mean(mp_recall2), np.std(mp_recall2), np.mean(mp_f12), np.std(mp_f12), np.mean(mp_acc2), np.std(mp_acc2),
      np.mean(mp_auc2), np.std(mp_auc2), np.mean(mp_hd2), np.std(mp_hd2), np.mean(mp_vs2), np.std(mp_vs2))
print("******************* MaximumDiameter 3-5 *******************")
print(np.mean(mp_recall3), np.std(mp_recall3), np.mean(mp_f13), np.std(mp_f13), np.mean(mp_acc3), np.std(mp_acc3),
      np.mean(mp_auc3), np.std(mp_auc3), np.mean(mp_hd3), np.std(mp_hd3), np.mean(mp_vs3), np.std(mp_vs3))
print("******************* MaximumDiameter >5 *******************")
print(np.mean(mp_recall4), np.std(mp_recall4), np.mean(mp_f14), np.std(mp_f14), np.mean(mp_acc4), np.std(mp_acc4),
      np.mean(mp_auc4), np.std(mp_auc4), np.mean(mp_hd4), np.std(mp_hd4), np.mean(mp_vs4), np.std(mp_vs4))

print('FP_diameter_list', FP_diameter_list)
print('FP_name_list', FP_name_list)


# 结果
FP_diameter_list = [

]
