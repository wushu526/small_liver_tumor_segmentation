# import itk
# import vtk
import os
from re import L
import nibabel as nib
import numpy as np
from skimage import measure, morphology
import skimage.measure
from radiomics import featureextractor

settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None
settings['interpolator'] = 'sitkBSpline'
settings['correctMask'] = True
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
################## Computing metrics###################################


def func_maximumdiameter(MaxDiameter):
    if MaxDiameter > 0.5 and MaxDiameter <= 1:
        return 1
    elif MaxDiameter > 1 and MaxDiameter <= 2:
        return 2
    elif MaxDiameter > 2 and MaxDiameter <= 3:
        return 3
    elif MaxDiameter > 3 and MaxDiameter <= 5:
        return 4
    else:
        return 5


def func_diameters(path_img, data_label, num, affine):
    lab_num = 0
    lab_1, lab_2, lab_3, lab_4, lab_5 = 0, 0, 0, 0, 0
    for i in range(num):
        volume = np.sum(data_label == i + 1)
        if volume > 5:
            dat_t = np.zeros_like(data_label)
            dat_t[data_label == i + 1] = 1
            new_nii = nib.Nifti1Image(dat_t, affine)
            qform = new_nii.get_qform()
            new_nii.set_qform(qform)
            sfrom = new_nii.get_sform()
            new_nii.set_sform(sfrom)
            nib.save(new_nii, 'tumor32.nii.gz')
            nii = nib.load(path_img)
            qform = nii.get_qform()
            nii.set_qform(qform)
            sfrom = nii.get_sform()
            nii.set_sform(sfrom)
            nib.save(nii, 'tumor33.nii.gz')
            featureVector = extractor.execute('tumor33.nii.gz', 'tumor32.nii.gz')
            MaximumDiameter = float(featureVector['original_shape_Maximum3DDiameter']) / 10  #cm
            maxx = func_maximumdiameter(MaximumDiameter)
        else:
            maxx = 1
        lab_num += 1
        if maxx == 5:  # >5cm
            lab_5 += 1
        elif maxx == 4:  # 3-5cm
            lab_4 += 1
        elif maxx == 3:  # 2-3cm
            lab_3 += 1
        elif maxx == 2:  # 1-2cm
            lab_2 += 1
        elif maxx == 1:  # 0.5-1cm
            lab_1 += 1
    return [lab_num, lab_1, lab_2, lab_3, lab_4, lab_5]



path_ = '/home3/HWGroup/Data/Share/Tumor_detection/tohospital_duijie3/tohospital_registration3/'
files = os.listdir(path_)

# 统计测试集的长径和类别
# path_ts = '/home3/HWGroup/wushu/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data/Task33_gongwei_tumor/imagesTs/'  # 前两批数据建模的测试集
path_ts = '/home3/HWGroup/wushu/nnUNet/DATASET/visualization_nnUNet_raw/nnUNet_raw_data/Task92_GongweiTumor/imagesTs/'  # 前三批数据建模的测试集
new_files = []
for fi in os.listdir(path_ts):
    file_name = fi.split('.')[0]
    if file_name in files:
        new_files.append(file_name)
files = new_files
print(files)
####################


label_1, label_2, label_3, label_4, label_5 = 0, 0, 0, 0, 0
diameter1_1, diameter1_1_2, diameter1_2_3, diameter1_3_5, diameter1_5 = 0, 0, 0, 0, 0
diameter2_1, diameter2_1_2, diameter2_2_3, diameter2_3_5, diameter2_5 = 0, 0, 0, 0, 0
diameter3_1, diameter3_1_2, diameter3_2_3, diameter3_3_5, diameter3_5 = 0, 0, 0, 0, 0
diameter4_1, diameter4_1_2, diameter4_2_3, diameter4_3_5, diameter4_5 = 0, 0, 0, 0, 0
diameter5_1, diameter5_1_2, diameter5_2_3, diameter5_3_5, diameter5_5 = 0, 0, 0, 0, 0

for file in files:
    print(file)
    path_file = os.path.join(path_, file)
    path_img_file = os.path.join(path_file, 'delay.nii.gz')
    path_lesion_corrected = os.path.join(path_file, 'lesion_corrected.nii.gz')
    if not os.path.exists(path_lesion_corrected):
        continue
    nii_lesion_corrected = nib.load(path_lesion_corrected)
    dat = nii_lesion_corrected.get_fdata()
    affine = nii_lesion_corrected.affine
    lists = np.unique(dat)
    print(lists)
    for ii in lists:
        if ii == 0:
            pass
        else:
            mask = np.zeros_like(dat)
            mask[dat == ii] = 1
            data_tumor_label, nums = measure.label(mask, return_num=True, connectivity=1)
            if ii == 1:
                lists = func_diameters(path_img_file, data_tumor_label, nums, affine)
                label_1 += lists[0]
                diameter1_1 += lists[1]
                diameter1_1_2 += lists[2]
                diameter1_2_3 += lists[3]
                diameter1_3_5 += lists[4]
                diameter1_5 += lists[5]
            elif ii == 2:
                lists = func_diameters(path_img_file, data_tumor_label, nums, affine)
                label_2 += lists[0]
                diameter2_1 += lists[1]
                diameter2_1_2 += lists[2]
                diameter2_2_3 += lists[3]
                diameter2_3_5 += lists[4]
                diameter2_5 += lists[5]
            elif ii == 3:
                lists = func_diameters(path_img_file, data_tumor_label, nums, affine)
                label_3 += lists[0]
                diameter3_1 += lists[1]
                diameter3_1_2 += lists[2]
                diameter3_2_3 += lists[3]
                diameter3_3_5 += lists[4]
                diameter3_5 += lists[5]
            elif ii == 4:
                lists = func_diameters(path_img_file, data_tumor_label, nums, affine)
                label_4 += lists[0]
                diameter4_1 += lists[1]
                diameter4_1_2 += lists[2]
                diameter4_2_3 += lists[3]
                diameter4_3_5 += lists[4]
                diameter4_5 += lists[5]
            elif ii == 5:
                lists = func_diameters(path_img_file, data_tumor_label, nums, affine)
                label_5 += lists[0]
                diameter5_1 += lists[1]
                diameter5_1_2 += lists[2]
                diameter5_2_3 += lists[3]
                diameter5_3_5 += lists[4]
                diameter5_5 += lists[5]
            else:
                raise Exception



print(label_1, diameter1_1, diameter1_1_2, diameter1_2_3, diameter1_3_5, diameter1_5)
print(label_2, diameter2_1, diameter2_1_2, diameter2_2_3, diameter2_3_5, diameter2_5)
print(label_3, diameter3_1, diameter3_1_2, diameter3_2_3, diameter3_3_5, diameter3_5)
print(label_4, diameter4_1, diameter4_1_2, diameter4_2_3, diameter4_3_5, diameter4_5)
print(label_5, diameter5_1, diameter5_1_2, diameter5_2_3, diameter5_3_5, diameter5_5)
