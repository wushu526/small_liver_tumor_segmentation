import os
import time
import shutil
import json
import nibabel as nib
import numpy as np
# import cv2
# from PIL import Image
# import mitk
# from mitk.image_conversion.dcm2nii import dcm2nii
# from mitk.image_conversion.dcmseg2nii import dcmseg2nii1
# from mitk import DcmDataSet, nii2dcmseg, nii_resample, nii_resize
# import pydicom
# import skimage
from skimage import measure, morphology
import SimpleITK as sitk
# from functools import wraps
import radiomics
from radiomics import featureextractor

#### radiomics ####
# label_num = 1
# extractor = featureextractor.RadiomicsFeatureExtractor()
# settings = {}
# settings['binWidth'] = 25
# settings['geometryTolerance'] = 1000
# settings['resampledPixelSpacing'] = None
# settings['interpolator'] = 'sitkBSpline'
# settings['verbose'] = True
# extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
# featureVector = extractor.execute('/home3/HWGroup/licp/Tumor_detection/tohospital_registration/2021077498/delay.nii.gz',
#                                   '/home3/HWGroup/licp/Tumor_detection/tohospital_registration/2021077498/lesion_correct2.nii.gz',
#                                   label_num)
# MaximumDiameter = float(featureVector['original_shape_Maximum3DDiameter'])/10  #cm
# print(MaximumDiameter)

# img = sitk.ReadImage('/home3/HWGroup/licp/Tumor_detection/tohospital_registration/2021077498/delay.nii.gz')
# label = sitk.ReadImage('/home3/HWGroup/licp/Tumor_detection/tohospital_registration/2021077498/lesion_correct.nii.gz')
# radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=label)
# print(radiomicsshape.getMaximum3DDiameterFeatureValue())
'''
raw文件夹下先放入对应的未处理过的imagesTr,imagesTs,labelsTr,labelsTs
首先先把raw_splitted里面的imagesTr,imagesTs,labelsTr,labelsTs处理成需要的格式:
imagesTr,imagesTs的单期nii文件名要加上'_0000',多期对应好期相
labelsTr,labelsTs每个标签要有对应的json
本代码的目的是将原肿瘤或占位标签处理成多个单独的小肿瘤或占位，同时删除大于某长径的或小于某连通域的小肿瘤或占位
'''


# 处理肿瘤占位标签，计算大于某个长径的占位
def func_MaximumDiameter_(path_img_file, path_lab_file):
    '''计算长径
    '''
    img = sitk.ReadImage(path_img_file)
    label = sitk.ReadImage(path_lab_file)
    mask = nib.load(path_lab_file).get_fdata()
    regionprop = measure.regionprops(mask)
    print('regionprop', regionprop[0].area, regionprop[0].bbox, regionprop[0].bbox_area)
    [data_tumor_label, num] = measure.label(mask, connectivity=1, return_num=True)
    if num == 0:
        raise Exception('no tumors')
    elif num == 1:
        pass
    else:
        vols = {}
        affine = nib.load(path_lab_file).affine
        for i in range(num):
            volume = np.sum(data_tumor_label == i + 1)
            if volume > 3:
                vols[i] = volume
        print('vols', vols, ', num', num)
        new_dat_label = np.zeros_like(data_tumor_label)
        x2_x1 = 0
        y2_y1 = 0
        z2_z1 = 0
        for j in range(num):
            # print(j+1)
            dat_t = np.zeros_like(data_tumor_label)
            dat_t[data_tumor_label == j + 1] = 1
            dat_t = dat_t.astype('uint16')
            new_nii = nib.Nifti1Image(dat_t, affine)
            nib.save(new_nii, 'tumor6.nii.gz')
            lab = sitk.ReadImage('tumor6.nii.gz')
            radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=lab)
            max3Ddiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
            print('max3Ddiameter', max3Ddiameter)
            # 长径小于2cm
            if max3Ddiameter < 2:
                # new_dat_label[dat_t==1] = 1
                (x_arr, y_arr, z_arr) = np.where(dat_t == 1)
                x2_x1_ = np.max(x_arr) - np.min(x_arr)
                y2_y1_ = np.max(y_arr) - np.min(y_arr)
                z2_z1_ = np.max(z_arr) - np.min(z_arr)
                # print('x', np.min(x_arr), np.max(x_arr), x2_x1_)
                # print('y', np.min(y_arr), np.max(y_arr), y2_y1_)
                # print('z', np.min(z_arr), np.max(z_arr), z2_z1_)
                if x2_x1_ > x2_x1:
                    x2_x1 = x2_x1_
                if y2_y1_ > y2_y1:
                    y2_y1 = y2_y1_
                if z2_z1_ > z2_z1:
                    z2_z1 = z2_z1_
                # 计算x,y,z的位置和长度
        print(x2_x1, y2_y1, z2_z1)


# 处理肿瘤占位标签，删除大于某个长径的占位
def cut_tumor_long(path_img_file, path_lab_file, path_label_less_num, file, cut_diameter):
    '''切分长径
    '''
    # if os.path.exists(os.path.join(path_label_less_num, file)):
    #     print(os.path.join(path_label_less_num, file), '已处理!!!')
    # else:
    print(file)
    img = sitk.ReadImage(path_img_file)
    label = sitk.ReadImage(path_lab_file)
    mask_nii = nib.load(path_lab_file)
    mask = mask_nii.get_fdata().astype('uint16')
    affine = mask_nii.affine
    [data_tumor_label, num] = measure.label(mask, connectivity=1, return_num=True)
    if num == 0:
        raise Exception('no tumors')
    elif num == 1:
        radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=label)
        max3Ddiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
        print('num=1', max3Ddiameter)
        if max3Ddiameter < cut_diameter:
            new_label = nib.Nifti1Image(mask, affine)
            nib.save(new_label, os.path.join(path_label_less_num, file))
            print('已保存nii', os.path.join(path_label_less_num, file))
    else:
        vols = {}
        for i in range(num):
            volume = np.sum(data_tumor_label == i + 1)
            # if volume < 3:
            #     print('too small!',volume)
            # else:
            vols[i] = volume
            # volume太小的抛弃 !这里会影响标签的数值的连续性，影响预处理，不能加
            # if volume > 3:
            #     vols[i] = volume
        print('vols', vols)
        new_dat_label = np.zeros_like(data_tumor_label,dtype='uint16')
        for j in range(num):
            dat_t = np.zeros_like(data_tumor_label)
            dat_t[data_tumor_label == j + 1] = 1
            dat_t = dat_t.astype('uint16')
            new_nii = nib.Nifti1Image(dat_t, affine)
            nib.save(new_nii, 'tumor6.nii.gz')
            lab = sitk.ReadImage('tumor6.nii.gz')
            radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=lab)
            max3Ddiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
            print(j + 1, 'max3Ddiameter:', max3Ddiameter)
            if max3Ddiameter < cut_diameter:
                new_dat_label[dat_t == 1] = 1
        if np.max(new_dat_label) != 0:
            new_dat_label = new_dat_label.astype('uint16')
            new_label = nib.Nifti1Image(new_dat_label, affine)
            nib.save(new_label, os.path.join(path_label_less_num, file))
            print('已保存nii', os.path.join(path_label_less_num, file))


################## make multi labels 剔除所有小的连通域占位，太小的nnDetection预处理会报错 ################
def cut_tumor_small(dir_less_num, dir_label, file):
    if os.path.exists(os.path.join(dir_less_num, file)):
        lab = nib.load(os.path.join(dir_less_num, file))
        seg = lab.get_fdata().astype('uint16')
        affine = lab.affine
        [seg, num] = measure.label(seg, connectivity=1, return_num=True)
        # 去除特别小的占位 (这里需要在实际情况中针对性调整一下)
        # seg = morphology.remove_small_objects(seg, min_size=5, connectivity=1) # 123
        # seg = morphology.remove_small_objects(seg, min_size=4, connectivity=1) # 124
        # seg = morphology.remove_small_objects(seg, min_size=2, connectivity=1) # 125
        # seg = morphology.remove_small_objects(seg, min_size=4, connectivity=1)
        # seg = morphology.remove_small_objects(seg, min_size=3, connectivity=1)
        # seg = morphology.remove_small_objects(seg, min_size=4, connectivity=1)
        seg = morphology.remove_small_objects(seg, min_size=10, connectivity=1)
        # seg = morphology.remove_small_objects(seg, min_size=30, connectivity=1)
        # seg = morphology.remove_small_objects(seg, min_size=20, connectivity=1)
        # seg = morphology.remove_small_objects(seg, min_size=100, connectivity=1)
        # 分开每个小的占位
        [seg, num1] = measure.label(seg, connectivity=1, return_num=True)
        new_num = np.unique(seg)
        # 顺序连续排列不同连通域
        if len(new_num) > 1:
            for i in range(1, len(new_num)):
                if i != 0:
                    seg[seg == new_num[i]] = i
        seg = seg.astype('uint16')
        newnew_num = np.unique(seg)
        print('连通域初始数量    :', num)
        print('去除小连通域后数量:', len(new_num) - 1, new_num)
        print('再排列后数量      :', len(newnew_num) - 1, newnew_num)
        # 有可能为空
        if len(newnew_num) - 1 != 0:
            nib.Nifti1Image(seg, affine).to_filename(os.path.join(dir_label, file))
            file_json = file.split('.nii.gz')[0] + '.json'
            dict_json = {}
            dict_json["instances"] = {"1": 0}
            if len(newnew_num) > 1:
                for i in range(1, len(newnew_num)):
                    if i != 0:
                        dict_json['instances'][str(i)] = 0
            print(dict_json)
            json_new = json.dumps(dict_json)
            with open(os.path.join(dir_label, file_json), 'w', encoding='utf-8') as js_file:
                js_file.write(json_new)


def cut_tumor_long_small(dir_raw_images, dir_raw_labels, dir_label_less_num_, dir_raw_splitted_labels,
                         dir_raw_splitted_images):
    # raw_images
    for file in os.listdir(dir_raw_images):
        # if file in['2020036993.nii.gz','2021120890.nii.gz']:
            path_img_file = os.path.join(dir_raw_images, file)
            path_label_file = os.path.join(dir_raw_labels, file)
            label_nii = nib.load(path_label_file)
            dat = label_nii.get_fdata()
            # affine_Ts = labelTs_nii.affine
            lists = np.unique(dat)
            for ii in lists:
                if ii == 0:
                    pass
                else:
                    # 切分不同长径的占位
                    cut_tumor_long(path_img_file, path_label_file, dir_label_less_num_, file, cut_diameter)
                    # 分开每个连通域占位，去除小连通域，并生成对应的json
                    cut_tumor_small(dir_label_less_num_, dir_raw_splitted_labels, file)
    # 复制images，并修改文件名
    for file in os.listdir(dir_raw_splitted_labels):
        # if file in['2020036993.nii.gz','2021120890.nii.gz']:
            if file.endswith('.nii.gz'):
                file_ = file.split('.nii.gz')[0] + '_0000.nii.gz'
                shutil.copyfile(os.path.join(dir_raw_images, file), os.path.join(dir_raw_splitted_images, file_))


def cutting_tumor(dir_total, cut_diameter):
    '''
    nnDetection预处理切分长径总入口
    dir_total: 数据主路径
    cut_diameter: 切分长径大小,单位cm
    '''
    dir_raw = os.path.join(dir_total, 'raw')
    dir_raw_imagesTr = os.path.join(dir_raw, 'imagesTr')
    dir_raw_imagesTs = os.path.join(dir_raw, 'imagesTs')
    dir_raw_labelsTr = os.path.join(dir_raw, 'labelsTr')
    dir_raw_labelsTs = os.path.join(dir_raw, 'labelsTs')
    dir_label_less_num_tr = os.path.join(dir_raw, f'labels_less{cut_diameter}')
    dir_label_less_num_ts = os.path.join(dir_raw, f'labels_less{cut_diameter}')
    dir_raw_splitted = os.path.join(dir_total, 'raw_splitted')
    dir_raw_splitted_imagesTr = os.path.join(dir_raw_splitted, 'imagesTr')
    dir_raw_splitted_imagesTs = os.path.join(dir_raw_splitted, 'imagesTs')
    dir_raw_splitted_labelsTr = os.path.join(dir_raw_splitted, 'labelsTr')
    dir_raw_splitted_labelsTs = os.path.join(dir_raw_splitted, 'labelsTs')
    os.makedirs(dir_label_less_num_tr,exist_ok=True)
    os.makedirs(dir_label_less_num_ts,exist_ok=True)
    os.makedirs(dir_raw_splitted_imagesTr,exist_ok=True)
    os.makedirs(dir_raw_splitted_imagesTs,exist_ok=True)
    os.makedirs(dir_raw_splitted_labelsTr,exist_ok=True)
    os.makedirs(dir_raw_splitted_labelsTs,exist_ok=True)

    # Training
    cut_tumor_long_small(dir_raw_imagesTr, dir_raw_labelsTr, dir_label_less_num_tr, dir_raw_splitted_labelsTr,
                         dir_raw_splitted_imagesTr)
    # Testing
    cut_tumor_long_small(dir_raw_imagesTs, dir_raw_labelsTs, dir_label_less_num_ts, dir_raw_splitted_labelsTs,
                         dir_raw_splitted_imagesTs)

# dir_total = '/home3/HWGroup/wushu/LUNA16/Task120_LiverTumorSmallTest/'
# cut_diameter = 2
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task121_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task122_LiverTumorSmall/'
# cut_diameter = 2
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task123_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task124_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task125_LiverTumorSmall/'
# cut_diameter = 3

# dir_total = '/home3/HWGroup/wushu/LUNA16/Task127_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/'
# cut_diameter = 3

# 第二批数据
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task129_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task130_LiverTumorSmall/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/LUNA16/Task131_LiverTumorSmall/'
# cut_diameter = 3

# 第三批数据
# dir_total = '/home3/HWGroup/wushu/nnDetection/DATASET/Task132_GongweiTumor/'
# cut_diameter = 3
# dir_total = '/home3/HWGroup/wushu/nnDetection/DATASET/Task133_GongweiTumor/'
# cut_diameter = 3
dir_total = '/home3/HWGroup/wushu/nnDetection/DATASET/Task134_GongweiTumor/'
cut_diameter = 3

cutting_tumor(dir_total, cut_diameter)



