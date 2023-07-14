import os
import nibabel as nib
import numpy as np
from skimage import measure
import SimpleITK as sitk
import radiomics


def func_merge_box_25D(path_25D: str, path_box: str, path_merge_box_25D: str):
    """2.5D预测结果遗漏的用nnDetectionbox预测的box填上

    Args:
        path_25D (str): 2.5D UNet预测路径
        path_box (str): nnDetection预测路径
        path_merge_box_25D (str): 2.5D预测结果遗漏的用nnDetectionbox预测的box填上的结果路径
    """
    nii_25D = nib.load(path_25D)
    arr_25D = nii_25D.get_data()
    nii_box = nib.load(path_box)
    arr_box = nii_box.get_data()
    nii_affine = nii_box.affine
    [arr_box_data, num] = measure.label(arr_box, connectivity=1, return_num=True)
    if num > 0:
        for j in range(num):
            arr_one_box = np.zeros_like(arr_box_data)
            arr_box_data = arr_box_data
            arr_one_box[arr_box_data == j + 1] = 1
            multi = arr_25D * arr_one_box
            if np.max(multi) == 0:
                print('2.5D预测有遗漏,用box补充')
                arr_25D[arr_one_box == 1] = 1
        arr_25D = arr_25D.astype('uint8')
    nib.Nifti1Image(arr_25D, nii_affine).to_filename(path_merge_box_25D)
    print('已生成', path_merge_box_25D)


def func_merge_nnunet_box_25D(path_img: str,
                              path_pred_nnunet: str,
                              path_pred_box: str,
                              path_pred_25D: str,
                              path_merge_box_25D: str,
                              path_merge_box_25D_one: str,
                              path_merge: str,
                              do_del_merge_box_25D_file: bool = False):
    """合并nnUNet,nnDetection,2.5D UNet预测肝癌的结果

    Args:
        path_img (str): 预测原图路径
        path_pred_nnunet (str): nnUNet预测路径
        path_pred_box (str): nnDetection预测路径
        path_pred_25D (str): 2.5D UNet预测路径
        path_merge_box_25D (str): 2.5D预测结果遗漏的用nnDetectionbox预测的box填上的结果路径
        path_merge_box_25D_one (str): nnDetection合并2.5D UNet的结果再计算连通域后单个占位的路径
        path_merge (str): 总预测结果合并路径
        do_del_merge_box_25D_file (bool): False  默认不删除中间merge文件
                                          True   删除中间merge文件
    """
    label_nnunet_nii = nib.load(path_pred_nnunet)
    label_nnunet_data = label_nnunet_nii.get_fdata()
    label_nnunet_affine = label_nnunet_nii.affine
    if not os.path.exists(path_pred_box) or not os.path.exists(path_pred_25D):
        nib.save(label_nnunet_nii, path_merge)
        return
    merge_data = np.zeros_like(label_nnunet_data)
    merge_data[label_nnunet_data >= 1] = 1
    # nnDetection预测的小于3cm长径的保留
    img = sitk.ReadImage(path_img)
    # 小于3cm
    cut_diameter = 3
    # 先合并nnDetection和2.5D UNet的结果
    func_merge_box_25D(path_pred_25D, path_pred_box, path_merge_box_25D)
    label_merge_box_25D = sitk.ReadImage(path_merge_box_25D)
    label_merge_box_25D_nii = nib.load(path_merge_box_25D)
    label_merge_box_25D_data = label_merge_box_25D_nii.get_fdata()
    label_merge_box_25D_affine = label_merge_box_25D_nii.affine
    [label_merge_box_25D_data_seperate, num_merge_box_25D] = measure.label(label_merge_box_25D_data,
                                                                           connectivity=1,
                                                                           return_num=True)
    print('num_merge_box_25D:', num_merge_box_25D)
    # 将nnUNet预测的结果merge到label_merge_box_25D上
    if num_merge_box_25D == 0:
        pass
    elif num_merge_box_25D == 1:
        radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=label_merge_box_25D)
        max3Ddiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
        print('num=1', max3Ddiameter)
        if max3Ddiameter < cut_diameter:
            merge_data[label_merge_box_25D_data_seperate == 1] = 1
    else:
        for j in range(num_merge_box_25D):
            label_merge_box_25D_data_one = np.zeros_like(label_merge_box_25D_data_seperate)
            label_merge_box_25D_data_one[label_merge_box_25D_data_seperate == j + 1] = 1
            label_merge_box_25D_data_one = label_merge_box_25D_data_one.astype('uint8')
            label_merge_box_25D_one_nii = nib.Nifti1Image(label_merge_box_25D_data_one, label_merge_box_25D_affine)
            nib.save(label_merge_box_25D_one_nii, path_merge_box_25D_one)
            print('已生成', path_merge_box_25D_one)
            label_merge_box_25D_one = sitk.ReadImage(path_merge_box_25D_one)
            radiomicsshape = radiomics.shape.RadiomicsShape(inputImage=img, inputMask=label_merge_box_25D_one)
            max3Ddiameter = radiomicsshape.getMaximum3DDiameterFeatureValue() / 10
            print(j + 1, 'max3Ddiameter:', max3Ddiameter)
            # 去掉长径大于cut_diameter的占位
            if max3Ddiameter < cut_diameter:
                merge_data[label_merge_box_25D_data_one == 1] = 1
    print('合并标签中...')
    merge_data = merge_data.astype('uint8')
    nib.Nifti1Image(merge_data, label_nnunet_affine).to_filename(path_merge)
    # 删除中间文件
    if do_del_merge_box_25D_file:
        if os.path.exists(path_merge_box_25D):
            os.remove(path_merge_box_25D)
            print('已删除', path_merge_box_25D)
        if os.path.exists(path_merge_box_25D_one):
            os.remove(path_merge_box_25D_one)
            print('已删除', path_merge_box_25D_one)


#***********************************************************************************
# path_box = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/2019002800_boxes.nii.gz'
# path_25D = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/2019002800.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/2019002800_merge.nii.gz'

# 将nnunet,nnDetection和2.5D UNet标签合并
# dir_pred_nnunet = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/pred_nnunet/'
# dir_images = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/images/'
# dir_box = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/pred_box/'
# dir_25D = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/pred_25D/'
# dir_merge = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/for_test/merge/'

# 131  Tr
# dir_pred_nnunet = '/home3/HWGroup/wushu/nnUNet/DATASET/nnUNet_raw/pred/pred_tumor_gongwei_032_imagesTr/'
# dir_images = '/home3/HWGroup/wushu/LUNA16/Task131_LiverTumorSmall/raw_splitted/imagesTs__/'
# dir_box = '/home3/HWGroup/wushu/LUNA16/models/Task131_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_Tr_0.4/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.2/pred/pred_boxes_131_Tr_0.4_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/LUNA16/models/Task131_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_2.5D_results_add_nnunet_131_0.4_postprocess_less3_Tr/'

# 128  Tr
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task74_TumorSeg/predict_results_tr/'
# dir_images = '/home3/HWGroup/wushu/LUNA16/Task128_LiverTumorSmall/raw_splitted/imagesTr____/'
# dir_box = '/home3/HWGroup/wushu/LUNA16/models/Task128_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_Tr/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/pred/pred_boxes_128_0.4_postprocess_unet4_Tr/'
# dir_merge = '/home3/HWGroup/wushu/LUNA16/models/Task128_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_2.5D_results_add_nnunet_128_0.4_postprocess_less3_Tr/'

# 132  Ts
# dir_pred_nnunet = '/home3/HWGroup/wushu/nnUNet/DATASET/visualization_nnUNet_raw/pred/pred_gongweitumor_091/'
# dir_images = '/home3/HWGroup/wushu/nnUNet/DATASET/visualization_nnUNet_raw/nnUNet_raw_data/Task091_GongweiTumor/imagesTs/'
# dir_box = '/home3/HWGroup/wushu/nnDetection/models/Task132_GongweiTumor/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_0.4/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/pred/pred_boxes_132_Ts_0.4_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/nnDetection/models/Task132_GongweiTumor/RetinaUNetV001_D3V001_3d/fold0/test_2.5D_add_nnunet_132_0.4_less3_Ts/'
# 133  Ts
# dir_pred_nnunet = '/home3/HWGroup/wushu/nnUNet/DATASET/visualization_nnUNet_raw/pred/pred_gongweitumor_092/'
# dir_images = '/home3/HWGroup/wushu/nnUNet/DATASET/visualization_nnUNet_raw/nnUNet_raw_data/Task092_GongweiTumor/imagesTs/'
# dir_box = '/home3/HWGroup/wushu/nnDetection/models/Task133_GongweiTumor/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii_0.4/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/pred/pred_boxes_133_Ts_0.4_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/nnDetection/models/Task133_GongweiTumor/RetinaUNetV001_D3V001_3d/fold0/test_2.5D_add_nnunet_133_0.4_less3_Ts/'

# lcp 54  Ts
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/predicts/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_2.5D_nnunet_54_0.5_less3_Ts/'
# 54  Ts2
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/predicts2/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task54_LiverTumorSmall/data2/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess2/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_2.5D_nnunet_54_0.5_less3_Ts/'
# lcp 55  Ts
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task055_TumorSeg/predicts/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task055_TumorSeg/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_2.5D_nnunet_55_0.5_less3_Ts/'
# 55  Ts2
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_TumorSeg/test2/predicts2/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task55_LiverTumorSmall/data2/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess2/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_2.5D_nnunet_55_0.5_less3_Ts/'
# # 51 Ts1
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task051_TumorSeg/test1/predicts1/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task51_LiverTumorSmall/data1/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task51_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_51_Ts_0.5_postprocess/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_2.5D_nnunet_51_0.5_less3_Ts/'
# # 51 Ts2
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task051_TumorSeg/test2/predicts2/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task51_LiverTumorSmall/data2/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task51_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_51_Ts_0.5_postprocess2/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_2.5D_nnunet_51_0.5_less3_Ts/'
# # 53 Ts1
# # dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task053_TumorSeg/test1/predicts1/'
# # dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task53_LiverTumorSmall/datat1/raw_splitted/imagesTs/'
# # dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task53_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/'
# # dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/pred/pred_boxes_53_Ts_0.5_postprocess/'
# # dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/pred/pred_boxes_2.5D_nnunet_53_0.5_less3_Ts/'
# # 53 Ts2
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task053_TumorSeg/test2/predicts2/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task53_LiverTumorSmall/data2/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task53_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/pred/pred_boxes_53_Ts_0.5_postprocess2/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/pred/pred_boxes_2.5D_nnunet_53_0.5_less3_Ts/'
# 52 Ts1
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task052_TumorSeg/test1/predicts1/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task52_LiverTumorSmall/data1/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task51_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_51_Ts_0.5_postprocess1/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/pred/pred_boxes_2.5D_nnunet_52_0.5_less3_Ts1/'
# 52 Ts2
# dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task052_TumorSeg/test2/predicts2/'
# dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task52_LiverTumorSmall/data2/raw_splitted/imagesTs/'
# dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task51_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
# dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/pred/pred_boxes_51_Ts_0.5_postprocess2/'
# dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/pred/pred_boxes_2.5D_nnunet_52_0.5_less3_Ts2/'

# 55 problem 数据
dir_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_TumorSeg/predicts3/'
dir_images = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task55_LiverTumorSmall/data3/raw_splitted/imagesTs/'
dir_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii3/'
dir_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess3_problem/'
dir_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_2.5D_nnunet_55_0.5_less3_Ts3_problem/'

os.makedirs(dir_merge, exist_ok=True)
files = os.listdir(dir_pred_nnunet)
for file_name in files:
    if file_name.endswith('.nii.gz'):
        print(file_name)
        # path_img
        img_file_name = file_name.split('.nii.gz')[0] + '_0000.nii.gz'
        path_img = os.path.join(dir_images, img_file_name)
        # path_pred_nnunet
        path_pred_nnunet = os.path.join(dir_pred_nnunet, file_name)
        # path_pred_box
        box_file_name = file_name.split('.nii.gz')[0] + '_boxes.nii.gz'
        path_pred_box = os.path.join(dir_box, box_file_name)
        # path_pred_25D
        path_pred_25D = os.path.join(dir_25D, file_name)
        # path_merge_box_25D
        merge_box_25D_file_name = file_name.split('.nii.gz')[0] + '_boxes_25D.nii.gz'
        path_merge_box_25D = os.path.join(dir_merge, merge_box_25D_file_name)
        # path_merge_box_25D_one
        merge_box_25D_one_file_name = file_name.split('.nii.gz')[0] + '_boxes_25D_one.nii.gz'
        path_merge_box_25D_one = os.path.join(dir_merge, merge_box_25D_one_file_name)
        # path_merge
        path_merge = os.path.join(dir_merge, file_name)
        func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
                                  path_merge_box_25D_one, path_merge, True)


# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/imagesTs/2020037971_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/predicts2/2020037971.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/2020037971_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess2/2020037971.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971_2/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971_2/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971_2/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)

# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/imagesTs/2020037971_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/predicts1/2020037971.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/2020037971_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess1/2020037971.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020037971/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)

# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/imagesTs/2021115777_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/predicts2/2021115777.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/2021115777_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess2/2021115777.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777_2/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777_2/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777_2/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)

# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/imagesTs/2021115777_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/predicts1/2021115777.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/2021115777_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess1/2021115777.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2021115777/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)

# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/imagesTs/2020047498_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test2/predicts2/2020047498.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/2020047498_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess2/2020047498.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498_2/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498_2/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498_2/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)

# path_img = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/imagesTs/2020047498_0000.nii.gz'
# path_pred_nnunet = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task054_TumorSeg/test1/predicts1/2020047498.nii.gz'
# path_pred_box = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task54_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii1/2020047498_boxes.nii.gz'
# path_pred_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/pred/pred_boxes_54_Ts_0.5_postprocess1/2020047498.nii.gz'
# path_merge_box_25D = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498/boxes_25D.nii.gz'
# path_merge_box_25D_one = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498/boxes_25D_one.nii.gz'
# path_merge = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/for_pred/resize_testing/2020047498/predict.nii.gz'
# func_merge_nnunet_box_25D(path_img, path_pred_nnunet, path_pred_box, path_pred_25D, path_merge_box_25D,
#                           path_merge_box_25D_one, path_merge, True)