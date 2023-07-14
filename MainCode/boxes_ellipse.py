import os
import time
import pickle
from pathlib import Path
import nibabel as nib
import numpy as np
import cv2
from PIL import Image
import shutil
import mitk
from mitk.image_conversion.dcm2nii import dcm2nii
from mitk.image_conversion.dcmseg2nii import dcmseg2nii1
from mitk import DcmDataSet, nii2dcmseg, nii_resample, nii_resize
import skimage
from skimage import measure, morphology
import SimpleITK as sitk

### 将nnDetection预测的box变成椭圆形状

def generate_ball_mask(img_height, img_width, img_depth, radius, center_x, center_y, center_z):
    x = np.array(list(range(img_height))).reshape([img_height, 1, 1])
    y = np.array(list(range(img_width))).reshape([1, img_width, 1])
    z = np.array(list(range(img_depth))).reshape([1, 1, img_depth])
    # circle mask
    mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
    mask = mask.astype('uint16')
    print(mask.shape)
    return mask


def generate_circle_mask(img_height, img_width, radius, center_x, center_y):
    y, x = np.ogrid[0:img_height, 0:img_width]
    # circle mask
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    return mask


def load_pickle(path: Path, **kwargs):
    """
    Load pickle file

    Args:
        path: path to pickle file
        **kwargs: keyword arguments passed to :func:`pickle.load`

    Returns:
        Any: json data
    """
    if isinstance(path, str):
        path = Path(path)
    if not any([fix == path.suffix for fix in [".pickle", ".pkl"]]):
        path = Path(str(path) + ".pkl")

    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def box2nii_postprep(training_dir):
    """
    将nndeection预测出的boxes调整成椭圆存成nii标签
    """
    threshold = 0.5

    prediction_dir = os.path.join(training_dir, "test_predictions")  #\
    # if test else training_dir / "val_predictions"
    # save_dir = training_dir / "test_predictions_nii" \
    save_dir = os.path.join(training_dir, "test_predictions_nii_postprepcess1")  #\
    # if test else training_dir / "val_predictions_nii"
    # save_dir.mkdir(exist_ok=True)
    # if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 测试集
    label_dir = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task73_TumorSeg/imagesTs/'
    # 总 训练加测试集
    # label_dir = '/home3/HWGroup/wushu/LUNA16/Task117_LiverTumorSmall/raw_splitted/labelsTr_Ts_total/'

    # case_ids = [p.split("_boxes.pkl")[0] for p in os.listdir(prediction_dir)]
    case_ids = [p.split(".nii.gz")[0] for p in os.listdir(label_dir)]
    for cid in case_ids:
        # if not os.path.exists(os.path.join(save_dir, f"{cid}_postprep.nii.gz")):
        # if cid == "2020063385":
        print(cid)
        # 获取头信息
        res = load_pickle(os.path.join(prediction_dir, f"{cid}_boxes.pkl"))
        lab = sitk.ReadImage(os.path.join(label_dir, f"{cid}.nii.gz"))
        nii_lab = nib.load(os.path.join(label_dir, f"{cid}.nii.gz"))
        nii_affine = nii_lab.affine

        print(nii_affine)
        shape = lab.GetSize()
        # shape = res["original_size_of_raw_data"]
        spacing = res["itk_spacing"]
        nii_shape = [shape[0], shape[1], shape[2]]
        nii_spacing = spacing

        boxes = res["pred_boxes"]
        scores = res["pred_scores"]
        labels = res["pred_labels"]

        _mask = scores >= threshold
        boxes = boxes[_mask]
        labels = labels[_mask]
        scores = scores[_mask]

        idx = np.argsort(scores)
        scores = scores[idx]
        boxes = boxes[idx]
        labels = labels[idx]

        # 将z方向等体素到x y上去，相应的z spacing和shape要修改
        z_divide_xy = abs(int(spacing[2] / spacing[1]))
        new_spacing = list(spacing)
        new_spacing[2] = new_spacing[2] / z_divide_xy
        new_spacing = tuple(new_spacing)

        new_nii_affine = nii_affine
        new_nii_affine[2][2] = new_nii_affine[2][2] / z_divide_xy
        print(new_nii_affine)

        mask_slicing = np.zeros((shape[0], shape[1], shape[2] * z_divide_xy), dtype=np.uint8)

        # prediction_meta = {}
        for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
            # 按照nnDetection预测的box格式分别对应xyz
            center_z = int((pbox[0] + pbox[2]) / 2)
            center_y = int((pbox[1] + pbox[3]) / 2)
            center_x = int((pbox[4] + pbox[5]) / 2)
            radius = int((pbox[3] - pbox[1]) / 2)
            # 在等体素的矩阵画球
            mask = generate_ball_mask(shape[0], shape[1], shape[2] * z_divide_xy, radius, center_x, center_y,
                                      center_z * z_divide_xy)
            mask_slicing[mask == 1] = instance_id
            # 预测信息
            # prediction_meta[int(instance_id)] = {
            #     "score": float(pscore),
            #     "label": int(plabel),
            #     "box": list(map(int, pbox))
            # }

        print(f"Created instance mask with {mask_slicing.max()} instances.")
        # # 保存等体素的nii
        # mask_slicing_itk = sitk.GetImageFromArray(mask_slicing)
        # mask_slicing_itk.SetOrigin(res["itk_origin"])
        # mask_slicing_itk.SetDirection(res["itk_direction"])
        # mask_slicing_itk.SetSpacing(new_spacing)
        # mask_slicing_itk_path = os.path.join(save_dir, f"{cid}_boxes.nii.gz")
        # sitk.WriteImage(mask_slicing_itk, mask_slicing_itk_path)

        mask_slicing_nii = nib.Nifti1Image(mask_slicing, new_nii_affine)
        mask_slicing_nii_path = os.path.join(save_dir, f"{cid}_spac.nii.gz")
        nib.save(mask_slicing_nii, mask_slicing_nii_path)

        # 将修改好的球的label进行resize到原图
        instance_mask_path = os.path.join(save_dir, f"{cid}_postprep.nii.gz")
        mask_slicing_itk_nii = nib.load(mask_slicing_nii_path)
        new_instance_mask_nii = nii_resize(mask_slicing_itk_nii,
                                           target_shape=nii_shape,
                                           target_spacing=nii_spacing,
                                           interpolation='nearest')
        nib.save(new_instance_mask_nii, instance_mask_path)
        if os.path.exists(mask_slicing_nii_path):
            os.remove(mask_slicing_nii_path)


def box2nii_postprep1(training_dir):
    """
    将nndeection预测出的boxes调整成椭圆存成nii标签
    """
    threshold = 0.5

    prediction_dir = os.path.join(training_dir, "test_predictions")  #\
    # if test else training_dir / "val_predictions"
    # save_dir = training_dir / "test_predictions_nii" \
    save_dir = os.path.join(training_dir, "test_predictions_nii_postprepcess2")  #\
    # if test else training_dir / "val_predictions_nii"
    # save_dir.mkdir(exist_ok=True)
    # if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 测试集
    label_dir = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task73_TumorSeg/imagesTs/'
    # 总 训练加测试集
    # label_dir = '/home3/HWGroup/wushu/LUNA16/Task117_LiverTumorSmall/raw_splitted/labelsTr_Ts_total/'

    # case_ids = [p.split("_boxes.pkl")[0] for p in os.listdir(prediction_dir)]
    case_ids = [p.split(".nii.gz")[0] for p in os.listdir(label_dir)]
    for cid in case_ids:
        # if not os.path.exists(os.path.join(save_dir, f"{cid}_postprep.nii.gz")):
        # if cid == "2020062827_ellipse":
            print(cid)
            # 获取头信息
            res = load_pickle(os.path.join(prediction_dir, f"{cid}_boxes.pkl"))
            # lab = sitk.ReadImage(os.path.join(label_dir, f"{cid}.nii.gz"))
            nii_lab = nib.load(os.path.join(label_dir, f"{cid}.nii.gz"))
            nii_affine = nii_lab.affine

            # print(nii_affine)
            # shape = lab.GetSize()
            shape = nii_lab.get_fdata().shape

            boxes = res["pred_boxes"]
            scores = res["pred_scores"]
            labels = res["pred_labels"]

            _mask = scores >= threshold
            boxes = boxes[_mask]
            labels = labels[_mask]
            scores = scores[_mask]

            idx = np.argsort(scores)
            scores = scores[idx]
            boxes = boxes[idx]
            labels = labels[idx]

            mask = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)

            # prediction_meta = {}
            for instance_id, (pbox, pscore, plabel) in enumerate(zip(boxes, scores, labels), start=1):
                mask_instance = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)
                center_x = int((pbox[1] + pbox[3]) / 2)
                axes_x = int((pbox[3] - pbox[1]) / 2)
                center_y = int((pbox[4] + pbox[5]) / 2)
                axes_y = int((pbox[5] - pbox[4]) / 2)
                for i in range(shape[2]):
                    if (i < pbox[2] - 1) and i > pbox[0]:
                        print(i)
                        circle = np.zeros(shape=(shape[0], shape[1]), dtype=np.uint8)
                        cv2.ellipse(img=circle,
                                    center=(
                                        center_x,
                                        center_y,
                                    ),
                                    axes=(axes_x, axes_y),
                                    angle=0,
                                    startAngle=0,
                                    endAngle=360,
                                    color=1,
                                    thickness=-1)
                        # cv2.fillPoly()
                        mask_instance[:, :, i] = circle
                mask[mask_instance == 1] = instance_id
                # 预测信息
                # prediction_meta[int(instance_id)] = {
                #     "score": float(pscore),
                #     "label": int(plabel),
                #     "box": list(map(int, pbox))
                # }

            print(f"Created instance mask with {mask.max()} instances.")

            mask_nii = nib.Nifti1Image(mask, nii_affine)
            mask_path = os.path.join(save_dir, f"{cid}_ellipse.nii.gz")
            nib.save(mask_nii, mask_path)

            # # 将修改好的球的label进行resize到原图
            # instance_mask_path = os.path.join(save_dir, f"{cid}_postprep.nii.gz")
            # mask_slicing_itk_nii = nib.load(mask_slicing_nii_path)
            # new_instance_mask_nii = nii_resize(mask_slicing_itk_nii,
            #                                 target_shape=nii_shape,
            #                                 target_spacing=nii_spacing,
            #                                 interpolation='nearest')
            # nib.save(new_instance_mask_nii, instance_mask_path)
            # if os.path.exists(mask_slicing_nii_path):
            #     os.remove(mask_slicing_nii_path)

# 椭圆
# box2nii_postprep('/home3/HWGroup/wushu/LUNA16/models/Task117_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
# box2nii_postprep('/home3/HWGroup/wushu/LUNA16/models/Task116_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
# box2nii_postprep('/home3/HWGroup/wushu/LUNA16/models/Task118_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
# box2nii_postprep('/home3/HWGroup/wushu/LUNA16/models/Task120_LiverTumorSmallTest/RetinaUNetV001_D3V001_3d/fold0/')
# box2nii_postprep('/home3/HWGroup/wushu/LUNA16/models/Task121_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')

# 改进方法 椭圆
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task117_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task116_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task118_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task120_LiverTumorSmallTest/RetinaUNetV001_D3V001_3d/fold0/')
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task121_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')
box2nii_postprep1('/home3/HWGroup/wushu/LUNA16/models/Task122_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/')