import argparse
import logging
import os
import numpy as np
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet, UNet_2, UNet_3, UNet_4
from utils.dataset import BasicDataset
import nibabel as nib
from nibabel import Nifti1Image
import cv2
import pickle
from skimage import measure, morphology
from preprocess import PreProcessing
from mitk import npy_regionprops_denoise, nii_resample

# print(torch.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')


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


class ModelPredict(PreProcessing):
    def __init__(self,
                 model_type,
                 classes,
                 model,
                 ornt,
                 spacing,
                 shape,
                 data_type,
                 threshold=0.1,
                 channels=5,
                 *args,
                 **kwargs):
        super().__init__(ornt, spacing, shape, data_type, *args, **kwargs)
        net = model_type(n_channels=channels, n_classes=classes, bilinear=False)
        net.to(device=device)
        net.load_state_dict(torch.load(model, map_location=device), strict=False)
        self.net = net
        self.threshold = threshold

    def predict_image_slice(self, full_img):
        net = self.net
        out_threshold = self.threshold
        net.eval()
        img = torch.from_numpy(BasicDataset.preprocess(full_img))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = net(img)
            if net.n_classes > 1:
                probs = F.softmax(output[-1], dim=1)
                probs = torch.argmax(probs, dim=1)
                mask = torch.squeeze(probs).cpu().numpy()
            else:
                probs = torch.sigmoid(output[-1])
                probs = probs.squeeze(0)
                tf = transforms.Compose(
                    [transforms.ToPILImage(),
                     transforms.Resize(full_img.shape[0]),
                     transforms.ToTensor()])
                probs = tf(probs.cpu())
                full_mask = probs.squeeze().cpu().numpy()
                mask = (full_mask > out_threshold)
        return mask

    def predict_image_all(self, images_nii, box_json_path, preprocessed_box_nii):
        affine = images_nii.affine
        images = images_nii.get_fdata()
        shape = images.shape
        pred_mask = np.zeros_like(images,dtype='uint16')
        preprocessed_box_nii_array = preprocessed_box_nii.get_fdata()
        [box_nii_array, num] = measure.label(preprocessed_box_nii_array, connectivity=1, return_num=True)
        print('num:', num)
        step = 2
        if num == 0:
            pass
        elif num == 1:
            arr_list = np.where(box_nii_array==1)
            x_max, x_min = np.max(arr_list[0]), np.min(arr_list[0])
            y_max, y_min = np.max(arr_list[1]), np.min(arr_list[1])
            z_max, z_min = np.max(arr_list[2]), np.min(arr_list[2])
            x_ = x_max - x_min
            y_ = y_max - y_min
            z_ = np.max(arr_list[2]) - np.min(arr_list[2])
            print('x_max, x_min, x_:', x_max, x_min, x_)
            print('y_max, y_min, y_:', y_max, y_min, y_)
            print('z_max, z_min, z_:', z_max, z_min, z_)
            if max(x_, y_) < 32:
                cut_more_x = (64 - x_) / 2
                if (64 - x_) % 2  == 0:
                    cut_more_x_left = int(cut_more_x)
                else:
                    cut_more_x_left = int(cut_more_x + 1)
                cut_more_y = (64 - y_) / 2
                if (64 - y_) % 2  == 0:
                    cut_more_y_left = cut_more_y
                else:
                    cut_more_y_left = int(cut_more_y + 1)
                left_top_point = (int(x_min-cut_more_x_left), int(y_min-cut_more_y_left))
                right_bottom_point = (int(x_min-cut_more_x_left+64), int(y_min-cut_more_y_left+64))
                for z in range(z_min, z_max+1):
                    # box_img = images[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z-step:z+step+1]
                    box_img = images[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    pred_mask_slice = self.predict_image_slice(box_img)
                    print(pred_mask_slice.shape, np.max(pred_mask_slice))
                    if np.max(pred_mask_slice) == 0:
                        pred_mask[preprocessed_box_nii_array>0] = 1
                    else:
                        pred_mask[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z] = pred_mask_slice
            else:
                pass
        else:
            # new_dat_label = np.zeros_like(box_nii_array)
            for j in range(num):
                dat_t = np.zeros_like(box_nii_array)
                dat_t[box_nii_array == j + 1] = 1
                arr_list = np.where(dat_t==1)
                x_max, x_min = np.max(arr_list[0]), np.min(arr_list[0])
                y_max, y_min = np.max(arr_list[1]), np.min(arr_list[1])
                z_max, z_min = np.max(arr_list[2]), np.min(arr_list[2])
                x_ = x_max - x_min
                y_ = y_max - y_min
                z_ = np.max(arr_list[2]) - np.min(arr_list[2])
                print('x_max, x_min, x_:', x_max, x_min, x_)
                print('y_max, y_min, y_:', y_max, y_min, y_)
                print('z_max, z_min, z_:', z_max, z_min, z_)
                if max(x_, y_) < 32:
                    cut_more_x = (64 - x_) / 2
                    if (64 - x_) % 2  == 0:
                        cut_more_x_left = int(cut_more_x)
                    else:
                        cut_more_x_left = int(cut_more_x + 1)
                    cut_more_y = (64 - y_) / 2
                    if (64 - y_) % 2  == 0:
                        cut_more_y_left = cut_more_y
                    else:
                        cut_more_y_left = int(cut_more_y + 1)
                    left_top_point = (int(x_min-cut_more_x_left), int(y_min-cut_more_y_left))
                    right_bottom_point = (int(x_min-cut_more_x_left+64), int(y_min-cut_more_y_left+64))
                    for z in range(z_min, z_max+1):
                        # box_img = images[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z-step:z+step+1]
                        box_img = images[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                        pred_mask_slice = self.predict_image_slice(box_img)
                        print(pred_mask_slice.shape, np.max(pred_mask_slice))
                        if np.max(pred_mask_slice) == 0:
                            pred_mask[box_nii_array == j + 1] = 1
                        else:
                            pred_mask[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z] = pred_mask_slice
            else:
                pass
        pred_mask[images==0] = 0 # 去掉预测在肝脏外部的部分
        mask_nii = nib.Nifti1Image(pred_mask, affine)
        return mask_nii

    def run(self, image_nii, box_json_path, box_nii):
        print('预处理原图...')
        preprocessed_nii = self.processing_image(image_nii)
        preprocessed_box_nii = self.geometric_transformation(box_nii, interpolation='nearest')
        print('开始预测...')
        mask_nii = self.predict_image_all(preprocessed_nii, box_json_path, preprocessed_box_nii)
        return mask_nii


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        dest='model_path',
        default=
        '/home3/HWGroup/wushu/liver_tumor/code/checkpoints/train_small_tumor_v140/CP_epoch383_dice0.7491900205612183.pth'
        )
    parser.add_argument('-t', '--mask_threshold', dest='threshold', type=float, default=0.5)
    parser.add_argument('-mt', '--model_type', dest='model_type', type=str, default=UNet_4)
    parser.add_argument('-ca', '--channels', dest='channels', type=int, default=1)
    parser.add_argument('-cs', '--classes', dest='classes', type=int, default=1)
    parser.add_argument('-dt', '--data_type', dest='data_type', type=str, default='dce')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ornt = ["L", "A", "S"]
    spacing=[0.9, 0.9, 3.5]
    shape=[400, 400]
    data_type = 'liver'

    # 翠萍训练的nnDetection Task55 Ts
    # img_nii_dir = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_TumorSeg/imagesTs/'
    # box_json_dir = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii/'
    # pred_dir = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess/'
    
    # img_nii_dir = '/home3/HWGroup/licp/Tumor_detection/nnDetection/datasets/Task55_LiverTumorSmall/data2/raw_splitted/imagesTs/'
    # box_json_dir = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii2/'
    # pred_dir = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess2/'
    
    img_nii_dir = '/home3/HWGroup/licp/Tumor_detection/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_TumorSeg/imagesTs/'
    box_json_dir = '/home3/HWGroup/licp/Tumor_detection/nnDetection/models/Task55_LiverTumorSmall/RetinaUNetV001_D3V001_3d/fold0/test_predictions_nii3/'
    pred_dir = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/pred/pred_boxes_55_Ts_0.5_postprocess3_problem/'

    for file in os.listdir(img_nii_dir):
        start_time = time.time()
        print(file)
        liver_nii = nib.load(os.path.join(img_nii_dir, file))
        liver_spacing = liver_nii.header.get_zooms()
        spacing[2] = liver_spacing[2]
        print(spacing)
        if '_0000' in file:
            file = file.split('_0000')[0] + '.nii.gz'
        json_name = file.split('.nii.gz')[0] + '_boxes.json'
        box_json_path = os.path.join(box_json_dir, json_name)
        box_name = file.split('.nii.gz')[0] + '_boxes.nii.gz'
        box_nii_path = os.path.join(box_json_dir, box_name)
        if os.path.exists(box_json_path) and os.path.exists(box_nii_path):
            box_nii = nib.load(box_nii_path)
            print('准备预测...')
            liver_model = ModelPredict(args.model_type, args.classes, args.model_path, ornt, spacing, shape, data_type,
                                args.threshold, args.channels)
            liver_predict = liver_model.run(liver_nii, box_json_path, box_nii)
            liver_predict = nii_resample(liver_predict, liver_nii, interpolation='nearest')
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            nib.save(liver_predict, os.path.join(pred_dir, file))
            end_time = time.time()
            print('time: ', (end_time - start_time))
