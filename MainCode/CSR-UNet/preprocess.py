import os
from turtle import st
import mitk
import nibabel as nib
import numpy as np
from skimage import measure, morphology
import cv2
# from scipy.ndimage import interpolation


class PreProcessing:

    def __init__(self,
                 ornt,
                 spacing,
                 shape,
                 data_type,
                 apply_histeq=False,
                 apply_N4=False,
                 apply_standardization=False,
                 apply_normalization=True):
        self.ornt = ornt
        self.spacing = spacing
        self.shape = shape
        self.apply_histeq = apply_histeq
        self.apply_N4 = apply_N4
        self.apply_standardization = apply_standardization
        self.apply_normalization = apply_normalization
        self.data_type = data_type

    def geometric_transformation(self, image_nii: nib.Nifti1Image, *args, **kwargs):
        # 方向校准
        reoriented_nii = mitk.nii_reorientation(image_nii, end_ornt=self.ornt)
        # 大小校准
        resized_nii = mitk.nii_resize(reoriented_nii,
                                      target_shape=self.shape,
                                      target_spacing=self.spacing,
                                      *args,
                                      **kwargs)
        return resized_nii

    @staticmethod
    def N4(image_nii: nib.Nifti1Image):
        return mitk.npy_correct_bias_N4(image_nii)

    @staticmethod
    def histeq(array: np.ndarray, *args, **kwargs):
        array = mitk.npy_histogram_equalization(array, *args, **kwargs)
        return array

    @staticmethod
    def standardization(array: np.ndarray):
        mean = np.mean(array)
        std = np.std(array)
        array = (array - mean) / (std + 0.00001)
        return array

    @staticmethod
    def normalization(array: np.ndarray):
        maxv = np.max(array)
        minv = np.min(array)
        array = (array - minv) / (maxv - minv + 0.00001)
        return array

    def processing_image(self, image_nii):
        image_nii = self.geometric_transformation(image_nii)  #, interpolation='nearest')
        if self.apply_N4:
            image_nii = self.N4(image_nii)

        array = image_nii.get_fdata()
        affine = image_nii.affine

        # 为了增强对比度，取1%-99.9%的值
        percentile_99 = np.percentile(array, 99.95)
        # percentile_1 = np.percentile(array, 99.9) // 10
        # 方法1
        # array[array > percentile_99] = percentile_99
        # array[array < percentile_1] = 0
        # 方法2
        np.clip(array, 0, percentile_99, out=array)

        if self.apply_histeq:
            array = self.histeq(array)
        if self.apply_standardization:
            array = self.standardization(array)
        if self.apply_normalization:
            array = self.normalization(array)

        new_nii = nib.Nifti1Image(array, affine)
        return new_nii


class PreProcessingTrain(PreProcessing):

    def __init__(self,
                 path_image,
                 path_label,
                 path_liver_npy,
                 path_label_npy,
                 path_liver_preprocessed=None,
                 path_label_preprocessed=None,
                 ornt=None,
                 spacing=None,
                 shape=None,
                 data_type=None,
                 *args,
                 **kwargs):
        self.path_image = path_image
        self.path_label = path_label
        self.path_image_npy = path_liver_npy
        self.path_label_npy = path_label_npy
        if not os.path.exists(self.path_image_npy):
            os.makedirs(self.path_image_npy)
        if not os.path.exists(self.path_label_npy):
            os.makedirs(self.path_label_npy)
        if path_liver_preprocessed and path_label_preprocessed:
            self.path_image_preprocessed = path_liver_preprocessed
            self.path_label_preprocessed = path_label_preprocessed
            if not os.path.exists(self.path_image_preprocessed):
                os.makedirs(self.path_image_preprocessed)
            if not os.path.exists(self.path_label_preprocessed):
                os.makedirs(self.path_label_preprocessed)
        self.filenames = os.listdir(path_label)
        first_nii = nib.load(os.path.join(path_label, self.filenames[0]))
        if not ornt:
            ornt = nib.orientations.io_orientation(first_nii.affine)
        if not shape:
            shape = first_nii.shape
        if not spacing:
            spacing = np.asarray(first_nii.header.get_zooms())
        super().__init__(ornt, spacing, shape, data_type, *args, **kwargs)

    def processing_image_by_name(self, filename):
        image_nii = nib.load(os.path.join(self.path_image, filename))
        new_nii = self.processing_image(image_nii)
        return new_nii

    def processing_multi_dce_image_by_name(self, filename, dce):
        filename_0000 = filename.split('.')[0] + ('_0000.nii.gz')
        image_nii_0000 = nib.load(os.path.join(self.path_image, filename_0000))
        new_nii_0000 = self.processing_image(image_nii_0000)
        filename_0001 = filename.split('.')[0] + ('_0001.nii.gz')
        image_nii_0001 = nib.load(os.path.join(self.path_image, filename_0001))
        new_nii_0001 = self.processing_image(image_nii_0001)
        if dce == 2:
            return new_nii_0000, new_nii_0001
        elif dce == 4:
            filename_0002 = filename.split('.')[0] + ('_0002.nii.gz')
            image_nii_0002 = nib.load(os.path.join(self.path_image, filename_0002))
            new_nii_0002 = self.processing_image(image_nii_0002)
            filename_0003 = filename.split('.')[0] + ('_0003.nii.gz')
            image_nii_0003 = nib.load(os.path.join(self.path_image, filename_0003))
            new_nii_0003 = self.processing_image(image_nii_0003)
            return new_nii_0000, new_nii_0001, new_nii_0002, new_nii_0003
        else:
            raise ValueError('dce 仅支持2或4')

    def processing_label_by_name(self, filename):
        image_nii = nib.load(os.path.join(self.path_label, filename))
        new_nii = self.geometric_transformation(image_nii, interpolation='nearest')
        return new_nii

    def saving_npy(self, npy_path, filename, data, count):
        data = np.array(data).astype('float16')
        #### 修改.npy文件名 ####
        np.save(os.path.join(npy_path, filename.split('.')[0] + '_' + self.data_type + '_' + str(count)), data)

    def processing_nii2npy(self, filename, offset_low, offset_high):
        image_nii = self.processing_image_by_name(filename)
        label_nii = self.processing_label_by_name(filename)
        img = image_nii.get_fdata()
        lab = label_nii.get_fdata()
        lenth = img.shape[2]
        count = 0
        step = 2
        for i in range(offset_low, lenth - offset_high):
            count += 1
            dat_label_liver = lab[:, :, i]
            if np.max(dat_label_liver) != 0:
                # save label as npy
                self.saving_npy(npy_path=self.path_label_npy, filename=filename, data=dat_label_liver, count=count)
                # save data as npy
                dat_liver = img[:, :, i - step:i + step + 1]
                self.saving_npy(npy_path=self.path_image_npy, filename=filename, data=dat_liver, count=count)
            else:
                n = np.random.randint(10)
                if  n < 1:
                    print('i', i, 'n', n)
                    # save label as npy
                    self.saving_npy(npy_path=self.path_label_npy, filename=filename, data=dat_label_liver, count=count)
                    # save data as npy
                    dat_liver = img[:, :, i - step:i + step + 1]
                    self.saving_npy(npy_path=self.path_image_npy, filename=filename, data=dat_liver, count=count)

    def processing_nii2npy_all(self, offset_low, offset_high):
        for file in self.filenames:
            print(file)
            self.processing_nii2npy(file, offset_low, offset_high)

    def processing_nii2npy_tumor(self, filename, offset_low, offset_high):
        image_nii = self.processing_image_by_name(filename)
        label_nii = self.processing_label_by_name(filename)
        img = image_nii.get_fdata()
        lab = label_nii.get_fdata()
        lenth = img.shape[2]
        count = 0
        step = 2
        for i in range(offset_low, lenth - offset_high, 16):
            for j in range(80, 400, 32):
                for k in range(80, 400, 32):
                    dat_label_liver = lab[j:j+128, k:k+128, i]
                    if np.max(dat_label_liver) == 1:
                        count += 1
                        print(j,j+128, k,k+128, i)
                        # save label as npy
                        self.saving_npy(npy_path=self.path_label_npy, filename=filename, data=dat_label_liver, count=count)
                        # save data as npy
                        dat_liver = img[j:j+128, k:k+128, i - step:i + step + 1]
                        self.saving_npy(npy_path=self.path_image_npy, filename=filename, data=dat_liver, count=count)

    def processing_nii2npy_all_tumor(self, offset_low, offset_high):
        for file in self.filenames:
            print(file)
            self.processing_nii2npy_tumor(file, offset_low, offset_high)

    def processing_nii2npy_small_tumor(self, filename, enable_25D=False):
        image_nii = self.processing_image_by_name(filename)
        label_nii = self.processing_label_by_name(filename)
        img = image_nii.get_fdata()
        lab = label_nii.get_fdata()
        # lenth = img.shape[2]
        seg_shape = [lab.shape[0], lab.shape[1], lab.shape[2]]
        # count = 0
        [seg, num] = measure.label(lab, connectivity=1, return_num=True)
        step = 2
        for j in range(num):
            dat = np.zeros_like(seg)
            dat[seg == j + 1] = 1
            arr_list = np.where(dat==1)
            # print(np.max(arr_list[0]), np.min(arr_list[0]))
            # print(np.max(arr_list[1]), np.min(arr_list[1]))
            # print(np.max(arr_list[2]), np.min(arr_list[2]))
            x_max, x_min = np.max(arr_list[0]), np.min(arr_list[0])
            y_max, y_min = np.max(arr_list[1]), np.min(arr_list[1])
            z_max, z_min = np.max(arr_list[2]), np.min(arr_list[2])
            x_ = x_max - x_min
            y_ = y_max - y_min
            z_ = np.max(arr_list[2]) - np.min(arr_list[2])
            print('max(x_, y_)', max(x_, y_))
            if max(x_, y_) < 32:
                cut_more_x = (64 - x_) / 2
                if (64 - x_) % 2  == 0:
                    cut_more_x_left = int(cut_more_x)
                    # cut_more_x_right = int(cut_more_x)
                else:
                    cut_more_x_left = int(cut_more_x + 1)
                    # cut_more_x_right = int(cut_more_x)
                cut_more_y = (64 - y_) / 2
                if (64 - y_) % 2  == 0:
                    cut_more_y_left = int(cut_more_y)
                    # cut_more_y_right = cut_more_y
                else:
                    cut_more_y_left = int(cut_more_y + 1)
                    # cut_more_y_right = int(cut_more_y)
                left_top_point = [int(x_min-cut_more_x_left), int(y_min-cut_more_y_left)]
                right_bottom_point = [int(x_min-cut_more_x_left+64), int(y_min-cut_more_y_left+64)]
                # 如果最大切割位置超过图像x,y ， 重新从x-64和y-64赋值
                if int(x_min-cut_more_x_left+64) > seg_shape[0]:
                    right_bottom_point[0] = seg_shape[0]
                    left_top_point[0] = seg_shape[0] - 64
                if int(y_min-cut_more_y_left+64) >seg_shape[1]:
                    right_bottom_point[1] = seg_shape[1]
                    left_top_point[1] = seg_shape[1] - 64
            else:
                continue
            print(left_top_point, right_bottom_point)
            print(right_bottom_point[0] - left_top_point[0], right_bottom_point[1] - left_top_point[1])
            for z in range(z_min, z_max+1):
                if z > 1 and z < seg_shape[2]-1: # 不取最上和最下两层
                    # 原图
                    if enable_25D:
                        if z == 2:
                            z_ = 0
                            z__ = 5
                        elif z == seg_shape[2]-2:
                            z_ = z - 2
                            z__ = seg_shape[2]
                        else:
                            z_ = z-step
                            z__ = z+step+1
                        if z__ - z_ != 5:
                            continue
                        new_img = img[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z_:z__] # 2.5D --图像不取最上和最下两层
                        # print(new_img.shape)
                    else:
                        new_img = img[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z] # 2D
                    self.saving_npy(npy_path=self.path_image_npy, filename=filename, data=new_img, count=z)
                    # Only for visualization
                    # new_img = img[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    # new_img = new_img * 256
                    # cv2.imwrite(os.path.join(self.path_image_npy, filename.split('.')[0] + '_' + self.data_type + '_' + str(z)+'.jpg'), new_img)

                    # 标签
                    new_dat = dat[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    # print('np.max new_dat',np.max(new_dat))
                    self.saving_npy(npy_path=self.path_label_npy, filename=filename, data=new_dat, count=z)
                    # Only for visualization
                    # new_dat[new_dat==1] = 256
                    # cv2.imwrite(os.path.join(self.path_label_npy, filename.split('.')[0] + '_' + self.data_type + '_' + str(z)+'.jpg'), new_dat)

    def processing_nii2npy_small_tumor_multi_dce(self, filename, dce=2):
        if dce == 2:
            image_nii_0000, image_nii_0001 = self.processing_multi_dce_image_by_name(filename, dce)
        elif dce == 4:
            image_nii_0000, image_nii_0001, image_nii_0002, image_nii_0003 = self.processing_multi_dce_image_by_name(filename, dce)
            img_0002 = image_nii_0002.get_fdata()
            img_0003 = image_nii_0003.get_fdata()
        else:
            raise ValueError('dce 仅支持2或4')
        img_0000 = image_nii_0000.get_fdata()
        img_0001 = image_nii_0001.get_fdata()
        label_nii = self.processing_label_by_name(filename)
        lab = label_nii.get_fdata()
        seg_shape = [lab.shape[0], lab.shape[1], lab.shape[2]]
        [seg, num] = measure.label(lab, connectivity=1, return_num=True)
        for j in range(num):
            dat = np.zeros_like(seg)
            dat[seg == j + 1] = 1
            arr_list = np.where(dat==1)
            x_max, x_min = np.max(arr_list[0]), np.min(arr_list[0])
            y_max, y_min = np.max(arr_list[1]), np.min(arr_list[1])
            z_max, z_min = np.max(arr_list[2]), np.min(arr_list[2])
            x_ = x_max - x_min
            y_ = y_max - y_min
            z_ = np.max(arr_list[2]) - np.min(arr_list[2])
            print('max(x_, y_)', max(x_, y_))
            if max(x_, y_) < 32:
                cut_more_x = (64 - x_) / 2
                if (64 - x_) % 2  == 0:
                    cut_more_x_left = int(cut_more_x)
                else:
                    cut_more_x_left = int(cut_more_x + 1)
                cut_more_y = (64 - y_) / 2
                if (64 - y_) % 2  == 0:
                    cut_more_y_left = int(cut_more_y)
                else:
                    cut_more_y_left = int(cut_more_y + 1)
                left_top_point = [int(x_min-cut_more_x_left), int(y_min-cut_more_y_left)]
                right_bottom_point = [int(x_min-cut_more_x_left+64), int(y_min-cut_more_y_left+64)]
                # 如果最大切割位置超过图像x,y ， 重新从x-64和y-64赋值
                if int(x_min-cut_more_x_left+64) > seg_shape[0]:
                    right_bottom_point[0] = seg_shape[0]
                    left_top_point[0] = seg_shape[0] - 64
                if left_top_point[0] < 0:
                    left_top_point[0] = 0
                    right_bottom_point[0] = 64
                if int(y_min-cut_more_y_left+64) >seg_shape[1]:
                    right_bottom_point[1] = seg_shape[1]
                    left_top_point[1] = seg_shape[1] - 64
            else:
                continue
            print(left_top_point, right_bottom_point)
            print(right_bottom_point[0] - left_top_point[0], right_bottom_point[1] - left_top_point[1])
            for z in range(z_min, z_max+1):
                # 原图
                if dce == 4: # plain + arterial +  venous + delay
                    new_img = np.zeros(shape=(64,64,4))
                    new_img[:,:,0] = img_0000[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    new_img[:,:,1] = img_0001[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    new_img[:,:,2] = img_0002[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    new_img[:,:,3] = img_0003[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                elif dce == 2: # delay + arterial
                    new_img = np.zeros(shape=(64,64,2))
                    new_img[:,:,0] = img_0000[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                    new_img[:,:,1] = img_0001[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                print(new_img.shape)
                self.saving_npy(npy_path=self.path_image_npy, filename=filename, data=new_img, count=z)

                # 标签
                new_dat = dat[left_top_point[0]:right_bottom_point[0],left_top_point[1]:right_bottom_point[1],z]
                # print('np.max new_dat',np.max(new_dat))
                self.saving_npy(npy_path=self.path_label_npy, filename=filename, data=new_dat, count=z)

    def processing_nii2npy_all_small_tumor(self, enable_25D):
        for file in self.filenames:
            print(file)
            self.processing_nii2npy_small_tumor(file, enable_25D)

    def processing_nii2npy_all_small_tumor_multi_dce(self, dce):
        for file in self.filenames:
            print(file)
            self.processing_nii2npy_small_tumor_multi_dce(file, dce)

    def processing_nii(self, filename):
        image_nii = self.processing_image_by_name(filename)
        nib.save(image_nii, os.path.join(self.path_image_preprocessed, filename))
        label_nii = self.processing_label_by_name(filename)
        nib.save(label_nii, os.path.join(self.path_label_preprocessed, filename))

    def processing_nii_all(self):
        for file in self.filenames:
            print(file)
            self.processing_nii(file)
