from preprocess import PreProcessingTrain



if __name__ == "__main__":
    # path_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/sort/pure_liver/'
    # path_label_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/sort/tumor/'
    # path_liver_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/preprocessed/tumor/pure_liver_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/preprocessed/tumor/label_tumor_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/preprocessed/tumor/pure_liver_preprocessed/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/v6.0/preprocessed/tumor/label_tumor_preprocessed/'

    # data_v1.7
    # path_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/sort/pure_liver/'
    # path_label_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/sort/tumor/'
    # path_liver_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/preprocessed/2.5D/tumor/pure_liver_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/preprocessed/2.5D/tumor/tumor_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/preprocessed/2.5D/tumor/pure_liver_preprocessed/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.7/preprocessed/2.5D/tumor/tumor_preprocessed/'

    # # data_v1.8
    # path_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/sort/pure_liver_reslice/'
    # path_label_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/sort/tumor/'
    # path_liver_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/preprocessed/2.5D/tumor/pure_liver_reslice_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/preprocessed/2.5D/tumor/tumor_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/preprocessed/2.5D/tumor/pure_liver_reslice_preprocessed/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.8/preprocessed/2.5D/tumor/tumor_preprocessed/'
    
    # data_v1.9
    # path_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/sort/pure_liver_reslice/'
    # path_label_liver = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/sort/tumor/'
    # path_liver_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/preprocessed/2.5D/tumor/pure_liver_reslice_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/preprocessed/2.5D/tumor/tumor_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/preprocessed/2.5D/tumor/pure_liver_reslice_preprocessed/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/quanjing_MR/quanjing_MR_data/data_v1.9/preprocessed/2.5D/tumor/tumor_preprocessed/'

    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     path_liver_preprocessed=path_image_preprocessed,
    #     path_label_preprocessed=path_label_preprocessed,
    #     ornt=["L", "A", "S"],
    #     # spacing=[1, 1, 2.5],
    #     # spacing=[1, 1, 1],
    #     # shape=[400, 400],
    #     spacing=[0.7576, 0.7576, 0.7576],
    #     shape=[528, 528],
    #     data_type='liver')

    # # preprocessing_train.processing_nii_all()
    # preprocessing_train.processing_nii2npy_all(offset_low=2, offset_high=2)  # offset_high>2
    
    # small liver tumor
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/imagesTr_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/labelsTr_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/imagesTr_/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/labelsTr_/'
    
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/imagesTs/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/labelsTs/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/imagesTs_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/labelsTs_npy/'
    # path_image_preprocessed = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/imagesTs_/'
    # path_label_preprocessed = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/preprocessed/labelsTs_/'

    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.0/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.1/preprocessed/imagesTr_npy_1/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.1/preprocessed/labelsTr_npy_1/'
    
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.2/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.2/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.2/preprocessed/imagesTr_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.1.2/preprocessed/labelsTr_npy/'
    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     # path_liver_preprocessed=path_image_preprocessed,
    #     # path_label_preprocessed=path_label_preprocessed,
    #     ornt=["L", "A", "S"],
    #     # spacing=[1, 1, 2.5],
    #     # spacing=[1, 1, 1],
    #     # shape=[400, 400],
    #     spacing=[0.9, 0.9, 3.5],
    #     shape=[400, 400],
    #     data_type='liver')
    # # preprocessing_train.processing_nii_all()
    # preprocessing_train.processing_nii2npy_all_small_tumor()

    ########## 第三批
    # # 原图与标签 2D
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/preprocessed/imagesTr_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/preprocessed/labelsTr_npy/'
    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     ornt=["L", "A", "S"],
    #     spacing=[0.9, 0.9, 3.5],
    #     shape=[400, 400],
    #     data_type='liver')
    # # preprocessing_train.processing_nii_all()
    # preprocessing_train.processing_nii2npy_all_small_tumor()

    # # resize的原图和标签 2D
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/preprocessed/imagesTr_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/preprocessed/labelsTr_npy/'
    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     ornt=["L", "A", "S"],
    #     spacing=[0.9, 0.9, 3.5],
    #     shape=[400, 400],
    #     data_type='liver')
    # # preprocessing_train.processing_nii_all()
    # preprocessing_train.processing_nii2npy_all_small_tumor()

    # # 原图与标签 2.5D
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/preprocessed/imagesTr_25D_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.0/preprocessed/labelsTr_25D_npy/'
    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     ornt=["L", "A", "S"],
    #     spacing=[0.9, 0.9, 3.5],
    #     shape=[400, 400],
    #     data_type='liver')
    # preprocessing_train.processing_nii2npy_all_small_tumor(enable_25D=True)

    # # resize的原图和标签 2.5D
    # path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/sort/imagesTr/'
    # path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/sort/labelsTr/'
    # path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/preprocessed/imagesTr_25D_npy/'
    # path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.3.1/preprocessed/labelsTr_25D_npy/'
    # preprocessing_train = PreProcessingTrain(
    #     path_liver,
    #     path_label_liver,
    #     path_liver_npy=path_liver_npy,
    #     path_label_npy=path_label_npy,
    #     ornt=["L", "A", "S"],
    #     spacing=[0.9, 0.9, 3.5],
    #     shape=[400, 400],
    #     data_type='liver')
    # preprocessing_train.processing_nii2npy_all_small_tumor(enable_25D=True)

    ######### 前五批
    # resize的delay期原图和标签 2D
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/sort/imagesTr/'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/imagesTr_2D_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/labelsTr_2D_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor(enable_25D=False)

    # resize的delay期原图和标签 2.5D
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/sort/imagesTr/'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/imagesTr_25D_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/labelsTr_25D_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor(enable_25D=True)

    # resize的arterial+delay期原图和标签，ants配准, channels=2
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/sort/imagesTr/'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/preprocessed/imagesTr_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.1/preprocessed/labelsTr_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor_multi_dce(dce=2)
    # resize的arterial+delay期原图和标签，transmorph配准, channels=2
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/sort/imagesTr'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/preprocessed/imagesTr_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.2/preprocessed/labelsTr_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor_multi_dce(dce=2)
    
    # resize的plain+arterial+venous+delay期原图和标签，ants配准，channels=4
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/sort/imagesTr/'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/preprocessed/imagesTr_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.3/preprocessed/labelsTr_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor_multi_dce(dce=4)
    # resize的plain+arterial+venous+delay期原图和标签，transmorph配准，channels=4
    path_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/sort/imagesTr/'
    path_label_liver = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/sort/labelsTr/'
    path_liver_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/preprocessed/imagesTr_npy/'
    path_label_npy = '/home3/HWGroup/wushu/liver_tumor/data/v1.4.4/preprocessed/labelsTr_npy/'
    preprocessing_train = PreProcessingTrain(
        path_liver,
        path_label_liver,
        path_liver_npy=path_liver_npy,
        path_label_npy=path_label_npy,
        ornt=["L", "A", "S"],
        spacing=[0.9, 0.9, 3.5],
        shape=[400, 400],
        data_type='liver')
    preprocessing_train.processing_nii2npy_all_small_tumor_multi_dce(dce=4)