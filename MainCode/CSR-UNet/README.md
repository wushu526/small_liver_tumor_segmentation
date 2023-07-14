#CSR-UNet

CSR-UNet main functions

runs -- training records
unet -- main 2.5D UNet class and functions 
preprocess.py -- Preprocess base class and functions
preprocess_tumor.py -- input images and labels and output preprocessed path
train_small_tumor_v140.py -- main training functions (Delay 1 phase)
predict_small_tumor140.py -- main training functions (Delay 1 phase)
tests_merge_total.py -- merge nnUNet, nnDetection and 2.5D UNet predicting nifty results to final result nifty
utils/evaluation_gongwei/count_missing_FP.py -- 统计通过diameter_FP.py和diameter_missing.py统计出来的list
utils/evaluation_gongwei/diameter_classes_numbers.py -- 统计不同肿瘤种类和不同长径下对应的肿瘤数量
utils/evaluation_gongwei/diameter_FP.py -- 计算统计预测结果和标签在不同长径下对应的分割指标和检测指标，并且统计假阳性数据和对应长径列表
utils/evaluation_gongwei/diameter_liange.py -- 计算统计预测结果和标签在良性和恶性分类下对应的分割指标和检测指标，并且统计假阳性数据和对应长径列表
utils/evaluation_gongwei/diameter_missing.py -- 计算统计预测遗漏的数据，占位和对应长径列表
utils/evaluation_gongwei/diameter_multiclasses.py -- 计算统计预测结果和标签在肿瘤不同分类下对应的分割指标和检测指标
utils/evaluation_gongwei/diameter_total.py -- 计算统计预测结果和标签在肿瘤所有数据的分割指标和检测指标
