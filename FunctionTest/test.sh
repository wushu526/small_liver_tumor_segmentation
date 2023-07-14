CUDA_VISIBLE_DEVICES=3 nndet_sweep 131 RetinaUNetV001_D3V001_3d 0
CUDA_VISIBLE_DEVICES=3 nndet_predict 131 RetinaUNetV001_D3V001_3d --fold 0
nndet_boxes2nii 131 RetinaUNetV001_D3V001_3d --test
