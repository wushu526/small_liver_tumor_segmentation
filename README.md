#nnDetection

# Installation

## Docker

The easiest way to get started with nnDetection is the provided is to build a Docker Container with the provided Dockerfile.

Please install docker and [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) before continuing.

All projects which are based on nnDetection assume that the base image was built with the following tagging scheme `nnDetection:[version]`.
To build a container (nnDetection Version 0.1) run the following command from the base directory:

```bash
docker build -t nndetection:0.1 --build-arg env_det_num_threads=6 --build-arg env_det_verbose=1 .
```

(`--build-arg env_det_num_threads=6` and `--build-arg env_det_verbose=1` are optional and are used to overwrite the provided default parameters)

The docker container expects data and models in its own `/opt/data` and `/opt/models` directories respectively.
The directories need to be mounted via docker `-v`. For simplicity and speed, the ENV variables `det_data` and `det_models` can be set in the host system to point to the desired directories. To run:

```bash
docker run --gpus all -v ${det_data}:/opt/data -v ${det_models}:/opt/models -it --shm-size=24gb nndetection:0.1 /bin/bash
```

Warning:
When running a training inside the container it is necessary to [increase the shared memory](https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container) (via --shm-size).


## Source

*Please note that nndetection requires Python 3.8+.*

1. Install CUDA (>10.1) and cudnn (make sure to select [compatible versions](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)!)
2. [Optional] Depending on your GPU you might need to set `TORCH_CUDA_ARCH_LIST`, check [compute capabilities](https://developer.nvidia.com/cuda-gpus) here.
3. Install [torch](https://pytorch.org/) (make sure to match the pytorch and CUDA versions!) (requires pytorch >1.7+) and [torchvision](https://github.com/pytorch/vision)(make sure to match the versions!).
4. Clone nnDetection, `cd [path_to_repo]` and `pip install -e .`
5. Set environment variables (more info can be found below):
   - `det_data`: [required] Path to the source directory where all the data will be located
   - `det_models`: [required] Path to directory where all models will be saved
   - `OMP_NUM_THREADS=1` : [required] Needs to be set! Otherwise bad things will happen... Refer to batchgenerators documentation.
   - `det_num_threads`: [recommended] Number processes to use for augmentation (at least 6, default 12)
   - `det_verbose`: [optional] Can be used to deactivate progress bars (activated by default)
   - `MLFLOW_TRACKING_URI`: [optional] Specify the logging directory of mlflow. Refer to the [mlflow documentation](https://www.mlflow.org/docs/latest/tracking.html) for more information.

Note: nnDetection was developed on Linux => Windows is not supported.

<details close>
<summary>Test Installation</summary>
<br>
Run the following command in the terminal (!not! in pytorch root folder) to verify that the compilation of the C++/CUDA code was successfull:


```bash
python -c "import torch; import nndet._C; import nndet"
```

To test the whole installation please run the Toy Data set example.
</details>

<details close>
<summary>Maximising Training Speed</summary>
<br>
To get the best possible performance we recommend using CUDA 11.0+ with cuDNN 8.1.X+ and a (!)locally compiled version(!) of Pytorch 1.7+
</details>



### 代码流程

```bash
conda activate nndetection39
# python=3.9
# 找了好多pytorch与cuda版本，这个环境在8004上是没问题的
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
# 首先数据要先把原图和标签预处理到nnDetection需要的文件夹和文件格式，不同占位通过连通域自动生成连续存储的不同数值标签，这时可以选择去除不同的连通域大小的区域，尽量保证后续nndet_prep预处理不会报错，处理完再进行以下操作
# 小的连通域预处理后会消失就与json信息对不上，还有就是必须要让不同连通域的标签值连续（与每个json里的数值对应上），不然预处理也会报错的。注意预处理出错的数据，看看是不是标签和json值对应不上？
# 将原肿瘤或占位标签处理成多个单独的小肿瘤或占位，同时删除大于某长径的或小于某连通域的小肿瘤或占位
# conda环境需要有radiomics库
MainCode/cut_tumor2nndetection.py
# 预处理
CUDA_VISIBLE_DEVICES=3 nndet_prep 117
# 如果经常出现dimension的错误，就先拿一小部分训练集预处理，到测试gpu对不同大小crop的性能的时候中断程序，然后把所有数据放回去再重新预处理
# 如果预处理还是报错，再针对报错的标签修改，去除某些在预处理后会消失的seg值
nndet_unpack /home3/HWGroup/wushu/LUNA16/Task117_LiverTumorSmall/preprocessed/D3V001_3d/imagesTr 6
# 训练
CUDA_VISIBLE_DEVICES=3 nndet_train 117
# 防止与服务器断开跑nohup，最好先进入某个路径下再跑，不然找不到.out文件
cd nnDetection/
CUDA_VISIBLE_DEVICES=2 nohup nndet_train 117 > 117.out 2>&1 &
# sweep
CUDA_VISIBLE_DEVICES=3 nndet_sweep 117 RetinaUNetV001_D3V001_3d 0
# 预测测试集数据
CUDA_VISIBLE_DEVICES=3 nndet_predict 117 RetinaUNetV001_D3V001_3d --fold 0
# boxes2nii
nndet_boxes2nii 117 RetinaUNetV001_D3V001_3d --test
nndet_boxes2nii 117 RetinaUNetV001_D3V001_3d --threshold 0.9 --test
# 将nnDetection预测的box变成椭圆形状（附加功能）- conda环境需要有opencv,mitk库
MainCode/boxes_ellipse.py
```


### 5折交叉验证

`CUDA_VISIBLE_DEVICES=3 nndet_train 111 -o exp.fold=4 --sweep`

`nndet_sweep 111 RetinaUNetV001_D3V001_3d 0`

### 评估

`nndet_eval 111 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes`

### 从折叠中复制所有模型和预测 只有一个fold

`nndet_consolidate 016 RetinaUNetV001_D3V001_3d --sweep_boxes --num_folds 1`

### 预测

`nndet_predict 111 RetinaUNetV001_D3V001_3d --fold 0` 预测测试集

`nndet_predict 111 RetinaUNetV001_D3V001_3d --fold 0 --test`

`CUDA_VISIBLE_DEVICES=38877MiB nndet_predict 115 RetinaUNetV001_D3V001_3d --fold 0`

### 结果转换boxes2nii

`nndet_boxes2nii 115 RetinaUNetV001_D3V001_3d --test`测试集

`nndet_boxes2nii 111 RetinaUNetV001_D3V001_3d --threshold 0.9 --test` 测试集

`nndet_boxes2nii 016 RetinaUNetV001_D3V001_3d`

注意：要在不同的服务器上进行单独的环境搭建，进行pip install -e .的时候会build新的
