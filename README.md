# EfficientDet: Scalable and Efficient Object Detection

## Introduction

Here is our pytorch implementation of the model described in the paper **EfficientDet: Scalable and Efficient Object Detection** [paper](https://arxiv.org/abs/1911.09070) (*Note*: We also provide pre-trained weights, which you could see at ./trained_models) 
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of our model's output.</i>
</p>


## Datasets


| Dataset                | Classes |    #Train images      |    #Validation images      |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2017               |    80   |          118k         |              5k            |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── images
      ├── train2017 
      └── val2017
      └── test2017

  ```
  
## How to use our code

With our code, you can:
[Download train2017](https://www.kaggle.com/jipingsun/object-detection-obama)
Just copy the images in train2017 and paste in the train2017 directory of this repository and you are all set to train and test the model to recognize obama. [cheers!!!]
* **Train your model** by running **python main.py** 
* **Evaluate mAP for COCO dataset** by running **python mAP_evaluation.py**
* **Test your model for COCO dataset** by running **python main.py**
* **Test your model for video** by running **python main.py**
* **You can change almost the important parameters of model and do experimentation using config.ini file** and then comment the lines in main.py file to do the tasks accordingly **python main.py**

## Experiments

We trained our model by using 3 NVIDIA GTX 1080Ti. Below is mAP (mean average precision) for COCO val2017 dataset 

|   Average Precision   |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.314   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |      IoU=0.50     |   area=   all   |   maxDets=100   |   0.461   |
|   Average Precision   |      IoU=0.75     |   area=   all   |   maxDets=100   |   0.343   |
|   Average Precision   |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.093   |
|   Average Precision   |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.358   |
|   Average Precision   |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.517   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=1     |   0.268   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=10    |   0.382   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.403   |
|     Average Recall    |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.117   |
|     Average Recall    |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.486   |
|     Average Recall    |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.625   |


## Results

Some predictions are shown below:

<img src="demo/1.jpg" width="280"> <img src="demo/2.jpg" width="280"> <img src="demo/3.jpg" width="280">

<img src="demo/4.jpg" width="280"> <img src="demo/5.jpg" width="280"> <img src="demo/6.jpg" width="280">

<img src="demo/7.jpg" width="280"> <img src="demo/8.jpg" width="280"> <img src="demo/9.jpg" width="280">


## Requirements

* **python 3.6**
* **pytorch 1.2**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **pycocotools**
* **efficientnet_pytorch**

## References
- Mingxing Tan, Ruoming Pang, Quoc V. Le. "EfficientDet: Scalable and Efficient Object Detection." [EfficientDet](https://arxiv.org/abs/1911.09070).
- Our implementation borrows some parts from [RetinaNet.Pytorch](https://github.com/yhenon/pytorch-retinanet)
  

## Citation

    @article{EfficientDet,
        Author = {Deepak},
        Title = {A Pytorch Implementation of EfficientDet Object Detection},
        Reference = {https://github.com/signatrix/efficientdet},
        Year = {2020}
    }