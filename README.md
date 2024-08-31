# EndmemberNet
Official Implementation for "Real-time Deep Learning Spectral Imaging In vivo".
## Online Demo

We provide a live demo for EndmemberNet at http://fdudml.cn:6789. You can also use the colab <a target="_blank" href="https://colab.research.google.com/github/cxm12/UNiFMIR/blob/main/UniFMIR.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>, the openxlab app [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/ryanhe312/UniFMIR) or employ the following steps to run the demo locally.

![demo1](G:\dl\unmixing\ui\github\demo\demo1.png)

## User Interface for EndmemberNet

1. Download the Finetuned Models

You can download the finetuned models and the examples of EndmemberNet from [the release](https://github.com/cxm12/UNiFMIR/releases). Then, you can 'tar -xzvf' the file and put the folder in the root directory of EndmemberNet .

```
exampledata/
    Unmixing/

checkpoints/
    detection/
    segmentation/
```

1. Install Packages 

* We use Anaconda to create enviroment.

```
conda create -n unmixing python=3.9
conda activate unmixing
```

* Pytorch 2.0.1, CUDA 11.7 and CUDNN 

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

* Python Packages: 

You can install the required python packages by the following command:

```
pip install ultralytics gitpython opencyv-python lmdb imageio scikit-image tensorboard medpy
```

2. Run the Web Interface

You can run the web interface by the following command:

```
python app.py
```

Then, you can visit the web interface at [http://127.0.0.1:7866/](http://127.0.0.1:7866/). You can upload your own image or use our examples to run EndmemberNet.

## Train and Test EndmemberNet

### 1. Prepare the datasets

You can download our preprocessed data from [the Zenodo repository](https://doi.org/10.5281/zenodo.8401470) and extract it into the `detection/` folder. The data path should be structured as follows:

```
detection/
    data/
    	hyps/
    	images/
    	scripts/
    	unmixing/
```

### 2. Train detection model

Run the following code to train and test the detection model:

```
cd detection

CUDA_VISIBLE_DEVICES=0 python train.py --data data/unmixing/unmixing.yaml --name unmixing

sed -i 's/\r//' test.sh && bash test.sh
```

You can then find the trained detection model and the test dataset results in the `detection/runs/train/` and `detection/runs/detect/` folders, respectively.

### 3. Train segmentation model

Run the following code to train and test the segmentation model:

```
cd segmentation

CUDA_VISIBLE_DEVICES=0 python train.py

CUDA_VISIBLE_DEVICES=0 python test.py
```

You can then find the trained segmentation model and the test dataset results in the ` segmentation/train_log/` and `metric/` folders, respectively.

### 4. Train segmentation model

Run the following code to generate the unmixing results in the `unmix/` folder:

```
python unmixing.py
```

### 5. Demo

Copy `detection/runs/train/unmixing/weights/best.pt` to `checkpoints/detection/`, and copy `segmentation/train_log/checkpoints/detection/multimodal.pkl` to `checkpoints/segmentation/`. After that, run the following command:

```
python demo.py
```

Then, you can visit the web interface at [http://127.0.0.1:7866/](http://127.0.0.1:7866/). You can upload your own image or use our examples to run EndmemberNet.


## CITATION

If you use this code for your research, please cite our paper.

```bibtex

```