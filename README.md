
# DQPGT

This is the official code for "DQPGT: Dynamic Quadruple Priors Guided Transformer for Low-light Image Enhancement."


## Abstract

In the task of low-light image enhancement, methods based on Retinex theory assume smooth illumination that often conflicts with complex real-world lighting conditions. This discrepancy leads to an illumination component that inadequately represents the actual lighting information. Such deviations further cause distortion in the derived reflectance component. When this reflectance component serves as guidance for the model, the enhanced images exhibit issues such as uneven exposure, amplified noise, and color distortion. To address the aforementioned problems, this paper proposes a Dynamic Quadruple Priors Guided Transformer (DQPGT) method based on the Kubelka-Munk theory. This method first utilizes the Dynamic Quadruple Priors Estimator (DQPE) to construct an illumination-invariant and spatially-adaptive quadruple prior feature, which incorporates the color and structural information of low-light images. This feature serves as guiding information for the Corruption Restorer (CR), directing the multi-head self-attention mechanism in the Transformer to model different regions of the image, thereby enabling precise region-specific enhancement. Meanwhile, addressing the difficult problem of separating noise and high-frequency details in multi-channel processing, the introduction of a frequency-domain channel attention mechanism calibrates cross-channel feature responses within the frequency-domain. It enhances feature channels dominated by high-frequency components while suppressing noise-dominant channels, thereby collaboratively optimizing features across both spatial and frequency dimensions. Extensive experiments demonstrate that the DQPGT method significantly outperforms state-of-the-art approaches in both quantitative and qualitative evaluations across multiple datasets, and exhibits excellent robustness and practicality in downstream computer vision tasks. 

For the Chinese link, please see[CN](https://github.com/LiuRuisen-star/DQPGT)


### 1. Download the project.

Please run the following command to ensure that you deploy our project locally.

```python
git clone https://github.com/LiuRuisen-star/DQPGT.git
```

### 2. Create Environment


- Make Conda Environment
```
conda create -n DQPGT python=3.7
conda activate DQPGT
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

### 2. Prepare Dataset
If `data` is empty，please download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/11M-HE0JTIiIaBN6v-rAkgw?pwd=8ukk) or [Google Drive](https://drive.google.com/drive/folders/1v5v03sDxqWjybB-PwPFuYWE1RFV36G3_?usp=sharing) and place the data file in the DQPGT folder.

**Note:** 
Please download the `text_list.txt` and then put it into the folder `data/SMID/SMID_Long_np/`

The final placement should be as follows:

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |--outdoor_static_np
    |    |    |    |--input
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |    |    |--GT
    |    |    |    |    |--MVI_0898
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |    |--MVI_0918
    |    |    |    |    |   |--0001.npy
    |    |    |    |    |   |--0002.npy
    |    |    |    |    |    ...
    |    |    |    |     ...
    |    |--SID
    |    |    |--short_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--long_sid2
    |    |    |    |--00001
    |    |    |    |    |--00001_00_0.04s.npy
    |    |    |    |    |--00001_00_0.1s.npy
    |    |    |    |    |--00001_01_0.04s.npy
    |    |    |    |    |--00001_01_0.1s.npy
    |    |    |    |     ...
    |    |    |    |--00002
    |    |    |    |    |--00002_00_0.04s.npy
    |    |    |    |    |--00002_00_0.1s.npy
    |    |    |    |    |--00002_01_0.04s.npy
    |    |    |    |    |--00002_01_0.1s.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |--SMID
    |    |    |--SMID_LQ_np
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...
    |    |    |--SMID_Long_np
    |    |    |    |--text_list.txt
    |    |    |    |--0001
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |    |--0002
    |    |    |    |    |--0001.npy
    |    |    |    |    |--0002.npy
    |    |    |    |     ...
    |    |    |     ...

```


### 3. Testing

Please ensure that the `pretrained_weights` folder contains our pre-trained weights. If your weight files are missing, please download them from [Baidu Netdisk](https://pan.baidu.com/s/1FvDJlTyz8LBXxQZuqS4GYQ?pwd=um55) or [Google Drive](https://drive.google.com/drive/folders/1U9x7LOJo6XiTmMtBkNuCzoeQrGaPSGsG?usp=sharing).

```shell
# activate the environment
conda activate DQPGT

# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic

# SID
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_SID.yml --weights pretrained_weights/SID.pth --dataset SID

# SMID
python3 Enhancement/test_from_dataset.py --opt Options/QuadPriorFormer_SMID.yml --weights pretrained_weights/SMID.pth --dataset SMID

# Unsupervised Datasets
python3 Enhancement/test_from_nomonitor.py --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset DICM

python3 Enhancement/test_from_nomonitor.py --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LIME

python3 Enhancement/test_from_nomonitor.py --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset MEF

python3 Enhancement/test_from_nomonitor.py --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset NPE

python3 Enhancement/test_from_nomonitor.py --opt Options/QuadPriorFormer_NoMonitor.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset VV
```

### Evaluating the Params and FLOPS of models
Please run `Enhancement/test_flops_para.py` to test the parameters (Params) and floating-point operations (FLOPS) of DQPGT.

### 4. Training
Please ensure that you have fully completed the environment setup and can correctly infer the parameters and floating points.

```shell
# activate the enviroment
conda activate DQPGT

# LOL-v1
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_real.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/QuadPriorFormer_LOL_v2_synthetic.yml

# SID
python3 basicsr/train.py --opt Options/QuadPriorFormer_SID.yml

# SMID
python3 basicsr/train.py --opt Options/QuadPriorFormer_SMID.yml

```

### 5.Acknowledgments

We thank the following article and the authors for their open-source codes.

```
@article{retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Yuanhao Cai and Hao Bian and Jing Lin and Haoqian Wang and Radu Timofte and Yulun Zhang},
  journal={2023 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023},
  pages={12470-12479},
  url={https://api.semanticscholar.org/CorpusID:257496232}
}

@INPROCEEDINGS{PQP,
  author={Wang, Wenjing and Yang, Huan and Fu, Jianlong and Liu, Jiaying},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Zero-Reference Low-Light Enhancement via Physical Quadruple Priors}, 
  year={2024},
  volume={},
  number={},
  pages={26057-26066},
  keywords={Training;Limiting;Lighting;Diffusion models;Distortion;Robustness;Pattern recognition;Low-light enhancement;diffusion;zero-reference;low-level vision;image processing},
  doi={10.1109/CVPR52733.2024.02462}}
```

