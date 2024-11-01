# KGEN

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.1.0/) and then put the files in `data/mimic_cxr`.

## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

or Run the following code in a terminal

```cmd
python main_train.py --useKG --useVTFCM --AM SA
```



## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

or Run the following code in a terminal

```cmd
python main_test.py --useKG --useVTFCM --AM SA --load results/KGEN/model_best_7.pth
```

Follow [CheXpert](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert) or [CheXbert](https://github.com/stanfordmlgroup/CheXbert) to extract the labels and then run `python compute_ce.py`. Note that there are several steps that might accumulate the errors for the computation, e.g., the labelling error and the label conversion. We refer the readers to those new metrics, e.g., [RadGraph](https://github.com/jbdel/rrg_emnlp) and [RadCliQ](https://github.com/rajpurkarlab/CXR-Report-Metric).



## VK-Mamba

### Installation

1. Install the official Mamba library by following the instructions in the [hustvl/Vim](https://github.com/hustvl/Vim) repository.
2. After installing the official Mamba library, replace the mamba_simpy.py file in the installation directory with the mamba_simpy.py file provided in this mamba block directory.

### VK-Mamba

The Mamba Block modules introduced in the paper are available in the mamba_module.py file in this directory. These modules offer advanced features for pan-sharpening and can be seamlessly integrated into your existing projects.

