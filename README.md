# fastMRI

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)
[![CircleCI](https://circleci.com/gh/facebookresearch/fastMRI.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/fastMRI)

[Website and Leaderboards](https://fastMRI.org) | [Dataset](https://fastmri.med.nyu.edu/) | [GitHub](https://github.com/facebookresearch/fastMRI) | [Publications](#list-of-papers)

Accelerating Magnetic Resonance Imaging (MRI) 通过获取更少的测量值，有可能降低医疗成本，将对患者的压力降到最低，并使MR成像在目前速度缓慢或昂贵的应用中成为可能。


[fastMRI](https://fastMRI.org) 
是Facebook AI Research（FAIR）和NYU Langone Health的一项合作研究项目，旨在研究使用AI来加快MRI扫描的速度。 NYU Langone Health已发布了完全匿名的膝盖和大脑MRI数据集，可以从[the fastMRI dataset page](https://fastmri.med.nyu.edu/)下载。可以找到与fastMRI项目相关的出版物 [at the end of this README](#list-of-papers).

该repository包含方便的PyTorch data loaders，subsampling 函数，评估指标以及简单基准方法的参考实现。它还包含fastMRI项目的某些出版物中方法的实现。

## Outline

1. [Documentation](#documentation)
2. [Dependencies and Installation](#Dependencies-and-Installation)
3. [Directory Structure & Usage](#directory-structure--usage)
4. [Testing](#testing)
5. [Training a model](#training-a-model)
6. [Submitting to the Leaderboard](#submitting-to-the-leaderboard)
7. [License](#license)
8. [List of Papers](#list-of-papers)

## Documentation

fastMRI数据集和基线重建性能的文档可以在[our paper on arXiv](https://arxiv.org/abs/1811.08839)中找到。本文会不断更新，以增加数据集和新的基准。如果在项目中使用fastMRI数据或代码，请考虑引用arXiv论文：

```BibTeX
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Matthew J. Muckley and Mary Bruno and Aaron Defazio and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and James Pinkerton and Duo Wang and Nafissa Yakubova and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```
对于代码文档，大多数函数和类都有随附的docstrings，您可以通过IPython中的“help”函数来访问它们。例如：

```python
from fastmri.data import SliceDataset

help(SliceDataset)
```

## Dependencies and Installation

We have tested this code using:

* Ubuntu 18.04
* Python 3.8
* CUDA 10.1
* CUDNN 7.6.5

首先根据安装说明安装PyTorch  [PyTorch Website](https://pytorch.org/get-started/) for your operating system and CUDA setup.

然后，导航到`fastmri`根目录并运行

```bash
pip install -e .
```

`pip` 将处理所有程序包依赖项。之后，您应该能够运行存储库中的大多数代码。

## Directory Structure & Usage

自2020年8月起，该repository已进行重构，以“fastmri”模块为中心的软件包进行操作，而可再现性的配置和脚本在“experimental”中。其他文件夹正在适应新结构，然后被弃用。

`fastmri`: 包含许多用于complex number math，coil combinations等的基本工具。

* `fastmri/data`: 包含原始`data`文件夹中的数据处理函数，可用于创建采样mask和提交文件。
* `fastmri/models`: 包含基线模型，包括U-Net和端到端Variational网络。 

`experimental`: 旨在帮助基线和论文重现的文件夹。

* `experimental/zero_filled`: Examples for saving images for leaderboard submission, zero-filled baselines from [fastMRI: An open dataset and benchmarks for accelerated MRI (Zbontar, J. et al., 2018)](https://arxiv.org/abs/1811.08839).
* `experimental/cs`: Compressed sensing baselines from [fastMRI: An open dataset and benchmarks for accelerated MRI (Zbontar, J. et al., 2018)](https://arxiv.org/abs/1811.08839).
* `experimental/unet`: U-Net baselines from [fastMRI: An open dataset and benchmarks for accelerated MRI (Zbontar, J. et al., 2018)](https://arxiv.org/abs/1811.08839).
* `experimental/varnet`: Code for reproducing [End-to-End Variational Networks for Accelerated MRI Reconstruction (Sriram, A. et al. 2020)](https://arxiv.org/abs/2004.06688).

Code for other papers can be found in:

* `banding_removal`: Code for reproducing [MRI Banding Removal via Adversarial Training (Defazio, A. et al., 2020)](https://arxiv.org/abs/2001.08699).
* `banding_removal/fastmri/common/subsample.py`: Code for implementing masks from [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry (Defazio, A., 2019)](https://arxiv.org/abs/1912.01101).

## Testing

Run `python -m pytest tests`.

## Training a model

The [data README](https://github.com/facebookresearch/fastMRI/tree/master/fastmri/data/README.md) 有一个关于如何加载数据和合并数据转换的简单样本。
This [jupyter notebook](https://github.com/facebookresearch/fastMRI/blob/master/fastMRI_tutorial.ipynb) 包含一个简单的教程，解释了如何开始使用数据。

Please look at [this U-Net demo script](https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/train_unet_demo.py) 有关如何使用PyTorch Lightning框架训练模型的样本。

## Submitting to the Leaderboard

在提供的测试数据上运行模型，并创建一个包含您的预测的zip文件。`fastmri` has a `save_reconstructions` function that saves the data in the correct format.

将zip文件上传到任何公共可访问的云存储（例如Amazon S3，Dropbox等）。提交指向zip文件的链接  [challenge website](https://fastmri.org/submit). 您需要先创建一个帐户，然后才能提交。

## License

fastMRI is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md).

## List of Papers

The following lists titles of papers from the fastMRI project. The corresponding abstracts, as well as links to preprints and code can be found [here](https://github.com/facebookresearch/fastMRI/blob/master/LIST_OF_PAPERS.md).

1. Zbontar, J., Knoll, F., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Zhang, Z. (2018). [fastMRI: An open dataset and benchmarks for accelerated MRI](https://arxiv.org/abs/1811.08839). *arXiv preprint arXiv:1811.08839*.
2. Zhang, Z., Romero, A., Muckley, M. J., Vincent, P., Yang, L., & Drozdzal, M. (2019). [Reducing uncertainty in undersampled MRI reconstruction with active acquisition](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Reducing_Uncertainty_in_Undersampled_MRI_Reconstruction_With_Active_Acquisition_CVPR_2019_paper.html). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2049-2058).
3. Defazio, A. (2019). [Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry](https://arxiv.org/abs/1912.01101). *arXiv preprint, arXiv:1912.01101*.
4. Defazio, A., Murrell, T., & Recht, M. P. (2020). [MRI Banding Removal via Adversarial Training](https://arxiv.org/abs/2001.08699). *arXiv preprint arXiv:2001.08699*.
5. Knoll, F., Zbontar, J., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., ... & Zhang, Z. (2020). [fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning](https://doi.org/10.1148/ryai.2020190007). *Radiology: Artificial Intelligence*, 2(1), e190007.
6. Knoll, F., Murrell, T., Sriram, A., Yakubova, N., Zbontar, J., Rabbat, M., ... & Recht, M. P. (2020). [Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge](https://doi.org/10.1002/mrm.28338). *Magnetic Resonance in Medicine*.
7. Sriram, A., Zbontar, J., Murrell, T., Zitnick, C. L., Defazio, A., & Sodickson, D. K. (2020). [GrappaNet: Combining parallel imaging with deep learning for multi-coil MRI reconstruction](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 14315-14322).
8. Recht, M. P., Zbontar, J., Sodickson, D. K., Knoll, F., Yakubova, N., Sriram, A., ... & Kline, M. (2020). [Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study](https://doi.org/10.2214/AJR.20.23313). *American Journal of Roentgenology*.
9. Pineda, L., Basu, S., Romero, A., Calandra, R., & Drozdzal, M. (2020). [Active MR k-space Sampling with Reinforcement Learning](https://arxiv.org/abs/2007.10469). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*.
10. Sriram, A., Zbontar, J., Murrell, T., Defazio, A., Zitnick, C. L., Yakubova, N., ... & Johnson, P. (2020). [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://arxiv.org/abs/2004.06688). In *International Conference on Medical Image Computing and Computer-Assisted Intervention*.
