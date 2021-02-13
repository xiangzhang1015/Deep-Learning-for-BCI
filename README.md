# Deep Learning-based Brain-Computer Interface (BCI) 

#### Author: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://linayao.com/) (lina.yao@unsw.edu.au) 

#### Contributor: [Ziyu Liu](https://github.com/ziyuliu-lion)


## Overview

This tutorial contains implementable python and jupyter notebook codes and benchmark datasets to learn how to recognize brain signals based on deep learning models. This tutorial associates our [survey on DL-based noninvasive brain signals](https://iopscience.iop.org/article/10.1088/1741-2552/abc902) and [book on DL-based BCI: Representations, Algorithms and Applications](https://www.worldscientific.com/worldscibooks/10.1142/q0282). 


- We present a new taxonomy of BCI signal paradigms according to the acquisition methods. ECOG: Electrocorticography, EEG: Electroencephalography, fNIRS: functional
near-infrared spectroscopy, fMRI: functional magnetic resonance imaging, EOG: Electrooculography, MEG: Magnetoencephalography.

<p align="center">
<img src="https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/BCI_signals.png" width="900" align="center">
</p>

- We systemically introduce the fundamental knowledge of deep learning models. 
MLP: Multi-Layer Perceptron, RNN: Recurrent Neural Networks, CNN: Convolutional Neural Networks, LSTM: Long Short-Term Memory, GRU: Gated Recurrent Units, AE: Autoencoder, RBM: Restricted Boltzmann Machine, DBN: Deep Belief Networks, VAE: Variational Autoencoder, GAN: Generative Adversarial Networks.
D-AE denotes Stacked-Autoencoder which refers to the autoencoder with multiple hidden layers. Deep Belief Network can be composed of AE or RBM, therefore, we divided DBN into DBN-AE (stacked AE) and DBN-RBM (stacked RBM).

<p align="center">
<img src="https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/dl_models.png" width="900" align="center">
</p>

- Moreover, the guidelines for the investigation and design of BCI systems are provided from the aspects of signal category, deep learning models, and applications. The following figures show the distribution on signals and DL models in frointer DL-based BCI studies.

Distribution on signals            | Distribution on DL models
:-------------------------:|:-------------------------:
![Distribution on signals](https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/bar_signal.png)  |  ![Distribution on DL models](https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/bar_model.png)



<br/><br/>

- Special attention has been given to the state-of-the-art studies on deep learning for EEG-based BCI research in terms of algorithms.
Specically, we introduces a number of advanced deep learning algorithms and frameworks aimed at several major issues in BCI including robust brain signal representation learning, cross-scenario classification, and semi-supervised classification.


- Furthermore, several novel prototypes of deep learning-based BCI systems are proposed which shed light on real-world applications such as authentication, visual reconstruction, language interpretation, and neurological disorder diagnosis. Such applications can dramatically benefit both healthy individuals and those with disabilities in real life. 


## BCI Dataset

Collection of brain signals is both financially and temporally costly. We extensively explore the benchmark data sets applicable to rain signal research and provides 31 public data sets with download links that cover most brain signal types.


|   Brain Signals   |Dataset  | #-Subject | #-Classes | Sampling Rate (Hz) | #-Channels | Download Link |
|--------------------|------|--------------------|-------------|-------|----|----|
| ECoG | BCI-C IV, Data set IV | 3 | 5 | 1000 |48-64| [link](http://www.bbci.de/competition/iv/) |


In order to let the readers have a quick access of the dataset and can play around it, we provide the well-processed and ready-ro-use [dataset](https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/dataset/) of EEG Motor Movement/Imagery Database ([EEGMMIDB](https://archive.physionet.org/pn4/eegmmidb/)). This dataset contains 109 subjects while the EEG signals are recorded in 64 channels with 160 Hz sampling rate. After our clearning and sorting, each npy file represents a subject: the data shape of each npy file is [N, 65], the first 64 columns correspond to 64 channel features, the last column denotes the class label. The N varies for different subjects, but N should be either 259520 or 255680. This is the inherent difference in the original dataset.



### Running the code  

In our [tutorial](https://github.com/xiangzhang1015/ML_BCI_tutorial/tree/main/tutorial/) files, you will learn the pipline and workflow of BCI system including data acquiction, pre-processing, feature extraction (optional), classification, and evaluation. We present necessary references and actionable codes of the most typical deep learning models (GRU, LSTM, CNN, GNN) while taking advantage of temporal, spatial, and typographical depencencies. We also provide [python codes](https://github.com/xiangzhang1015/ML_BCI_tutorial/tree/main/pythonscripts) that are very handy. For example, to check the EEG classification performance of CNN, run the following code:
```
python 4-2_CNN.py 
```

For PyTorch beginners, we highly recommond [Morvan Zhou's pyTorch Tutorials](https://github.com/MorvanZhou/PyTorch-Tutorial).

## Citing

If you find our research is useful for your research, please consider citing our survey or book:
```
@article{zhang2020survey,
  title={A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers},
  author={Zhang, Xiang and Yao, Lina and Wang, Xianzhi and Monaghan, Jessica JM and Mcalpine, David and Zhang, Yu},
  journal={Journal of Neural Engineering},
  year={2020},
  publisher={IOP Publishing}
}

@book{zhang2021deep,
  title={Deep Learning for EEG-based Brain-Computer Interface: Representations, Algorithms and Applications},
  author={Zhang, Xiang and Yao, Lina},
  year={2021},
  publisher={World Scientific Publishing}
}

```

## Requirements 

The tutorial codes are tested to work under Python 3.7. 

Recent versions of Pytorch, torch-geometric, numpy, and scipy are required. All the required basic packages can be installed using the following command:
'''
pip install -r requirements.txt
'''
*Note:* For [toch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and the related dependices (e.g., cluster, scatter, sparse), the higher version may work but haven't been tested yet.


## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <xiang_zhang@hms.harvard.edu>.

## License

This tutorial is licensed under the MIT License.

