# Deep Learning for Brain-Computer Interface (BCI) 

#### Author: [Dr. Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Prof. Lina Yao](https://linayao.com/) (lina.yao@unsw.edu.au) 

#### Tutorial contributor:  [Dr. Xiang Zhang](http://xiangzhang.info/), [Ziyu Liu](https://github.com/ziyuliu-lion)

## Update

**The whole well-processed and DL-ready data of 109 subjects from EEGMMIDB are uploaded!**

## Overview

This tutorial contains implementable python and jupyter notebook codes and benchmark datasets to learn how to recognize brain signals based on deep learning models. This tutorial associates our [survey on DL-based noninvasive brain signals](https://iopscience.iop.org/article/10.1088/1741-2552/abc902) and [book on DL-based BCI: Representations, Algorithms and Applications](https://www.worldscientific.com/worldscibooks/10.1142/q0282). 


- We present a new taxonomy of BCI signal paradigms according to the acquisition methods. ECOG: Electrocorticography, EEG: Electroencephalography, fNIRS: functional
near-infrared spectroscopy, fMRI: functional magnetic resonance imaging, EOG: Electrooculography, MEG: Magnetoencephalography.

<p align="center">
<img src="https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/BCI_signals.png" width="900" align="center">
</p>

- We systemically introduce the fundamental knowledge of deep learning models. 
MLP: Multi-Layer Perceptron, RNN: Recurrent Neural Networks, CNN: Convolutional Neural Networks, LSTM: Long Short-Term Memory, GRU: Gated Recurrent Units, AE: Autoencoder, RBM: Restricted Boltzmann Machine, DBN: Deep Belief Networks, VAE: Variational Autoencoder, GAN: Generative Adversarial Networks.
D-AE denotes Stacked-Autoencoder which refers to the autoencoder with multiple hidden layers. Deep Belief Network can be composed of AE or RBM, therefore, we divided DBN into DBN-AE (stacked AE) and DBN-RBM (stacked RBM).

<p align="center">
<img src="https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/dl_models.png" width="900" align="center">
</p>

- Moreover, the guidelines for the investigation and design of BCI systems are provided from the aspects of signal category, deep learning models, and applications. The following figures show the distribution on signals and DL models in frointer DL-based BCI studies.

Distribution on signals            | Distribution on DL models
:-------------------------:|:-------------------------:
![Distribution on signals](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/bar_signal.png)  |  ![Distribution on DL models](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/bar_model.png)

<br/><br/>

- Special attention has been given to the state-of-the-art studies on deep learning for EEG-based BCI research in terms of algorithms.
Specically, we introduces a number of advanced deep learning algorithms and frameworks aimed at several major issues in BCI including robust brain signal representation learning, cross-scenario classification, and semi-supervised classification.


- Furthermore, several novel prototypes of deep learning-based BCI systems are proposed which shed light on real-world applications such as authentication, visual reconstruction, language interpretation, and neurological disorder diagnosis. Such applications can dramatically benefit both healthy individuals and those with disabilities in real life. 


## BCI Dataset

Collection of brain signals is both financially and temporally costly. We extensively explore the benchmark data sets applicable to rain signal research and provides 31 public data sets with download links that cover most brain signal types.



| Brain Signals  |  Dataset    | #-Subject | #-Classes | Sampling Rate (Hz) | #-Channels | Download Link |
| :--------------------- | :------------------------------------- | :----------------- | :------------- | :-------------- | :------------------------------------------------------------ | :--- |
| FM EcoG                | BCI-C IV, Data set IV                  | 3                  | 5              | 1000            | 48 -- 64                                                      | [Link](http://www.bbci.de/competition/iv/) |
| MI EcoG                |   BCI-C III    <br>    Data set I      | 1                  | 2              | 1000            | 64                                                            | [Link](http://www.bbci.de/competition/iii/) |
| Sleeping EEG           | Sleep-EDF Telemetry                    | 22                 | 6              | 100             | 2 EEG, 1 EOG, 1 EMG                                           | [Link](https://physionet.org/physiobank/database/sleep-edfx/) |
| Sleeping EEG           | Sleep-EDF: Cassette                    | 78                 | 6              | 100, 1          |   2 EEG (100 Hz), 1 EOG (100 Hz),    <br>    1 EMG (1 Hz)    | [Link](https://physionet.org/physiobank/database/sleep-edfx/) |
| Sleeping EEG           | MASS-1                                 | 53                 | 5              | 256             | 17/19 EEG, 2 EOG, 5 EMG                                       | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG           | MASS-2                                 | 19                 | 6              | 256             | 19 EEG, 4 EOG, 1EMG                                         | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG           | MASS-3                                 | 62                 | 5              | 256             | 20 EEG, 2 EOG, 3 EMG                                          | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG           | MASS-4                                 | 40                 | 6              | 256             | 4 EEG, 4 EOG, 1 EMG                                          | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG           | MASS-5                                 | 26                 | 6              | 256             | 20 EEG, 2 EOG, 3 EMG                                          | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG           | SHHS                                   | 5804               | N/A            | 125, 50         |   2 EEG (125 Hz), 1EOG (50 Hz),    <br>    1 EMG (125 Hz)     | [Link](https://physionet.org/pn3/shhpsgdb/) |
| Seizure EEG            | CHB-MIT                                | 22                 | 2              | 256             | 18                                                            | [Link](https://physionet.org/pn6/chbmit/) |
| Seizure EEG            | TUH                                    | 315                | 2              | 200             | 19                                                            | [Link](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) |
| MI EEG                 | EEGMMI                                 | 109                | 4              | 160             | 64                                                            | [Link](https://physionet.org/pn4/eegmmidb/) |
| MI EEG                 | BCI-C II, Data set III                 | 1                  | 2              | 128             | 3                                                             | [Link](http://www.bbci.de/competition/ii/) |
| MI EEG                 | BCI-C III, Data set III a              | 3                  | 4              | 250             | 60                                                           | [Link](http://www.bbci.de/competition/iii/) |
| MI EEG                 | BCI-C III, Data set III b              | 3                  | 2              | 125             | 2                                                             | [Link](http://www.bbci.de/competition/iii/) |
| MI EEG                 | BCI-C III, Data set IV a               | 5                  | 2              | 1000            | 118                                                           | [Link](http://www.bbci.de/competition/iii/) |
| MI EEG                 | BCI-C III, Data set IV b               | 1                  | 2              | 1001            | 119                                                           | [Link](http://www.bbci.de/competition/iii/) |
| MI EEG                 | BCI-C III, Data set IV c               | 1                  | 2              | 1002            | 120                                                           | [Link](http://www.bbci.de/competition/iii/) |
| MI EEG                 | BCI-C IV, Data set I                   | 7                  | 2              | 1000            | 64                                                            | [Link](http://www.bbci.de/competition/iv/) |
| MI EEG                 | BCI-C IV, Data set II a                | 9                  | 4              | 250             | 22 EEG, 3 EOG                                                 | [Link](http://www.bbci.de/competition/iv/) |
| MI EEG                 | BCI-C IV, Data set II b                | 9                  | 2              | 250             | 3 EEG, 3 EOG                                                  | [Link](http://www.bbci.de/competition/iv/) |
| Emotional EEG          | AMIGOS                                 | 40                 | 4              | 128             | 14                                                            | [Link](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html) |
| Emotional EEG          | SEED                                   | 15                 | 3              | 200             | 62                                                            | [Link](http://bcmi.sjtu.edu.cn/~seed/download.html) |
| Emotional EEG          | DEAP                                   | 32                 | 4              | 512             | 32                                                            | [Link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| Others EEG             | Open MIIR                              | 10                 | 12             | 512             | 64                                                            | [Link](https://owenlab.uwo.ca/research/the_openmiir_dataset.html) |
| VEP                    | BCI-C II, Data set II b                | 1                  | 36             | 240             | 64                                                           | [Link](http://www.bbci.de/competition/ii/) |
| VEP                    | BCI-C III, Data set II                 | 2                  | 26             | 240             | 64                                                            | [Link](http://www.bbci.de/competition/iii/) |
| fMRI                   | ADNI                                   | 202                | 3              | N/A             | N/A                                                           | [Link](http://adni.loni.usc.edu/data-samples/access-data/) |
| fMRI                   | BRATS                                  | 65                 | 4              | N/A             | N/A                                                           | [Link](https://www.med.upenn.edu/sbia/brats2018/data.html) |
| MEG                    | BCI-C IV, Data set III                 | 2                  | 4              | 400             | 10                                                           | [Link](http://www.bbci.de/competition/iv/) |


In order to let the readers have a quick access of the dataset and can play around it, we provide the well-processed and ready-ro-use [dataset](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/dataset/) of EEG Motor Movement/Imagery Database ([EEGMMIDB](https://archive.physionet.org/pn4/eegmmidb/)). This dataset contains 109 subjects while the EEG signals are recorded in 64 channels with 160 Hz sampling rate. After our clearning and sorting, each npy file represents a subject: the data shape of each npy file is [N, 65], the first 64 columns correspond to 64 channel features, the last column denotes the class label. The N varies for different subjects, but N should be either 259520 or 255680. This is the inherent difference in the original dataset.



## Running the code  

In our [tutorial](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/tutorial/) files, you will learn the pipline and workflow of BCI system including data acquiction, pre-processing, feature extraction (optional), classification, and evaluation. We present necessary references and actionable codes of the most typical deep learning models (GRU, LSTM, CNN, GNN) while taking advantage of temporal, spatial, and topographical depencencies. We also provide [python codes](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts) that are very handy. For example, to check the EEG classification performance of CNN, run the following code:
```
python 4-2_CNN.py 
```

For PyTorch beginners, we highly recommond [Morvan Zhou's PyTorch Tutorials](https://github.com/MorvanZhou/PyTorch-Tutorial).

## Chapter resources

For the algorithms and applications introduced in the book, we provide the necessary implementary codes (TensorFlow version):
- [Adaptive feature learning](https://github.com/xiangzhang1015/know_your_mind) (Chapter 7)
- [MindID: EEG-based user identification](https://github.com/xiangzhang1015/MindID) (Chapter 9)
- [Reconstruction of image based on visual evoked EEG](https://github.com/xiangzhang1015/EEG\_Shape\_Reconstruction) (Chapter 10)
- [Brain typing: convert EEG to text](https://github.com/xiangzhang1015/Brain_typing) (Chapter 11)
- [Neurological disorder (seizure) diagnosis](https://github.com/xiangzhang1015/adversarial\_seizure\_detection) (Chapter 13)

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

