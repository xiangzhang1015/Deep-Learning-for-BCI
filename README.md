# Deep Learning-based Brain-Computer Interface (BCI) 

#### Author: [Xiang Zhang](http://xiangzhang.info/) (xiang_zhang@hms.harvard.edu), [Lina Yao](https://linayao.com/) (lina.yao@unsw.edu.au) 

#### Contributor: [Ziyu Liu](https://github.com/ziyuliu-lion)


## Overview

This tutorial is updating continuously.

This repository contains implementable python and jupyter notebook codes and benchmark datasets to learn how to recognize brain signals based on deep learning models. This tutorial associates our [survey on DL-based noninvasive brain signals](https://iopscience.iop.org/article/10.1088/1741-2552/abc902) and [book on DL-based BCI: Representations, Algorithms and Applications](). 


- We present a new taxonomy of BCI signal paradigms according to the acquisition methods. [Add Full name for abbrevations]

<p align="center">
<img src="https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/BCI_signals.png" width="900" align="center">
</p>

- We systemically introduce the fundamental knowledge of deep learning models. 

<p align="center">
<img src="https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/images/dl_models.png" width="900" align="center">
</p>

- Moreover, the guidelines for the investigation and design of BCI systems are provided from the aspects of signal category, deep learning models, and applications. The following figures show the distribution on signals and DL models in frointer DL-based BCI studies.

Solarized dark             |  Solarized Ocean
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


In order to let the readers have a quick access of the dataset and can play around it, we provide the well-processed and ready-ro-use [dataset](https://github.com/xiangzhang1015/ML_BCI_tutorial/blob/main/dataset/) of EEG Motor Movement/Imagery Database ([EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/)). 
Add introduce and how to find the data: like 109 people, more details are in jupyternotebook x , add link




### Running the code  

In our [tutorial](https://github.com/xiangzhang1015/ML_BCI_tutorial/tree/main/tutorial/) files, you will learn the pipline and workflow of BCI system including data acquiction, pre-processing, feature extraction (optional), classification, and evaluation. We present necessary references and actionable codes of the most typical deep learning models (GRU, LSTM, CNN, GNN) while taking advantage of temporal, spatial, and typographical depencencies. We also provide [python codes](https://github.com/xiangzhang1015/ML_BCI_tutorial/tree/main/pythonscripts) that are very handy. For example, to check the EEG classification performance of CNN, run the following code:
```
python 4-2_CNN.py 
```



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

