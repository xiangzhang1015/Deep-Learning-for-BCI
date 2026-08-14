<div align="center">

# Deep Learning for Brain-Computer Interface (BCI)

[![Book](https://img.shields.io/badge/Book-World%20Scientific-blue.svg)](https://www.worldscientific.com/worldscibooks/10.1142/q0282)
[![Survey](https://img.shields.io/badge/Survey-JNE%202020-green.svg)](https://iopscience.iop.org/article/10.1088/1741-2552/abc902)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

**[Dr. Xiang Zhang](http://xiangzhang.info/) · [Prof. Lina Yao](https://linayao.com/)**

*Companion repository for the book: Deep Learning for EEG-based Brain-Computer Interface: Representations, Algorithms and Applications (World Scientific, 2021)*

</div>

---

## Overview

This repository contains implementable Python and Jupyter Notebook code alongside benchmark datasets for learning how to classify brain signals using deep learning. It accompanies our [survey on deep learning for noninvasive brain signals](https://iopscience.iop.org/article/10.1088/1741-2552/abc902) and the [textbook](https://www.worldscientific.com/worldscibooks/10.1142/q0282).

The materials cover:

- A taxonomy of BCI signal paradigms by acquisition method (ECoG, EEG, fNIRS, fMRI, EOG, MEG)
- Core deep learning architectures: MLP, RNN, LSTM, GRU, CNN, AE, VAE, GAN, DBN, GIN
- Guidelines for designing BCI systems across signal categories, models, and applications
- Novel BCI prototypes for authentication, visual reconstruction, language interpretation, and neurological disorder diagnosis

<p align="center">
<img src="https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/BCI_signals.png" width="900">
</p>

<p align="center">
<img src="https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/dl_models.png" width="900">
</p>

Distribution on signals | Distribution on DL models
:---:|:---:
![signals](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/bar_signal.png) | ![models](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/images/bar_model.png)

---

## Dataset

Brain signal collection is both financially and temporally costly. We curated 31 public datasets with download links covering most brain signal types:

| Brain Signal | Dataset | Subjects | Classes | Rate (Hz) | Channels | Download |
|---|---|:---:|:---:|:---:|:---:|:---:|
| FM ECoG | BCI-C IV, Data set IV | 3 | 5 | 1000 | 48–64 | [Link](http://www.bbci.de/competition/iv/) |
| MI ECoG | BCI-C III, Data set I | 1 | 2 | 1000 | 64 | [Link](http://www.bbci.de/competition/iii/) |
| Sleeping EEG | Sleep-EDF Telemetry | 22 | 6 | 100 | 2 EEG, 1 EOG, 1 EMG | [Link](https://physionet.org/physiobank/database/sleep-edfx/) |
| Sleeping EEG | Sleep-EDF Cassette | 78 | 6 | 100/1 | 2 EEG, 1 EOG, 1 EMG | [Link](https://physionet.org/physiobank/database/sleep-edfx/) |
| Sleeping EEG | MASS-1 | 53 | 5 | 256 | 17/19 EEG, 2 EOG, 5 EMG | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG | MASS-3 | 62 | 5 | 256 | 20 EEG, 2 EOG, 3 EMG | [Link](https://massdb.herokuapp.com/en/) |
| Sleeping EEG | SHHS | 5804 | N/A | 125/50 | 2 EEG, 1 EOG, 1 EMG | [Link](https://physionet.org/pn3/shhpsgdb/) |
| Seizure EEG | CHB-MIT | 22 | 2 | 256 | 18 | [Link](https://physionet.org/pn6/chbmit/) |
| Seizure EEG | TUH | 315 | 2 | 200 | 19 | [Link](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) |
| MI EEG | EEGMMIDB | 109 | 4 | 160 | 64 | [Link](https://physionet.org/pn4/eegmmidb/) |
| MI EEG | BCI-C IV, Data set II a | 9 | 4 | 250 | 22 EEG, 3 EOG | [Link](http://www.bbci.de/competition/iv/) |
| Emotional EEG | AMIGOS | 40 | 4 | 128 | 14 | [Link](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html) |
| Emotional EEG | SEED | 15 | 3 | 200 | 62 | [Link](http://bcmi.sjtu.edu.cn/~seed/download.html) |
| Emotional EEG | DEAP | 32 | 4 | 512 | 32 | [Link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| fMRI | ADNI | 202 | 3 | N/A | N/A | [Link](http://adni.loni.usc.edu/data-samples/access-data/) |
| fMRI | BRATS | 65 | 4 | N/A | N/A | [Link](https://www.med.upenn.edu/sbia/brats2018/data.html) |

We also provide a well-processed, ready-to-use version of the [EEGMMIDB](https://archive.physionet.org/pn4/eegmmidb/) dataset (109 subjects, 64 channels, 160 Hz). Each `.npy` file represents one subject with shape `[N, 65]`: 64 EEG channel features plus one class label column.

---

## Running the Code

The [tutorial](https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/tutorial/) notebooks cover the full BCI pipeline: data acquisition, preprocessing, feature extraction, classification, and evaluation. To run the CNN classification example:

```bash
python 4-2_CNN.py
```

For PyTorch beginners, we recommend [Morvan Zhou's PyTorch Tutorials](https://github.com/MorvanZhou/PyTorch-Tutorial).

---

## Chapter Resources

Companion code for the book chapters (TensorFlow implementations):

| Chapter | Topic | Repository |
|:---:|---|---|
| 7 | Adaptive feature learning | [know_your_mind](https://github.com/xiangzhang1015/know_your_mind) |
| 9 | EEG-based user identification (MindID) | [MindID](https://github.com/xiangzhang1015/MindID) |
| 10 | Visual EEG shape reconstruction | [EEG_Shape_Reconstruction](https://github.com/xiangzhang1015/EEG_Shape_Reconstruction) |
| 11 | Brain typing: EEG to text | [Brain_typing](https://github.com/xiangzhang1015/Brain_typing) |
| 13 | Neurological disorder (seizure) diagnosis | [adversarial_seizure_detection](https://github.com/xiangzhang1015/adversarial_seizure_detection) |

---

## Requirements

Python 3.7. Install dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and its dependencies (cluster, scatter, sparse) require version-specific installation — follow the official guide.

---

## Citation

If this repository is useful for your research, please cite the survey or book:

```bibtex
@article{zhang2020survey,
  title   = {A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers},
  author  = {Zhang, Xiang and Yao, Lina and Wang, Xianzhi and Monaghan, Jessica JM and Mcalpine, David and Zhang, Yu},
  journal = {Journal of Neural Engineering},
  year    = {2020},
  publisher = {IOP Publishing}
}

@book{zhang2021deep,
  title     = {Deep Learning for EEG-based Brain-Computer Interface: Representations, Algorithms and Applications},
  author    = {Zhang, Xiang and Yao, Lina},
  year      = {2021},
  publisher = {World Scientific Publishing}
}
```

---

## Contact

For questions about the code or tutorial, contact [xiang.alan.zhang@gmail.com](mailto:xiang.alan.zhang@gmail.com).

## License

This repository is licensed under [CC BY-NC-SA 4.0](LICENSE). Commercial use is prohibited. Derivatives must carry the same license.
