# DL-hybrid-precoder
This is the source code for paper ["Deep Learning for Direct Hybrid Precoding in Millimeter Wave Massive MIMO Systems"](https://arxiv.org/abs/1905.13212)

In the paper, we proposes a novel neural network architecture, that we call an auto-precoder, and a deep-learning based approach that jointly senses the millimeter wave (mmWave) channel and designs the hybrid precoding matrices with only a few training pilots. More specifically, the proposed model leverages the prior observations of the channel to achieve two objectives. First, it optimizes the compressive channel sensing vectors based on the surrounding environment in an unsupervised manner to focus the sensing power on the most promising spatial directions. This is enabled by a novel neural network architecture that accounts for the constraints on the RF chains and models the transmitter/receiver measurement matrices as two complex-valued convolutional layers. Second, the proposed model learns how to construct the RF beamforming vectors of the hybrid architectures directly from the projected channel vector (the received sensing vector). The auto-precoder neural network that incorporates both the channel sensing and beam prediction is trained end-to-end as a multi-task classification problem. Each task is a multi-label classification problem. The network is shown in the following figure.

![Figure1](https://github.com/lxf8519/DL-hybrid-precoder/blob/master/NN_hybrid.jpg)

To find more information about the paper and other deep-learining based wireless communication work, please visit [DeepMIMO dataset applications](http://deepmimo.net/DeepMIMO_applications.html?i=1).

# Requirement
Tensorflow 1.12.0, Keras 2.2.4 (TF 2.0 does not work)

# Run training and tesing
1. Quick run: Run in terminal "python main_train_beamforming.py -train 1" to train the model and run "python main_train_beamforming.py -train 0" for testing. The default parameters are: dataset='DeepMIMO_dataset_train20.mat' and 'DeepMIMO_dataset_test20.mat' (which are corresponding to total transmit power of 20dB), epochs=15, batch_size=512, learning_rate=0.002.

2. If you need to change the dataset and parameters, they can be found in "main_train_beamforming.py".

3. The prediction accuracy results for the transmitter and receiver on the default dataset are given in the following table. The total transmit power for this dataset is 20 dBm.

| Transmit power (dBm)| 20 |
| -------- | ------ |
| Tx acc.(Mt=Mr=2) | 0.71 |
| Rx acc.(Mt=Mr=2) | 0.69 |
| Tx acc.(Mt=Mr=4) | 0.78 |
| Rx acc.(Mt=Mr=4) | 0.76 |
| Tx acc.(Mt=Mr=8) | 0.88 |
| Rx acc.(Mt=Mr=8) | 0.88 |

**To reproduce the results, the pre-trained model in Saved_model folder needs to be loaded for testing. Also, the datasets of DeepMIMO_dataset_train20.mat and DeepMIMO_dataset_test20.mat (and the corresponding label mat files) are required (See the following part for dataset)**.

# Dataset 
**The dataset can be downloaded [here](https://drive.google.com/open?id=1sMiDGhPYpblkkcQgvq4F5q7w2AINfkgL) from Google drive.** There are 4 files for training and testing and the labels. **After downlaoding, copy these 4 files to MIMO_dataset folder.** To generate your own dataset, visit [DeepMIMO.net](http://deepmimo.net/index.html).

# Citation
If you find the code is useful, please kindly cite our paper. Thanks.
```
@article{li2019deep,
  title={Deep Learning for Direct Hybrid Precoding in Millimeter Wave Massive MIMO Systems},
  author={Li, Xiaofeng and Alkhateeb, Ahmed},
  journal={arXiv preprint arXiv:1905.13212},
  month={May},
  year={2019}
}
```
# License
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
