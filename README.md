# EEG
A project using deep learning methods (Deep Belief Networks and etc.) to classify sleep stages with EEG signals.

##########################
1.final report

"Project report of Shaowei Zhang.pdf" is the final project report with 54 pages.

2.project presentation

"Project Presentation of Shaowei Zhang.pptx" is the PPT used during the presentation on Nov. 9.

3.codes

The following steps show that how to run the codes in the "code" fold.

1) To run the codes, you need firstly install a few libraries.
   a. [pyedflib] library from <https://github.com/holgern/pyedflib>
      This library is used to process edf files.

   b. [deep-belief-network] library from <https://github.com/albertbup/deep-belief-network>
      This library is based on tensorflow and is used to perform DBN classification.

   c. [hmmlearn] library from <https://github.com/hmmlearn/hmmlearn>
      This library is used to perform HMM.


2) Original dataset should be downloaded from <https://physionet.org/physiobank/database/sleep-edfx/> and be put into the same fold.
   Files start with "SC" will be used. Also, files end with "-PSG.edf" and "-Hypnogram.edf" will both be used.
   For instance, files named "SC4001E0-PSG.edf" and "SC4001EC-Hypnogram.edf" should be downloaded into the "code" fold.
   If you want, you could just click the files and visualize the waves by downloading the "Polyman" from <https://physionet.org/physiobank/database/sleep-edfx/Polyman/>.

3) The file named "dataprocessingfinal.py" is used to process edf files like "SC4001E0-PSG.edf" and "SC4001EC-Hypnogram.edf".
   The result will be saved like "train1.csv" and "label_train1.csv".
   By controlling the numbers for each sleep stage in line 98, dataset with different sample ratio could be created for further use.
   This file should be run first to generate files like "train1.csv" and "label_train1.csv" for further use.

4) The file named "DBNfinal.py" is used to perform the complete DBN classifier.
   The results includes Logistic Regression prediction, DBN prediction and DBN_HMM prediction.
   If files like "train1.csv" and "label_train1.csv" have already been generated into the "code" fold,then "DBNfinal.py" could be run directly.

5) The file named "classifiercomparison.py" is used to perform 4 classifiers to compare with the result of DBN.
   The classifiers are GNB, LR, SVM and RBM-Logitic.
   Likewise, it could be run directly if files like "train1.csv" and "label_train1.csv" have already been generated into the "code" fold.


4.systems

This project is more like machine learning analyzing than visualizing. Besides, the dataset is time series rather than images or videos.
As a result, there is no system in this project.

5.source codes

There's only part of the codes in this project that have the source codes. 
It's the part of how to use SupervisedDBNClassification and UnsupervisedDBN functions in dbn library.
The source codes can be found on the github website of [deep-belief-network] library, which is <https://github.com/albertbup/deep-belief-network> .

6.demo videos

This project is more like machine learning analysis.
The dataset is time series and there's no image classification part in this project.
The procedure of how to run the codes is described step by step in part 3.
So there is no demo video in this zip file.
