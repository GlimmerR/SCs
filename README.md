The preprocess code is used for reading dataset reported in the paper "Early-Prediction of Supercapacitors’ Cycle Life with Tunable Accuracy using Artificial Neural Networks".  

If you are interested in reproducing our results or using for other purposes. please

- 1、download the dataset from `https://dx.doi.org/10.6084/m9.figshare.11522082`.

- 2、extract the `raw.rar` file to the `/SCs/data/raw/`.

- 3、run the `main.py` file.

The raw.rar file contains 4 batches with a total of 113 supercapacitors. Each supercapacitor test is recorded in a excel file. The python files can read these excels and preprocess them. The first run will take hours to days, depending on the performance of the computer. The first run will save the data as binary files for future loading.
