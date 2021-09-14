# IronyTR

This repository contains the following:

* IronyTR Dataset: Extended Turkish Social Media Dataset for Irony Detection
* Model implementations used int IronyTR paper

## IronyTR Dataset
Extended Turkish Social Media Dataset for Irony Detection, extended over [Turkish Irony Dataset](https://github.com/teghub/Turkish-Irony-Dataset)

### Statistics
This dataset includes a total of 600 Turkish microblog texts, with two `.txt` files, each having 300 ironic and non-ironic sentences. Classification is made by a majority voting of 7. All data is retrieved from Turkish social media portals. 

### Files
**`ironic.txt`**
* Contains 300 lines, each line having only one attribute, the sentence itself.
* First 274 lines are in lexicographical order and do not have any Emoji’s.
* Remaining 7 lines have Emoji’s and they are not specifically ordered.

**`non-ironic.txt`**
* Contains 300 lines, each line having only one attribute, the sentence.
* First 293 lines are in lexicographical order and do not have any Emoji’s.
* Remaining 26 lines have Emoji’s and they are not specifically ordered.

**`lookup.txt`**
* Contains 2734 lines, each line having either an emoji, emoticon or word with its respective polarity score.
* First 10 lines have emoticons
* Next 2653 lines have Turkish words in lexicographical order.
* Remaining 71 lines have Emoji’s and they are not specifically ordered.
* Each line has a form of one of the following:
     * (Turkish word) (Polarity score of [-1,1] for the word)
          * `ışık 0.037 light`
     * (Turkish word) (Avg. polarity score of [-1,1] for the word) ((English translation #n) (Polarity of [-1,1] for the translation #n))+
          * `şekil 0.572 figure 0.807 shape 0.816 form 0.092`

## Implementations

### Traditional Methods
The lookups and helper codes for traditional methods are included. Explanation of the code structure is added as comments in the files for your convenience. You can refer to the paper for more information.

### LSTM
LSTM based methods (including CNN-LSTM and Bi-LSTM) are included with different settings and different feature combinations. You can refer to the paper for more information.

### BERT
First, you need to execute "data_prep.py" to convert your csv file into .tsv files. After that, you can execute "run_model.py" to evaluate your model. Statistics related to the evaluation results are saved in "outputs" directory. 

**Acknowledgement:** We benefited from [this implementation](https://github.com/ThilinaRajapakse/pytorch-transformers-classification) while implementing BERT model. We modified the implementation to add weight freezing and 10-fold cross-validation features. You can refer to the paper for more information.

# Citing: 

A.U.Ozturk, Y.Cemek, P.Karagoz, "IronyTR: Irony Detection in Turkish Informal Texts". (Under review)

Y.Cemek, C.Cidecio, A.U.Ozturk, R.F.Cekinel, P.Karagoz, "Türkçe Resmi Olmayan Metinlerde İroni Tespiti için Sinirsel Yöntemlerin İncelenmesi (Investigating the Neural Models for Irony Detection on Turkish Informal Texts)",  in IEEE 28th Signal Processing and Communications Applications Conference (SIU), Apr 2020.
