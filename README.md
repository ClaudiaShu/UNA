# UNA

This is the official code of our Paper ["Unsupervised hard Negative Augmentation for contrastive learning"](https://arxiv.org/pdf/2401.02594.pdf)

## Environments
This repository is tested on Python 3.8+

## About
We present Unsupervised hard Negative Augmentation (UNA), a method that generates synthetic negative instances based on the term frequency-inverse document frequency (TF-IDF) retrieval model.

# Getting started

## Environments

```
conda create -n una python=3.8
conda activate una
cd UNA
pip install -r requirements.txt
```

## Dataset preparation

### Training set
We used the training dataset from SimCSE, which can be downloaded by running the following script.
```
cd data/
bash data/download_wiki.sh
```

### Prepare the paraphrased sentences
To create the paraphrased sentences, run the following script:
```
cd data/augment/
python paraphrase.py
```

### Produce TF-IDF matrix offline 
Run the following script to prepare the TD-IDF matrix. If you don't want to prepare the matrix offline, uncomment lines 94-101 in file `data/dataset.py`.
```
cd data/augment/
python create_dict.py
```
Change the mode to 'para' to produce the TF-IDF matrix for paraphrasing.

### Evaluation set
The evaluation set can be downloaded by running the following script:
```
cd data/downstream/
bash download_dataset.sh
```

### Code structure
After preparing the datasets, the structure of the code should look like this:


```
.
├── data  
│   ├── augment                      
│   ├── ├── paraphrase.py            # code for creating the paraphrased lines
│   ├── └── create_dict.py           # code for creating matrices under folder tfidf/
│   ├── downstream                   # folder containing evaluation dataset
│   ├── stsbenchmark                 # folder containing validation dataset
│   ├── training                     # folder containing training dataset
├── evaluate                         # Evaluation code *
├── function   
│   ├── metrics.py
│   ├── seed.py                      # initialize random seeds
│   └── tfidf_una.py                 # file for calculating the TF-IDF matrices and vectors for UNA
├── model 			
│   ├── lambda_scheduler.py          # contains different schedulers             
│   └── models.py                    # backbone BERT/RpBERTa model  
├── script                           # folder that contain .sh scripts to run the pre-training file
├── tfidf
│   ├── ori                          # folder for pre-saved TF-IDF representation of the original training dataset.
│   └── para                         # folder for pre-saved TF-IDF representation of the original training and the paraphrased dataset.
├── run.py                           # run pretraining with UNA
├── una.py                           # run pretraining with FaceSwap
└── utils.py
```

## Train UNA

To reproduce our results (for STS) with UNA framework, run the following training scipt [here](https://github.com/ClaudiaShu/UNA/tree/main/script).

## Trained Model

Models can be downloaded from [here](https://drive.google.com/drive/folders/1INk_txCPAtTHgsegP1b6cb97Xt_ITo5a).

### Results

<img src=misc/table.png>

### Acknowledgement

* Our code is inspired by: [UDA](https://github.com/google-research/uda).
* Evaluation code thanks to [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE).

