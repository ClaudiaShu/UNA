# UNA


## Environments

## Citing

## About

## Highlights

# Getting started

## Environments

```
conda create -n una python=3.8
conda activate una
git clone https://github.com/ClaudiaShu/UNA.git
cd UNA
pip install -r requirements.txt
```

## Dataset preparation

### Training set

```
cd data/
bash data/download_wiki.sh
```

### Prepare the paraphrased sentences
```
cd data/augment/
python paraphrase.py
```

### Produce TF-IDF matrix offline 
```
cd data/augment/
python create_dict.py
```
Change the mode to 'para' to produce the TF-IDF matrix for paraphrasing.

### Evaluation set
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
* Evaluation code thanks to [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE)

## Train UNA




