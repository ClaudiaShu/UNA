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


## Train UNA

## Code structure

