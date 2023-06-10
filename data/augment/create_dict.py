import collections
import os
import csv
import copy
import math
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from nltk.tokenize import word_tokenize as wt

mode = "ori" # para

output_file = f"tfidf/{mode}"
os.makedirs(output_file, exist_ok=True)

data = "./data/training/wiki1m_for_simcse.txt"
# data = "./data/training/wiki_tiny.txt"

data_files = {}
data_files["train"] = data
extension = data.split(".")[-1]
if extension == "txt":
    extension = "text"
    datasets = load_dataset(extension, data_files=data_files)
elif extension == "csv":
    datasets = load_dataset(extension, data_files=data_files,
                            delimiter="\t" if "tsv" in data else ",")
    
if "para" in mode:
    trans_data = "./data/training/paraphrase.txt"
    trans_data_files = {}
    trans_data_files["train"] = trans_data
    extension = trans_data.split(".")[-1]
    if extension == "txt":
        extension = "text"
        trans_datasets = load_dataset(extension, data_files=trans_data_files)
    elif extension == "csv":
        trans_datasets = load_dataset(extension, data_files=trans_data_files,
                                delimiter="\t" if "tsv" in trans_data else ",")
    
def get_sentences(data, trans_data=None):
    logger.info('Start getting sentences from the dataset')
    sentences = []
    for i in tqdm(range(len(data['train']))):
        sentence = data['train'][i]['text']
        sentences.append(sentence)

    if "para" in mode:
        for i in tqdm(range(len(trans_data['train']))):
            sentence = trans_data['train'][i]['text']
            sentences.append(sentence)
    return sentences

def do_parse(sentence):
    return wt(sentence.lower())

def save_csv(data, filename):
    output_file = filename

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in data:
            writer.writerow(line)
        # writer.writerow(["Word"])  # Write the header
        # writer.writerows([[key] for key in sorted_idf])  # Write keys row by row

if "para" in mode:
    sentences = get_sentences(datasets, trans_datasets)
else:
    sentences = get_sentences(datasets)

word_doc_freq = collections.defaultdict(int)
logger.info('Creating the word dict')
for i in tqdm(range(len(sentences))):
    # Iterate through the whole corpus
    cur_word_dict = {}
    # wt splits the sentence into vocab
    cur_sent = copy.deepcopy(do_parse(sentences[i]))
    for word in cur_sent:
        cur_word_dict[word] = 1
    for word in cur_word_dict:
        word_doc_freq[word] += 1
        
# Compute IDF
idf = {}
for word in tqdm(word_doc_freq):
    idf[word] = math.log(len(sentences) * 1. / word_doc_freq[word])
# Save this IDF matrix offline, 1 for original, 1 for paraphrasing

# idf_file = f"tfidf/{mode}/term_idf.csv"

# with open(idf_file, 'w', newline='') as file:
#     fieldnames = ['Keys', 'Scores_idf']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for key, score in idf.items():
#         writer.writerow({'Keys':key, 'Scores_idf':score})

# Compute global replacement score
count = {}
tf_idf_sum = {}
tf_idf_max = {}
tf_idf_norm = {}
for i in tqdm(range(len(sentences))):
    cur_sent = copy.deepcopy(do_parse(sentences[i]))
    word_counts = collections.Counter(cur_sent)
    for word in word_counts:
        if word not in count:
            count[word] = 0
            tf_idf_sum[word] = 0
            tf_idf_max[word] = 0
            tf_idf_norm[word] = 0

        count[word] += 1
        # count[word] += word_counts[word]

        # word_freq = word_counts[word] / len(cur_sent)
        word_freq = math.log(1 + word_counts[word] / len(cur_sent))
        cache_score = word_freq * idf[word]
        norm_cache_score = cache_score/len(cur_sent)
        tf_idf_max[word] = cache_score if cache_score > tf_idf_max[word] else tf_idf_max[word]
        tf_idf_norm[word] = norm_cache_score if norm_cache_score > tf_idf_norm[word] else tf_idf_norm[word]
        tf_idf_sum[word] += cache_score

tf_idf_mean = {word: tf_idf_sum[word]/count[word] for word in idf.keys()}

# Save this count, tf_idf_max matrix offline

# count_file = f"tfidf/{mode}/term_count.csv"

# with open(count_file, 'w', newline='') as file:
#     fieldnames = ['Keys', 'Scores_count']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for key, score in count.items():
#         writer.writerow({'Keys':key, 'Scores_count':score})

# tfidfmax_file = f"tfidf/{mode}/term_tfidfmax.csv"

# with open(tfidfmax_file, 'w', newline='') as file:
#     fieldnames = ['Keys', 'Scores_max']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for key, score in tf_idf_max.items():
#         writer.writerow({'Keys':key, 'Scores_max':score})

tfidfmax_file = f"tfidf/{mode}/term_tfidfnorm.csv"

with open(tfidfmax_file, 'w', newline='') as file:
    fieldnames = ['Keys', 'Scores_norm']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for key, score in tf_idf_norm.items():
        writer.writerow({'Keys':key, 'Scores_norm':score})

# tfidfmax_file = f"tfidf/{mode}/term_tfidfsum.csv"

# with open(tfidfmax_file, 'w', newline='') as file:
#     fieldnames = ['Keys', 'Scores_sum']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for key, score in tf_idf_sum.items():
#         writer.writerow({'Keys':key, 'Scores_sum':score})

tfidfmax_file = f"tfidf/{mode}/term_tfidfmean.csv"

with open(tfidfmax_file, 'w', newline='') as file:
    fieldnames = ['Keys', 'Scores_mean']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for key, score in tf_idf_mean.items():
        writer.writerow({'Keys':key, 'Scores_mean':score})

filter_dict = count
keys_to_remove = []
for key, value in tqdm(filter_dict.items()):
    if value <= 4 or math.log(value) > 8:
        keys_to_remove.append(key)
# Save keys to remove offline

# filter_file = f"tfidf/{mode}/term_filter.csv"

# with open(filter_file, 'w', newline='') as file:
#     fieldnames = ['Keys']
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     for key in keys_to_remove:
#         writer.writerow({'Keys':key})

