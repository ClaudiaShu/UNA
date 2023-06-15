import collections
import csv
import copy
import json
import math
import string
import random
import itertools
from tqdm import tqdm
from absl import flags
import numpy as np
from loguru import logger
from typer import Option
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from sklearn.feature_extraction.text import TfidfVectorizer

FLAGS = flags.FLAGS

printable = set(string.printable)

nltk_stopwords = stopwords.words("english")

tf_mode = 'logfreq'

def do_parse(sentence):
    return wt(sentence.lower())

def build_vocab(examples):
    vocab = {}
    def add_to_vocab(sentence):
        for word in do_parse(sentence):
            if word not in vocab:
                vocab[word] = len(vocab)
    for i in range(len(examples)):
        add_to_vocab(examples[i])
    return vocab

def filter_unicode(st):
    return "".join([c for c in st if c in printable])

def read_csv_to_dict(file_path, cols):
    if cols == 1:
        out = []
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            for row in reader:
                key = row[headers[0]]
                out.append(key)
    elif cols == 2:
        out = {}
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            for row in reader:
                key = row[headers[0]]
                out[key] = float(row[headers[1]])
    else:
        raise ValueError("Invalid input datafile")
    
    return out

def read_data_stats(files):
    file_idf = files['file_idf']
    file_count = files['file_count']
    file_tfidfmax = files['file_tfidfmax']
    file_filter = files['file_filter']

    idf = read_csv_to_dict(file_idf, cols=2)
    count = read_csv_to_dict(file_count, cols=2)
    tfidfmax = read_csv_to_dict(file_tfidfmax, cols=2)
    # keys_to_remove = read_csv_to_dict(file_filter, cols=1)

    keys_to_remove = []

    score_dict = tfidfmax
    for key in keys_to_remove:
        score_dict.pop(key)
    out_dict = score_dict
    
    return {
        "idf": idf,
        "tf_idf": out_dict,
        "keys_rm": keys_to_remove
    }

def get_data_stats(examples):
    word_doc_freq = collections.defaultdict(int)
    logger.info('Creating the word dict')
    
    for i in tqdm(range(len(examples))):
        # Iterate through the whole corpus
        cur_word_dict = {}
        cur_sent = copy.deepcopy(do_parse(examples[i]))
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
            
    # Compute IDF
    idf = {}
    for word in tqdm(word_doc_freq):
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
        
    # Compute global replacement score
    count = {}
    tf_idf_max = {}
    for i in tqdm(range(len(examples))):
        cur_sent = copy.deepcopy(do_parse(examples[i]))
        word_counts = collections.Counter(cur_sent)
        for word in word_counts:
            if word not in count:
                count[word] = 0
                tf_idf_max[word] = 0

            count[word] += 1
    
            word_freq = math.log(1 + word_counts[word] / len(cur_sent))
            cache_score = word_freq * idf[word]
            tf_idf_max[word] = cache_score if cache_score > tf_idf_max[word] else tf_idf_max[word]

    keys_to_remove = []

    # Remove the out-out-range keys in tfidf
    score_dict = tf_idf_max
    for key in keys_to_remove:
        score_dict.pop(key)
    out_dict = score_dict
    
    return {
        "idf": idf,
        "tf_idf": out_dict,
        "keys_rm": keys_to_remove
    }

def read_data_stats_rand(files):
    file_idf = files['file_idf']
    file_count = files['file_count']
    file_tfidfmax = files['file_tfidfmax']
    file_filter = files['file_filter']

    idf = read_csv_to_dict(file_idf, cols=2)
    count = read_csv_to_dict(file_count, cols=2)
    tfidfmax = read_csv_to_dict(file_tfidfmax, cols=2)
    keys_to_remove = read_csv_to_dict(file_filter, cols=1)

    rand_dict = {key: 1 for key in tfidfmax}

    keys_to_remove = []


    score_dict = rand_dict
    for key in keys_to_remove:
        score_dict.pop(key)
    out_dict = score_dict
    
    return {
        "idf": idf,
        "tf_idf": out_dict,
        "keys_rm": keys_to_remove
    }

class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token
    
    def get_threshold_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token

class UnifRep(EfficientRandomGen):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, token_prob, data_stats, uda):
        self.token_prob = token_prob
        self.vocab_size = len(data_stats)
        self.vocab = data_stats["idf"]
        self.uda = uda
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, example):
        out = self.replace_tokens(do_parse(example))
        return ' '.join(word for word in out)

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = list(self.vocab.keys())
        self.token_ptr = len(self.token_list) - 1
        random.shuffle(self.token_list)

class ThresholdRep(EfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob, data_stats, uda, nbh_size):
        super(ThresholdRep, self).__init__()
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.uda = uda
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        self.keys_rm = set(data_stats["keys_rm"])
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.nbh_dict = {}
        self.nbh_count = {}
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf
                                  - self.normalized_tf_idf.min())
        self.normalized_tf_idf = (self.normalized_tf_idf
                                  / self.normalized_tf_idf.sum())
        self.nbh_size = nbh_size
        self.map_neighbour_prob_list()
        self.reset_token_list()
        self.reset_random_prob()
        self.replacement_count = 0
        self.wordlist_len = 0

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        # This is the added function
        word_counts = collections.Counter(all_words)
        for word in word_counts:
            word_freq = math.log(1 + word_counts[word] / len(all_words))
            if word in self.keys_rm:
                cur_tf_idf[word] = 0
            else:
                cur_tf_idf[word] += word_freq * self.idf[word]

        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        if self.uda:
            replace_prob = np.max(replace_prob) - replace_prob
        else:
            replace_prob = replace_prob - np.min(replace_prob)
        replace_prob = (replace_prob / replace_prob.sum() * self.token_prob * len(all_words)) 

        # Make sure that at least one of the word get replaced
        if replace_prob.max() < 1:
            replace_prob[replace_prob.argmax()] = 1

        return replace_prob

    def __call__(self, example):

        all_words = copy.deepcopy(do_parse(example))

        replace_prob = self.get_replace_prob(all_words)
        out = self.replace_tokens(
            do_parse(example),
            replace_prob[:len(all_words)]
            )
        output = ' '.join(word for word in out)
        if example == output:
            output = self.get_random_token()
        return output

    def map_neighbour_prob_list(self):
        """Map the tokens with terms in k-nearest neighbours with probability"""
        cache_len = len(self.tf_idf_keys)
        logger.info("Sequence mapping ...")
        for idx, word in tqdm(enumerate(self.tf_idf_keys)):
            if word not in self.nbh_dict:
                start = max(0, idx - self.nbh_size)
                end = min(cache_len, idx + self.nbh_size + 1)
                nbh_prob = self.tf_idf_values[start:idx]+self.tf_idf_values[idx:end]
                sum_nbh = sum(nbh_prob)
                nbh_prob = [x / sum_nbh for x in nbh_prob]
                nbh = np.random.choice(len(nbh_prob), (len(nbh_prob),), p=nbh_prob)
                
                self.nbh_dict[word] = nbh + start
                self.nbh_count[word] = 0

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                nbh = self.nbh_dict[word_list[i]]
                word_list[i] = self.get_random_token_from_nbh(nbh, word_list[i])
                self.replacement_count += 1
            self.wordlist_len += 1
        return word_list
    
    def get_random_token_from_nbh(self, nbh, word):
        token = self.tf_idf_keys[nbh[self.nbh_count[word]]]
        self.nbh_count[word] += 1
        if self.nbh_count[word] > len(nbh)-1:
            self.nbh_count[word] = 0
        return token
        # return np.random.choice(nbh)
    
    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
    
    def get_random():
        return random.random()
    
