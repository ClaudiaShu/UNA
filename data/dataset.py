import copy
import re
from loguru import logger
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from function.tfidf_una import *

class SimcseEvalDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.mode = kwargs['eval_mode']
        self.tokenizer = kwargs['tokenizer']
        self.features = self.load_eval_data()

    def load_eval_data(self):
        """
        add dec or test set
        """
        assert self.mode in ['eval', 'test'], 'mode should in ["eval", "test"]'
        if self.mode == 'eval':
            eval_file = self.args.dev_file
        else:
            eval_file = self.args.test_file
        feature_list = []
        with open(eval_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split("\t")
                assert len(line) == 7 or len(line) == 9
                score = float(line[4])
                data1 = self.tokenizer(line[5].strip(), max_length=self.args.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')
                data2 = self.tokenizer(line[6].strip(), max_length=self.args.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')

                feature_list.append((data1, data2, score))
        return feature_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

class TrainDataset(Dataset):
    '''
    Training dataset
    '''
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.tokenizer = kwargs['tokenizer']
        data = kwargs['data']

        # Dataset
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        data_files = {}
        if data is not None:
            data_files["train"] = data
        extension = data.split(".")[-1]
        if extension == "txt":
            extension = "text"
            datasets = load_dataset(extension, data_files=data_files)
        elif extension == "csv":
            datasets = load_dataset(extension, data_files=data_files,
                                    delimiter="\t" if "tsv" in data else ",")
        else:
            raise ValueError("Error with the type of file.")

        self.data = datasets
        self.column_names = self.data["train"].column_names

        # load translate data
        if self.args.do_para:
            trans_data = self.args.para_train_data
            trans_data_files = {}
            if trans_data is not None:
                trans_data_files["train"] = trans_data
            extension = trans_data.split(".")[-1]
            if extension == "txt":
                extension = "text"
                trans_datasets = load_dataset(extension, data_files=trans_data_files)
            elif extension == "csv":
                trans_datasets = load_dataset(extension, data_files=trans_data_files,
                                            delimiter="\t" if "tsv" in trans_data else ",")
            else:
                raise ValueError("Error with the type of file.")

            self.trans_data = trans_datasets
            self.trans_column_names = self.trans_data["train"].column_names

        if self.args.do_tf_idf:
            # # Build the data statistics for UNA
            # IDF for process 1 calculation
            # tf-idf for process 2 word selection
            # keys_rm for removing the words not to change in process 1
            
            # # Uncomment this to process the tfidf states online
            # sentences = self.get_sentences()
            # self.data_stats = self.get_tfidf(sentences)
            
            # # This loads the pre-saved tfidf stats file
            self.data_stats = self.get_tfidf_byfile()
            
            if self.args.do_hard_negatives:
                do_uda = False
            else:
                do_uda = True
            # mode 1: p1 probability replace; mode2: p1 random replace; mode3: p2 threshold selection
            if self.args.tfidf_mode == "uni":
                self.op = UnifRep(token_prob=self.args.token_prob, data_stats=self.data_stats, uda=do_uda)
            elif self.args.tfidf_mode == "tfidf":
                self.op = TfIdfWordRep(token_prob=self.args.token_prob, data_stats=self.data_stats, uda=do_uda, p_rand=self.args.p2_rand)
            elif self.args.tfidf_mode == "nbh":
                self.op = ThresholdRep(token_prob=self.args.token_prob, data_stats=self.data_stats, uda=do_uda, nbh_size=self.args.nbh_size)
            elif self.args.tfidf_mode == "group":
                self.op = GroupRep(token_prob=self.args.token_prob, data_stats=self.data_stats, uda=do_uda, nbh_size=self.args.gp_num)

    def __len__(self):
        # return 10000
        return len(self.data['train'])

    def __getitem__(self, index: int):

        anchor = self.data['train'][index]['text']

        if self.args.do_para:
            positive = self.trans_data['train'][index]['text']
        else:
            positive = copy.deepcopy(anchor)

        if self.args.do_tf_idf:
            if self.args.do_hard_negatives:
                anchor_hn = self.op(anchor)
                positive_hn = copy.deepcopy(anchor_hn)
                return {
                    "anchor": anchor,
                    "positive": positive,
                    "anchor_hn": anchor_hn,
                    "positive_hn": positive_hn
                }
            else:
                positive = self.op(positive)

        return {
            "anchor": anchor,
            "positive": positive
        }

    def get_sentences(self):
        logger.info('Start getting sentences from the dataset')
        sentences = []
        for i in tqdm(range(len(self.data['train']))):
            sentence = self.data['train'][i]['text']
            sentences.append(sentence)
        
        if self.args.do_para:
            for i in tqdm(range(len(self.trans_data['train']))):
                sentence = self.trans_data['train'][i]['text']
                sentences.append(sentence)
        return sentences

    def get_tfidf(self, sentences):
        # Process 2
        data_stats = get_data_stats(examples=sentences)
        return data_stats
    
    def get_tfidf_byfile(self):
        # Process 2
        if self.args.do_para:
            folder = 'para'
            files = {
                'file_idf':f'./tfidf/{folder}/term_idf.csv',
                'file_count':f'./tfidf/{folder}/term_count.csv',
                'file_tfidfmax':f'./tfidf/{folder}/term_tfidfmax.csv',
                'file_filter':f'./tfidf/{folder}/term_filter.csv',
            }
        else:
            folder = 'ori'
            files = {
                'file_idf':f'./tfidf/{folder}/term_idf.csv',
                'file_count':f'./tfidf/{folder}/term_count.csv',
                'file_tfidfmax':f'./tfidf/{folder}/term_tfidfmax.csv',
                'file_filter':f'./tfidf/{folder}/term_filter.csv',
            }
        if self.args.p2_rand:
            data_stats = read_data_stats_rand(files=files)
        else:
            data_stats = read_data_stats(files=files)
        return data_stats
