from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from nltk.tokenize import word_tokenize as wt

outfile = f"../training/wiki1m_for_simcse_test.txt"
batch_size = 64

class wiki_Dataset(Dataset):
    def __init__(self, data_files = "../training/wiki1m_for_simcse.txt"):
        data_wiki = load_dataset('text', data_files=data_files)
        self.data = data_wiki['train']

    def __len__(self):
        # return 100000
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['text']

def format_batch_texts(batch_texts):
    formated_bach = ["{}".format(text) for text in batch_texts]
    return formated_bach

dataset = wiki_Dataset()
data_loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False)

for idx, data in enumerate(tqdm(data_loader)):
    sentence = format_batch_texts(data)
    for i, trans_lines in enumerate(sentence):
        if len(wt(trans_lines)) == 0:
            print(trans_lines)
    # with open(outfile, 'a') as file:
    #     for i, trans_lines in enumerate(sentence):
    #         file.writelines(trans_lines)
    #         file.write('\n')