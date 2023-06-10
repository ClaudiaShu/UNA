import os
import time
import pickle
from os.path import join

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import *
import warnings
warnings.filterwarnings("ignore")


'''
All model names from Hugging Face use the following format Helsinki-NLP/opus-mt-{src}-{tgt}
where {src} and {tgt}correspond respectively to the source and target languages. 
'''
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def get_model(model_name='pegasus'):
    if model_name == 'pegasus':
        # Pegasus model for paraphrase
        model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
        tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
        return model, tokenizer
    elif model_name == 't5':
        # T5 model for paraphrase
        tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        return model, tokenizer
    else:
        raise ValueError

def format_batch_texts(batch_texts):
    formated_bach = ["{}".format(text) for text in batch_texts]
    return formated_bach

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
    # tokenize the text to be form of a list of token IDs
    inputs = tokenizer(sentence, truncation=True, padding="longest", return_tensors="pt").to(device)
    # generate the paraphrased sentences
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        outputs = model.module.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
    else:
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
    torch.cuda.empty_cache()
    # decode the generated sentences using the tokenizer to get them back to text
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


class wiki_Dataset(Dataset):
    def __init__(self, data_files = "data/training/wiki1m_for_simcse.txt"):
        data_wiki = load_dataset('text', data_files=data_files)
        self.data = data_wiki['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['text']


def main():
    batch_size = 4
    dataset = wiki_Dataset()
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            shuffle=False,
                            pin_memory=False)

    model_name = 'pegasus'

    outfile = f"data/training/paraphrase.txt"

    model, tokenizer = get_model(model_name=model_name)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader)):
            sentence = format_batch_texts(data)
            paraphrased_texts = get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=1)
            with open(outfile, 'a') as file:
                for i, trans_lines in enumerate(paraphrased_texts):
                    file.writelines(trans_lines)
                    file.write('\n')

if __name__ == '__main__':
    main()