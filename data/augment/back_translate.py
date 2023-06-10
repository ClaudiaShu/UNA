import os
import time
import pickle
from os.path import join

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import warnings
warnings.filterwarnings("ignore")


'''
All model names from Hugging Face use the following format Helsinki-NLP/opus-mt-{src}-{tgt}
where {src} and {tgt}correspond respectively to the source and target languages. 
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model size: 1316

# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-es'
# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name).to(device)

# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-es-en'
# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name).to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    first_model = nn.DataParallel(first_model)
    second_model = nn.DataParallel(second_model)

def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach


def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    tokenized_batch_texts = tokenizer(formated_batch_texts,
                                      return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate translation using model
    if torch.cuda.device_count() > 1:
        translated = model.module.generate(**tokenized_batch_texts)
    else:
        translated = model.generate(**tokenized_batch_texts)
    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    torch.cuda.empty_cache()

    return translated_texts


class wiki_Dataset(Dataset):
    def __init__(self, data_files = "../training/wiki1m_for_simcse.txt"):
        data_wiki = load_dataset('text', data_files=data_files)
        self.data = data_wiki['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['text']


def main():
    batch_size = 64
    dataset = wiki_Dataset()
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=False)

    outfile = "../training/translate_es.txt"

    first_model.eval()
    second_model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader)):
            # Check the model translation from the original language (English) to French
            translated_texts = perform_translation(data, first_model, first_model_tkn)
            # Perform the translation back to English
            back_translated_texts = perform_translation(translated_texts, second_model, second_model_tkn)

            with open(outfile, 'a') as file:
                for i, trans_lines in enumerate(back_translated_texts):
                    file.writelines(trans_lines)
                    file.write('\n')

if __name__ == '__main__':
    main()