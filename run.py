import argparse

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, utils

from data.dataset import SimcseEvalDataset, TrainDataset
from function.seed import seed_everything
from model.lambda_scheduler import *
from model.models import SimcseModelUnsup
from una import SimCSE

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

utils.logging.set_verbosity_error()  # Suppress standard warnings

parser = argparse.ArgumentParser(description="PyTorch SimCSE implementation")

parser.add_argument('--mode', type=str, default='una')
parser.add_argument('--disable_cuda', default=False, action='store_true', help='Whether you want to use cuda or not.')
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--do_mlm', default=True, action='store_true', help='Choose to add mlm function or not.')
parser.add_argument('--do_train', default=True, action='store_true')
parser.add_argument('--do_test', default=True, action='store_true')

# Parameters
parser.add_argument("--epochs", default=1, type=int, help="Set up the number of epochs you want to train.")
parser.add_argument("--batch_size", default=64, type=int, help="Set up the size of each batch you want to train.")
parser.add_argument("--batch_size_eval", default=256, type=int, help="Set up the size of each batch you want to evaluate.")
parser.add_argument("--lr", default=3e-5, type=float, help="Set up the learning rate.")
parser.add_argument("--max_len", default=32, type=int, help="Set up the maximum total input sequence length after tokenization.")
parser.add_argument("--pooling", choices=["cls", "cls_before_pooler", "last-avg", "first-last-avg", "last2_avg"], default="cls", type=str, help="Choose the pooling method")
parser.add_argument("--activation", default="tanh", type=str, choices=["tanh", "relu"])
parser.add_argument("--temperature", default=0.05, type=float, help="Set uo the temperature parameter.")
parser.add_argument("--dropout", default=0.1, type=float, help="Set up the dropout ratio")
parser.add_argument("--pretrained_model", default="roberta-base", type=str, choices=["bert-base-uncased", "roberta-base"])  
parser.add_argument("--num_workers", default=0, type=int)
# Paraphrasing
parser.add_argument("--do_para", default=False, action='store_true')
# UDA
parser.add_argument("--aug_every_n_steps", default=5, type=int)
parser.add_argument("--do_hard_negatives", default=False, action='store_true', help="apply hard negative sample generation or not")
parser.add_argument("--do_tf_idf", default=False, action='store_true', help="create hard negative pairs by replacing words using tf-idf")
parser.add_argument("--token_prob", default=0.5, type=float)
parser.add_argument("--tfidf_mode", default="nbh", choices=["uni", "nbh"], help=("uni mode for random replacement"))
parser.add_argument("--nbh_size", default=4000, type=int, help=["When mode is nbh, this decide the size of the neighbourhood radius"])
parser.add_argument("--gp_num", default=200, type=int, help=["When mode is group, this decide the number of divided groups"])
parser.add_argument("--p2_rand", default=False, action='store_true')
# Additional HP
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Set up the weight decay for optimizer.")
parser.add_argument("--log_every_n_steps", default=100, type=int, help="Frequency of keeping log")
parser.add_argument("--fp16_precision", action="store_true", help="Whether or not to use 516-bit precision GPU training.")
# Files
parser.add_argument("--train_data", type=str, default="./data/training/wiki1m_for_simcse.txt",
                    help="Choose the dataset you want to train with.")  
parser.add_argument("--para_train_data", type=str, default="./data/training/paraphrase.txt")
parser.add_argument("--dev_file", type=str, default="./data/stsbenchmark/sts-dev.csv")
parser.add_argument("--test_file", type=str, default="./data/stsbenchmark/sts-test.csv")
# Save model and define the output path. If the output is not none, the result is saved to result file by default.
parser.add_argument("--save_data", default=False, action='store_true', help="When the model is true, the pretrained model is saved under the output path")
parser.add_argument("--output_path", default="test")  

def main():
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    if args.output_path is not None:
        log_dir = os.path.join("results", args.mode, args.output_path)
        args.output_path = log_dir
    else:
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join("results", args.mode, current_time)
        args.output_path = log_dir
    os.makedirs(args.output_path, exist_ok=True)

    # Assert Bert/RoBERTa model
    if "roberta" in args.pretrained_model:
        args.arch = "roberta"
    elif "bert" in args.pretrained_model:
        args.arch = "bert"
    else:
        raise ValueError

    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = SimcseModelUnsup(args=args)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(args.device)

    # Load dataset
    train_dataset = TrainDataset(args=args, data=args.train_data, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              pin_memory=False)

    eval_dataset = SimcseEvalDataset(args=args, eval_mode="eval", tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size_eval,
                             num_workers=args.num_workers,
                             shuffle=True,
                             pin_memory=False)

    test_dataset = SimcseEvalDataset(args=args, eval_mode="test", tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size_eval,
                                 num_workers=args.num_workers,
                                 shuffle=True,
                                 pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    if args.arch == "bert":
        scheduler = get_linear_schedule(optimizer=optimizer, num_training_steps=len(train_loader)*args.epochs)
    elif args.arch == "roberta":
        scheduler = scheduler = get_constant_schedule(optimizer=optimizer)
    else:
        raise ValueError("Unsupported model type!")


    simcse = SimCSE(args=args, model=model, optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer)
    if args.do_train:
        simcse.train(train_loader=train_loader, eval_loader=eval_loader)
    if args.do_test:
        try:
            model.load_state_dict(torch.load(os.path.join(args.output_path, "simcse.pt")))
        except:
            model.load_state_dict(torch.load(os.path.join(args.test_model, "simcse.pt")))
        corrcoef = simcse.evaluate(model=model, dataloader=test_dataloader)
        print("corrcoef: {}".format(corrcoef))


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    main()



    
