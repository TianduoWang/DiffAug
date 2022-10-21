import sys
import io, os
import numpy as np
import logging
import argparse
from argparse import Namespace
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from diffaug.models import BertForCL
from utils import get_encoded_bs_and_es

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, 
            help='Transformers model name or path.')
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help='What evaluation mode to use.')
    parser.add_argument('--prefix_len', type=int, default=16)
    parser.add_argument('--temp', type=float, default=0.05)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    template = '*cls*_This_sentence_of_"*sent0*"_means*mask*.*sep+*'
    _, _, _, enc_bs, enc_es, enc_template = get_encoded_bs_and_es(template, tokenizer)

    template = template.replace('*mask*', tokenizer.mask_token )\
                       .replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')

    m_args={
        "apply_prompt": True,
        "mask_token_id": tokenizer.mask_token_id,
        "enc_bs": enc_bs,
        "enc_es": enc_es,
        "enc_template": enc_template,
        "use_prefix": True,
        "prefix_len": args.prefix_len,
        "temp": args.temp,
        "sup_label_num": 2,
    }
    m_args = Namespace(**m_args)
    model = BertForCL.from_pretrained(args.model_name_or_path, model_args=m_args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up the tasks
    if args.mode == 'test':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    else:
        args.tasks = ['STSBenchmark']
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        for i, s in enumerate(sentences):
            sentences[i] = template.replace('*sent0*', s).strip()
        # Tokenization
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
        return outputs.pooler_output.cpu()

    
    results = {}
    for task in args.tasks:
        se = senteval.engine.SE({'task_path': PATH_TO_DATA}, batcher)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))
        task_names, scores = [], []
        task_names.append('STSBenchmark')
        if task in results:
            scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
        else:
            scores.append("0.00")
        print_table(task_names, scores)

    elif args.mode == 'test':
        print("------ %s ------" % (args.mode))
        task_names, scores = [], []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()