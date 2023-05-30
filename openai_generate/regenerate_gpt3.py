import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import os
import torch, openai
import random, time
from utils import get_davinci003_response

openai_key = "" 
openai.api_key = openai_key


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/davinci_003_200_len.jsonl")
parser.add_argument('--output_dir', default="results")
parser.add_argument('--model', default="text-davinci-003") # "gpt-3.5-turbo", "gpt-4-0314",
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--total_instances', default=200, type=int)
parser.add_argument('--longest_charachter', default=1200, type=int)
parser.add_argument('--truncate_ratio', default=0.7, type=float)
parser.add_argument('--regen_number', default=20, type=int)
args = parser.parse_args()

print("Loading model {}".format(args.model))
# load saved json file
with open( args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

output_file = f"results/regen_davinci003_20_" + str( args.truncate_ratio ) +".jsonl"
random.seed(43)

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:]
print("Total instances: {}".format(len(data)))

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    question = dd['question'] 
    #prefix = "Continues the passage from the sentences provided in 180-300 words."
    gold_gen = dd['gold_completion']
    gen_completion = dd['gen_completion']
    
    prompt_gold_text = question + '\n' + gold_gen[ :int( args.truncate_ratio * len(gold_gen)) ]
    gold_gen_regen = get_davinci003_response(  prompt = prompt_gold_text, 
                                                max_tokens = args.max_new_tokens, 
                                                n= args.regen_number,
                                                logprobs=5,
                                                echo=False )
    time.sleep(1)
    original_human_response = get_davinci003_response(  prompt = question + '\n' + gold_gen,
                                                        max_tokens = 0,
                                                        n= 1,
                                                        logprobs=5,
                                                        echo=True)

    original_human_response_truncate = get_davinci003_response(  prompt = prompt_gold_text,
                                                        max_tokens = 0,
                                                        n= 1,
                                                        logprobs=5,
                                                        echo=True)
    
    prompt_gen_completion = question + '\n' + gen_completion[: int( args.truncate_ratio * len(gen_completion)) ]
    gen_completion_regen = get_davinci003_response( prompt = prompt_gen_completion, 
                                                max_tokens = args.max_new_tokens, 
                                                n= args.regen_number,
                                                logprobs=5,
                                                echo=False)

    original_gen_response = get_davinci003_response(    prompt = question + '\n' + gen_completion,
                                                        max_tokens = 0,
                                                        n = 1,
                                                        logprobs=5,
                                                        echo=True)

    original_gen_response_truncate = get_davinci003_response(    prompt = prompt_gen_completion,
                                                                max_tokens = 0,
                                                                n = 1,
                                                                logprobs=5,
                                                                echo=True)

    outputs.append(json.dumps({
        "question": question,
        "gold_gen_truncate": gold_gen[int( args.truncate_ratio*len(gold_gen)): ],
        "gen_completion_truncate": gen_completion[int( args.truncate_ratio*len(gen_completion)): ],
        "gold_gen_regen": gold_gen_regen,
        "original_human_response": original_human_response,
        "gen_completion_regen": gen_completion_regen,
        "original_gen_response": original_gen_response,
        "original_human_response_truncate": original_human_response_truncate,
        "original_gen_response_truncate": original_gen_response_truncate,
    }))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []
