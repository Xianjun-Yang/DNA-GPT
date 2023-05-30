import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json, math
import os
import torch, openai
import random
from torch.nn.functional import softmax

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

path = "" # path to the model

tokenizer = LlamaTokenizer.from_pretrained( path )

def gptx_generate(prompt, max_length, echo=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(): # torch.no_grad() or torch.cuda.amp.autocast():
        gen_tokens = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=0.7,
            max_length= max_length,
            num_return_sequences = 1,
        )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if echo:
        return gen_text, gen_text.startswith(prompt)
    else:
        return gen_text[len(prompt):], gen_text.startswith(prompt)
    
def gptx_prob(prompt, gen_text, echo=False):
    inputs = tokenizer(prompt + gen_text, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad(): # torch.no_grad() or torch.cuda.amp.autocast():
        outputs = model(
                        input_ids = inputs["input_ids"],
                        attention_mask= inputs["attention_mask"],
                        )
    logits = outputs.logits
    probabilities = softmax(logits, dim=-1)
    indices = inputs['input_ids'][0][1:].unsqueeze(-1)  # Add an extra dimension to match the probabilities tensor
    token_prob = torch.gather(probabilities[0, :, :], dim=1, index=indices).squeeze().tolist()
    # convert token_probabilities to log_probabilities, for inf value, use -10000, in one line
    log_probabilities = [math.log(x) if x != 0 else -100 for x in token_prob ]
    # Get the probability of each token in the input text
    if echo:
        return log_probabilities
    else:
        return log_probabilities[ prompt_inputs['input_ids'].size(1) - 1: ] # remove the start token

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="xsum_truncate/gptllama_150_len.jsonl")
parser.add_argument('--model', default="gptllama") # "gpt-3.5-turbo", "gpt-4-0314",
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--longest_charachter', default=1200, type=int)
parser.add_argument('--truncate_ratio', default=0.5, type=float)
parser.add_argument('--regen_number', default=10, type=int)
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()

print("Loading model {}".format(args.model))
device = torch.device("cuda:" + str( args.device ))
model = LlamaForCausalLM.from_pretrained( path ).half().to(device) #.half() to use FP16
model.eval() 

# load saved json file
with open( args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

output_file = f"xsum_truncate/wop_regen_gptllama_10_" + str(args.truncate_ratio) + ".jsonl"
random.seed(43)

data = data[ :]
print("Total instances: {}".format(len(data)))

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    #prefix = dd['prefix']
    question = dd['question']  # no prefix for reddit
    #question =  prefix +# question  # for xsum
    #question = "Continues the following sentences in around 200 words: " # for xsum
    gold_gen = dd['gold_completion']
    gen_completion = dd['gen_completion']
    
    prompt_gold_text = question + '\n' + gold_gen[ :int( args.truncate_ratio * len(gold_gen)) ]
    # regen from gptx_generate and gptx_prob
    gold_gen_regen = []
    for i in range(args.regen_number):
        truth = False
        count = 0
        while not truth:
            if count > 1 and count < 10:
                print('regen')
            elif count > 10:
                break
            gen_text, truth = gptx_generate(prompt_gold_text, args.max_new_tokens, echo=False)
            count += 1
        gptx_probabilities = gptx_prob( prompt_gold_text, gen_text, echo=False)
        gold_gen_regen.append(  {'gen_text': gen_text, 'gptx_probabilities': gptx_probabilities} )
    
    original_human_response = gptx_prob( question + '\n',
                                         gold_gen,
                                         echo=True)
    
    original_human_response_truncate = gptx_prob(  prompt_gold_text,
                                                    '',
                                                    echo=True)

    prompt_gen_completion = question + '\n' + gen_completion[: int( args.truncate_ratio * len(gen_completion)) ]
    gen_completion_regen = []
    for i in range(args.regen_number):
        truth = False
        count = 0
        while not truth:
            if count > 1 and count < 10:
                print('regen')
            elif count > 10:
                break
            gen_text, truth = gptx_generate(prompt_gen_completion, args.max_new_tokens, echo=False)
            count += 1
        gptx_probabilities = gptx_prob( prompt_gen_completion, gen_text, echo=False)
        gen_completion_regen.append(  {'gen_text': gen_text, 'gptx_probabilities': gptx_probabilities} )
    
    original_gen_response = gptx_prob( question + '\n',
                                        gen_completion,
                                        echo=True)
    
    original_gen_response_truncate = gptx_prob(  prompt_gen_completion,
                                                    '',
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
