import argparse
import tqdm
import json, math
import os
import torch
import random
from torch.nn.functional import softmax
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch

def gptx_generate(prompt, max_length, echo=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(): 
        gen_tokens = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=0.7,
            max_length= max_length,
            num_return_sequences = 1,
        )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    if echo:
        return gen_text, gen_text.startswith(prompt)
    else:
        return gen_text[len(prompt):], gen_text.startswith(prompt)
    
def gptx_prob(prompt, gen_text, echo=False):
    inputs = tokenizer(prompt + gen_text, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad(): 
        outputs = model(
                        input_ids = inputs["input_ids"],
                        attention_mask= inputs["attention_mask"],
                        )
    logits = outputs.logits
    probabilities = softmax(logits, dim=-1)
    # Get the probability of each token in the input text
    indices = inputs['input_ids'][0][1:].unsqueeze(-1)  # Add an extra dimension to match the probabilities tensor
    token_probabilities = torch.gather( probabilities[0, :, :], 1, indices).squeeze().tolist()
    token_probabilities = [0] + token_probabilities # add dummy probability for the first token
    log_probabilities = [math.log(x) if x != 0 else -100 for x in token_probabilities ]
    if echo:
        return log_probabilities
    else:
        return log_probabilities[ prompt_inputs['input_ids'].size(1): ]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results_reddit/gptx_200_len.jsonl")
parser.add_argument('--model', default="gptx")
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--longest_charachter', default=1200, type=int)
parser.add_argument('--truncate_ratio', default=0.5, type=float)
parser.add_argument('--regen_number', default=20, type=int)
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()

print("Loading model {}".format(args.model))
device = torch.device("cuda:" + str( args.device ))
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().to(device) #.half() to use FP16
model.eval()  # Set the model to evaluation mode

# load saved json file
with open( args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

output_file = f"results_reddit/wop_regen_gptx_20" + str(args.truncate_ratio) + ".jsonl"
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
    prefix = dd['prefix']
    question = prefix 
    
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
