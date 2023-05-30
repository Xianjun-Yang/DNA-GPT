import os, json, tqdm
import requests

api_key = ''

class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'
    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()
# Credits for this code go to https://github.com/Haste171/gptzero
output_file = 'results/detect_gptzero_200.jsonl'
with open( 'results/chatgpt_200_len.jsonl', 'r') as f:
    new_data = [json.loads(x) for x in f.read().strip().split("\n")]
print( 'Total instances: {}'.format( len( new_data)))

gptzero = GPTZeroAPI(api_key)

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = new_data[num_curr_outputs:]
print("Total instances: {}".format(len(data)))

outputs = []
for i, instance in tqdm.tqdm(enumerate(new_data), total=len(new_data)):
    gold_gen = instance['gold_completion']
    gen_completion = instance['gen_completion']

    gold_gen_prob = gptzero.text_predict(gold_gen)['documents'][0]['completely_generated_prob']
    gen_completion_prob = gptzero.text_predict(gen_completion)['documents'][0]['completely_generated_prob']

    new_data[i]['gold_gen_prob'] = gold_gen_prob
    new_data[i]['gen_completion_prob'] = gen_completion_prob

    outputs.append(json.dumps( new_data[i]))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []