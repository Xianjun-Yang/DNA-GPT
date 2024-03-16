## Official release for paper DNA-GPT: https://arxiv.org/abs/2305.17359
## Authors: Xianjun Yang (UCSB), Wei Cheng (NEC Labs America)
## MIT License
# Copyright (c) 2023 Xianjun (Nolan) Yang, NEC Labs America


####### install the following packages before using #########
# %pip install openai
# %pip install spacy
# %pip install nltk
# %pip install gradio
# %pip install rouge_score
# !python3 -m spacy download en_core_web_sm


import gradio as gr
import ssl
import nltk
import os
import openai
import re
from gradio import components

import six
import spacy
from nltk.stem.porter import PorterStemmer
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
import six
import numpy as np

openai_key = "" ### OpenAI AIP key for access to the re-generation service, users are suggested to delete the key after usage to avoid potential leakage of key
openai.api_key = openai_key
temperature = 0.7  ### This parameter controls text quality of chatgpt, by default it was set to 0.7 in the website version of ChatGPT.
max_new_tokens = 300  ### maximum length of generated texts from chatgpt
regen_number = 30  ### for faster response, users can set this value to smaller ones, such as 20 or 10, which will degenerate performance a little bit
question = ""
truncate_ratio = 0.5
threshold = 0.00025  ### for conservative decision, users can set this value to larger values, such as 0.0003 



PorterStemmer = PorterStemmer()
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')


def tokenize(text, stemmer, stopwords=[]):
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.

    Returns:
    A list of string tokens extracted from input text.
    """

    # Convert everything to lowercase.
    text = text.lower()
    # Replace any non-alpha-numeric characters with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

    tokens = re.split(r"\s+", text)
    if stemmer:
        # Only stem words more than 3 characters long.
        tokens = [stemmer.stem(x) if len(
            x) > 3 else x for x in tokens if x not in stopwords]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

    return tokens


def get_score_ngrams(target_ngrams, prediction_ngrams):
    """calcualte overlap ratio of N-Grams of two documents.

    Args:
    target_ngrams: N-Grams set of targe document.
    prediction_ngrams: N-Grams set of reference document.

    Returns:
    ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count]
    """
    intersection_ngrams_count = 0
    ngram_dict = {}
    for ngram in six.iterkeys(target_ngrams):
        intersection_ngrams_count += min(target_ngrams[ngram],
                                        prediction_ngrams[ngram])
        ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
    return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict


def get_ngram_info(article_tokens, summary_tokens, _ngram):
    """calculate N-Gram overlap score of two documents
    It use _create_ngrams in rouge_score.rouge_scorer to get N-Grams of two docuemnts, then revoke get_score_ngrams method to calucate overlap score
    Args:
    article_tokens: tokens of one document.
    summary_tokens: tokens of another document.

    Returns:
    ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count], total overlap n-gram count
    """
    article_ngram = _create_ngrams( article_tokens , _ngram)
    summary_ngram = _create_ngrams( summary_tokens , _ngram)
    ngram_score, ngram_dict = get_score_ngrams( article_ngram, summary_ngram) 
    return ngram_score, ngram_dict, sum( ngram_dict.values() )


def N_gram_detector(ngram_n_ratio):
    """calculate N-Gram overlap score from N=3 to N=25
    Args:
    ngram_n_ratio: a list of ratio of N-Gram overlap scores, N is from 1 to 25.

    Returns:
    N-Gram overlap score from N=3 to N=25 with decay weighting n*log(n)
    """
    score = 0
    non_zero = []

    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and 'score' in key or 'ratio' in key:
            score += 0. * ngram_n_ratio[key]
            continue
        if 'score' in key or 'ratio' in key:
            score += (idx+1) * np.log((idx+1)) * ngram_n_ratio[key]
            if ngram_n_ratio[key] != 0:
                non_zero.append(idx+1)
    return score / (sum(non_zero) + 1e-8)


def N_gram_detector_ngram(ngram_n_ratio):
    """sort the dictionary of N-gram key according to their counts
    Args:
    ngram_n_ratio: a list of ratio of N-Gram overlap scores, N is from 1 to 25.

    Returns:
    sorted dictionary of N-gram [key,value] according to their value counts
    """
    ngram = {}
    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and 'score' in key or 'ratio' in key:
            continue
        if 'ngramdict' in key:
            dict_ngram = ngram_n_ratio[key]

            for key_, value_ in dict_ngram.items():
                ngram[key_] = idx
    sorted_dict = dict(
        sorted(ngram.items(), key=lambda x: x[1], reverse=True))
    return sorted_dict


def tokenize_nltk(text):
    """tokenize text using word tokenizer
    Args:
    text: input text

    Returns:
    tokens of words
    """
    tokens = nltk.word_tokenize(text)
    return tokens


def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


def getOverlapedTokens(text1, text2):
    """get overlap of word tokens of two documents
    Args:
    text1: input text1
    text2: input text2

    Returns:
    dict of [token:count] of overlape words in two texts
    """
    overlap_dict = {}
    tokens1 = tokenize_nltk(text1.lower())
    tokens2 = tokenize_nltk(text2.lower())
    for n in range(3, 25):
        ngrams1 = get_ngrams(tokens1, n)
        ngrams2 = get_ngrams(tokens2, n)
        ngrams_set1 = set(ngrams1)
        ngrams_set2 = set(ngrams2)
        overlap = ngrams_set1.intersection(ngrams_set2)
        for element in overlap:
            overlap_dict[element] = n
    return overlap_dict


def get_html(text, dictionary):
    """prepare html format content 
    Args:
    text: input text that is reference 
    dictionary: overlap dict, overlap is calculated with reference text that is generated by 
                ChatGPT with prefix input that is the first half of the text.

    Returns:
    visualize html to output the results for explain why it is judged as generated by chatgpt
    """
    positions = []

    len_text = len(text)
    flag_vec = np.zeros(len_text)

    str_html = ""
    for key in dictionary:
        start = 0
        while True:
            index = text.find(key, start)
            if index == -1:
                break
            positions.append((key, index))
            flag_vec[index:index+len(key)] = 1
            start = index + len(key)
    status = flag_vec[0]
    # print(sum(flag_vec))
    for i in range(len_text):
        if i == 0:
            if status == 0:
                str_html = str_html + "<span>"+text[i]
            else:
                str_html = str_html + "<span style='color: red;'>"+text[i]
            continue
        if flag_vec[i] == status:
            str_html = str_html + text[i]
        else:
            str_html = str_html + "</span>"
            if flag_vec[i] == 0:
                str_html = str_html + "<span>"+text[i]
            else:
                str_html = str_html + "<span style='color: red;'>"+text[i]
            status = flag_vec[i]
        if i == len_text - 1:
            str_html = str_html + "</span>"
    return str_html


def truncate_string_by_words(string, max_words):
    words = string.split()
    if len(words) <= max_words:
        return string
    else:
        truncated_words = words[:max_words]
        return ' '.join(truncated_words)


def detection(text, option):
    """detect if give text is generated by chatgpt 
    Args:
    text: input text 
    option: target model to be checked: gpt-3.5-turbo or gpt-4-0314.

    Returns:
    decision: boolean value that indicate if the text input is generated by chatgpt with version as option
    most_matched_generatedtext[0]: most matched re-generated text by chatgpt using half prefix of the input text as the prompt
    """
    max_words = 350 #### can be adjusted
    if option == "GPT-3.5":
        model_name = "gpt-3.5-turbo-instruct"
    else:
        model_name = "gpt-4-0314"
    text = truncate_string_by_words(text, max_words)
    ngram_overlap_count =[]
    question = "continues the passage from the current text within in total around 300 words:"
    input_text = text
    human_prefix_prompt = input_text[:int(truncate_ratio*len(input_text))]
    # human_gen_text = openai.ChatCompletion.create(model=model_name,
    #                                             messages=[{"role": "system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
    #                                                         {"role": "user",
    #                                                             "content": question},
    #                                                         {"role": "assistant",
    #                                                             "content": human_prefix_prompt},
    #                                                         ],
    #                                             temperature=temperature,
    #                                             max_tokens=max_new_tokens,
    #                                             n=regen_number)
    from openai import OpenAI
    client = OpenAI(api_key=openai.api_key)
    completion = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=human_prefix_prompt,
        max_tokens=max_new_tokens,
        temperature=temperature
    )

    input_remaining = input_text[int(truncate_ratio*len(input_text)):]
    input_remaining_tokens = tokenize(input_remaining, stemmer=PorterStemmer)

    temp = []
    mx = 0
    mx_v = 0
    for i in range(regen_number):  # len(human_half)
        temp1 = {}
        gen_text = human_gen_text['choices'][i]['message']['content']

        ###### optional #######
        gen_text_ = truncate_string_by_words(gen_text, max_words-150)

        gpt_generate_tokens = tokenize(
            gen_text_, stemmer=PorterStemmer)
        if len(input_remaining_tokens) == 0 or len(gpt_generate_tokens) == 0:
            continue

        for _ngram in range(1, 25):
            ngram_score, ngram_dict, overlap_count = get_ngram_info(
                input_remaining_tokens, gpt_generate_tokens, _ngram)
            temp1['human_truncate_ngram_{}_score'.format(
                _ngram)] = ngram_score / len(gpt_generate_tokens)
            temp1['human_truncate_ngram_{}_ngramdict'.format(
                _ngram)] = ngram_dict
            temp1['human_truncate_ngram_{}_count'.format(
                _ngram)] = overlap_count

            if overlap_count > 0:
                if _ngram > mx_v:
                    mx_v = _ngram
                    mx = i

        temp.append({'machine': temp1})

    ngram_overlap_count.append(temp)
    gpt_scores = []

    top_overlap_ngram = []
    max_ind_list = []
    most_matched_generatedtext = []
    for instance in ngram_overlap_count:
        human_score = []
        gpt_score = []

        for i in range(len(instance)):
            # human_score.append(N_gram_detector(instance[i]['human']))
            gpt_score.append(N_gram_detector(instance[i]['machine']))
            top_overlap_ngram.append(
                N_gram_detector_ngram(instance[i]['machine']))

        # human_scores.append(sum(human_score))
        gpt_scores.append(np.mean(gpt_score))
        max_value = max(gpt_score)
        # print(len(gpt_score))
        max_index = mx  # gpt_score.index(max_value)
        max_ind_list.append(max_index)
        most_matched_generatedtext.append(
            human_gen_text['choices'][max_index]['message']['content'])
    print(gpt_scores[0])

    if gpt_scores[0] > threshold:
        decision = True
    else:
        decision = False
    return decision, most_matched_generatedtext[0]


def generate_html(key, option, text ):
    """prepare html format content for decision of DNA-GPT detection tool
    Args:
    key: key for access openai API, visit https://platform.openai.com/account/api-keys to check keys
    option: which version of OpenAI chatgpt to check: chatgpt 3.5 or 4.0
    text: input text  
    
    Returns:
    visualize html to output the results 
    """
    openai_key = key
    openai.api_key = openai_key
    label = "Detection Output: "
    res, max_over_lap_generatedtext = detection(text, option)
    text = text.lower()
    max_over_lap_generatedtext = max_over_lap_generatedtext.lower()
    dictionary = getOverlapedTokens(text, max_over_lap_generatedtext)
    html_ = get_html(text, dictionary)
    html_gen = get_html(max_over_lap_generatedtext, dictionary)
    if res:
        html_text = f"<p><strong>{label}:</strong></p> <p style='color: red;'>It is generated by ChatGPT3.5!<br><br></p><p><strong>Evidence:<br></strong><br></p><p><strong>Your Input:<br></strong><br></p>" + \
            html_+"<p><br><br><br><br><strong>GPT generated text by continue writing uisng half-input prompt:<br></strong><br></p>......."+html_gen
    else:
        html_text = f"<p><strong>{label}:</strong></p> <p style='color: green;'>It is generated by human!</p>"
    return html_text


def main():
    # input_key = gr.inputs.Textbox(
    # label="Enter your key for OpenAI access:", lines=1)
    input_key = components.Textbox(
        type="text", label="Enter your key for OpenAI access:")

    input_text = components.Textbox(
        type="text", label="Enter your text (Ideally between 180 and 300 words)")
    
    # input_text = gr.inputs.Textbox(
    #     label="Enter your text (Ideally between 180 and 300 words)", lines=20)

    copyright = "© 2023 NEC Labs America | Author: Xianjun Yang, Wei Cheng"
    
    # sigle choice option
    radio_options = ["GPT-3.5", "GPT-4"]
    radio = components.Radio(
        radio_options, label="Please choose one version to check:")

    input_group = [input_key, radio, input_text]

    greet_output = gr.outputs.Textbox(label="Detection Output:")

    interface = gr.Interface(fn=generate_html, inputs=input_group, title="DNA-GPT Demo (NEC Labs America)",
                            outputs=gr.outputs.HTML(), description=copyright)
    interface.launch(share=True)

if __name__ == "__main__":
    main()

