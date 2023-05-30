import numpy as np
import openai

import re, six
import spacy
from nltk.stem.porter import PorterStemmer
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

PorterStemmer = PorterStemmer()
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

def get_openai_response(prompt: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    output = response['choices'][0]['text']
    assert output.startswith(prompt)
    gen_text = output[len(prompt):].strip()
    return gen_text

def get_davinci003_response(prompt: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    # output = response['choices'][0]['text']
    # assert output.startswith(prompt)
    # gen_text = output[len(prompt):].strip()
    return response

def get_chatgpt_qa_response(prompt_text, temperature = 0.7, max_tokens=1000):
    messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_gpt4_qa_response(prompt_text, temperature = 0.7, max_tokens=1000):
    messages = [{"role":"system", "content": "You are a helpful assistant that answers the question provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-4-0314",
                messages = messages,
                temperature = temperature,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_gpt4_completion_response(prompt_text, max_tokens):
    messages = [{"role":"system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-4-0314",
                messages = messages,
                temperature = 0.7,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

def get_chatgpt_completion_response(prompt_text, max_tokens):
    messages = [{"role":"system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                {"role":"user", "content": prompt_text}]
    response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages,
                temperature = 0.7,
                max_tokens = max_tokens
    )
    return response['choices'][0]['message']['content']

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
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in stopwords]

    # One final check to drop any empty or invalid tokens.
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

    return tokens
