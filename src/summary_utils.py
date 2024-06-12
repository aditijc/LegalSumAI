import json
import pandas as pd
import csv
import numpy as np
import openai
from openai import OpenAI
from metrics_utils import *

__all__ = ['get_openai_response_summary', 'generate_summary', 'chain_of_density_prompting'] # Put all public method names in this list

#------------------------------------------------------------------------------

def get_openai_response_summary(prompt, template, client):
    response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": prompt}
            ]
    )
    return response.choices[0].message.content


def generate_summary(processed_string, client, task = "Summarize the factsheet so that it is understable to the general public ", num_words=325, total_tries=5):
    template = f'''
    Setting: You are a helpful assistant designed to assist users in summarizing factsheets.
    You have one main requirement that your response should be {num_words} long.

    Provide clear and precise answers to the user's questions. Avoid unnecessary details and keep your responses brief and to the point.

    Your goal is to understand the user's request, and provide text to
    fulfill the request.

    Your input will be a factsheet with the following format -->
    category1 : detail1
    category2 : detail2
    ...

    Your output will be text.
    '''

    prompt_create = f'''
    You are a lawyer describing the court case to the general public.
    Given the factsheet in the following text format -->
    category1 : detail1
    category2 : detail2
    ...

    {task}
    The summary should be in paragraph form. It must have exactly {num_words} words. Below is the factsheet:
    {processed_string}.

    Make sure it has exactly {num_words} words
     '''
    # Prompting until we get 5 summaries with approximately num_words words
    try_times = 0
    chat_summaries_short = []
    TOTAL_TRIES = total_tries
    variance = int(.20*num_words)
    max_tries = total_tries
    counter = 0
    while try_times < TOTAL_TRIES:
      summary = get_openai_response_summary(prompt=prompt_create, template=template, client=client)
      sum_words = len(summary.split(" "))
      if sum_words > num_words - variance and sum_words < num_words + variance:
        try_times += 1
        counter = 0
        chat_summaries_short.append(summary)
      if counter > max_tries:
        chat_summaries_short.append(summary)
        counter += 1
        break
    return chat_summaries_short
  

def find_best_summary(scores):
  best_index = scores.index(max(scores))
  return best_index


def chain_of_density_prompting(processed_csv, initial_prompt, client, ground_truth=None, num_words=325):
    responses = []
    # Refining steps
    prompts = [initial_prompt,
        f"providing more details about the contract terms that were disputed in exactly {num_words} words.",
        f"explaining the arguments presented by both parties in the court case in exactly {num_words} words.",
        f"describing the court's decision and the reasoning behind it in exactly {num_words} words."
    ]

    current_response = None
    for prompt in prompts:
        if current_response:
          prompt = f"Here is the previous summary {current_response}. Improve the summary by " + prompt
        if ground_truth:
            current_response = generate_summary(processed_csv, client, f"{prompt}", num_words, total_tries=5) # total tries = 5
            cosine_sim_scores = compute_cosine_similarity_list(current_response, ground_truth)
            best_idx = cosine_sim_scores.index(max(cosine_sim_scores))
            current_response = current_response[best_idx]
        else:
            current_response = generate_summary(processed_csv, client, f"{prompt}", num_words, 1)
        responses.append(current_response)

    return responses

    
