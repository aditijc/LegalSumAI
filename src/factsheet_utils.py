import json
import pandas as pd
import csv
import numpy as np
import openai
from openai import OpenAI
from summary_utils import chain_of_density_prompting
from sklearn.metrics import accuracy_score

__all__ = ['get_openai_response', 'generate_fact_sheet', 'combine_fact_sheets', \
            'save_factsheet', 'break_text', 'generate_maud_knowledge_prompt']

#------------------------------------------------------------------------------

def get_openai_response(prompt, client):
    response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly intelligent Legal AI trained to solve reasoning problems and learn iteratively from feedback."},
                {"role": "user", "content": prompt}
            ]
        )
    return response.choices[0].message.content


def generate_prelim_fact_sheet(facts, client, verbose=True):
    prompt = f"""
    Given the following legal case, create a detailed fact sheet covering the following categories:
    1. Case Information
    2. Parties Involved
    3. Legal Basis
    4. Case Background
    5. Court Proceedings
    6. Settlement and Agreements
    7. Outcome and Impact
    8. Miscellaneous

    Case Info: {facts}

    Please provide the fact sheet in a structured format. Output type should
    be a JSON object with the following keys: case_info, parties (state names), legal_basis,
    case_background, court_proceedings, settlement_and_agreements, outcome_and_impact,
    miscellaneous.
    Do not have sub categories in the values of the JSON object.
    Make the values into string types.

    """
    response = get_openai_response(prompt, client)

    if verbose: print(response)
    return response

def generate_fact_sheet(case_info, client, verbose=True):
    factsheet = generate_prelim_fact_sheet(case_info, client, verbose)
    prompt = f"""
    Create a factsheet from the following case using the IRAC framework:
    
    1. Issue: Identify the main legal question in the case.
    2. Rule: State the relevant laws or legal principles applicable to the case.
    3. Application: Analyze how these laws apply to the specific facts of the case.
    4. Conclusion: Provide a summary of the final decision or outcome of the case.
    
    Case Information: {factsheet}
    
    Please provide the fact sheet in a structured format. Output type should
    be a JSON object with the following keys: Case Information, Parties Involved, Issue, Rule, Application, Conclusion.
    Do not have sub categories in the values of the JSON object.
    Make the values into string types.
    
    """
    response = get_openai_response(prompt, client)

    if verbose: print(response)
    return response

def combine_fact_sheets(responses, client, verbose=True):
    summary = '\n'.join(responses)
    prompt = f"""
    Given the following JSON objects appended together, merge these factsheets
    together to create a detailed fact sheet using the IRAC framework:
    1. Issue: Identify the main legal question in the case.
    2. Rule: State the relevant laws or legal principles applicable to the case.
    3. Application: Analyze how these laws apply to the specific facts of the case.
    4. Conclusion: Provide a summary of the final decision or outcome of the case.

    Case Information: {summary}

    Please provide the fact sheet in a structured format. Output type should
    be a JSON object with the following keys: Case Information, Parties Involved, Issue, Rule, Application, and Conclusion.
    Do not have sub categories in the values of the JSON object.
    Make the values into string types.

    """
    response = get_openai_response(prompt, client)
    if verbose: print(response)
    return response


def save_factsheet(response, filename):
  if not response.startswith("{"):
    response = '\n'.join(response.split('\n')[1:-1])
  data = json.loads(response)
  print(data)
  with open(f"{filename}.csv", 'w', newline='') as file:
      writer = csv.writer(file)
      # Write the header
      writer.writerow(['Category', 'Details'])
      # Write the data
      for key, value in data.items():
        print(key)
        print(value)
        writer.writerow([key, value])


def break_text(raw_case, word_thresh=10000):
  # Max context length is 16385 tokens, assuming 1 token : 0.75 words
  case_words = raw_case.split(' ')
  sections = [' '.join(case_words[i:i+word_thresh]) for i in range(0, len(case_words), word_thresh)]
  return sections


# Function to generate the prompt for the maud_knowledge_definition task
def generate_maud_knowledge_prompt(text):
    prompt = f"""
    Read the following excerpt from a merger agreement and determine what counts as Knowledge.
    
    Excerpt: {text}
    
    Options:
    A: Actual knowledge
    B: Constructive knowledge
    
    Answer:
    """
    return prompt


# Function to classify text using chain_of_density_prompting
def classify_knowledge_one_shot(text, client):
    prompt = generate_maud_knowledge_prompt(text)
    initial_prompt = f"Given the following task and excerpt, create a factsheet with key information. \n \
        task: {prompt}."
    response_factsheet = get_openai_response(initial_prompt, client)
    task_specific_prompt = f"""Given the following factsheet with information about the task: \n
    Factsheet: {response_factsheet} \n Solve this task by returning the multiple choice answer
    regardless of whether or not you think you
    have enough information. \n Task: {prompt}."""
    response_task = get_openai_response(task_specific_prompt, client)
    return response_task

# Function to evaluate the model on a given dataset
def evaluate_model(dataset, client):
    predictions = []
    true_labels = []
    for _, row in dataset.iterrows():
        text = row['text']
        true_label = row['label']
        
        predicted_label = classify_knowledge_one_shot(text, client)
        print(f'Predicted Label: {predicted_label}')
        
        # Map the model's response to the corresponding label
        if 'Actual knowledge' in predicted_label:
            predicted_label = 0
        elif 'Constructive knowledge' in predicted_label:
            predicted_label = 1
        else:
            predicted_label = None
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
    
    # Compute the accuracy
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

if __name__ == "__main__":
    df = pd.read_csv("../data/maud_knowledge_definition.tsv", sep="\t", header=None, names=["label", "text"])
    print(df)
    # exit()
    client = OpenAI(api_key = input('Enter API Key: '))
    label_mapping = {"Actual knowledge": 0, "Constructive knowledge": 1}
    df["label"] = df["label"].map(label_mapping)

    accuracy = evaluate_model(df, client)
    print(f"Model Accuracy: {accuracy:.2f}")
