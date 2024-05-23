import json
import pandas as pd
import csv
import numpy as np
import openai
from openai import OpenAI

__all__ = ['get_openai_response', 'generate_fact_sheet', 'combine_fact_sheets', 'save_factsheet']

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

def generate_fact_sheet(summary, client, verbose=True):
    prompt = f"""
    Given the following legal case summary, create a detailed fact sheet covering the following categories:
    1. Case Information
    2. Parties Involved
    3. Legal Basis
    4. Case Background
    5. Court Proceedings
    6. Settlement and Agreements
    7. Outcome and Impact
    8. Miscellaneous

    Summary: {summary}

    Please provide the fact sheet in a structured format. Output type should
    be a JSON object with the following keys: case_info, parties, legal_basis,
    case_background, court_proceedings, settlement_and_agreements, outcome_and_impact,
    miscellaneous.
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
    together to create a detailed fact sheet covering the following categories:
    1. Case Information
    2. Parties Involved
    3. Legal Basis
    4. Case Background
    5. Court Proceedings
    6. Settlement and Agreements
    7. Outcome and Impact
    8. Miscellaneous

    Summary: {summary}

    Please provide the fact sheet in a structured format. Output type should
    be a JSON object with the following keys: case_info, parties, legal_basis,
    case_background, court_proceedings, settlement_and_agreements, outcome_and_impact,
    miscellaneous.
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


