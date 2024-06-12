import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


__all__ = ['compute_cosine_similarity', 'compute_cosine_similarity_list', 'exact_match']

def exact_match(pred, ground_truth):
    if len(pred) < len(ground_truth):
        ground_truth = ground_truth[:len(pred)]
    elif len(pred) > len(ground_truth):
        pred = pred[:len(ground_truth)]
    exact_match = load("exact_match")
    results = exact_match.compute(references=pred, predictions=ground_truth)
    return round(results["exact_match"], 2)

def compute_cosine_similarity(sheet1, sheet2, verbose=True):
  """
  Compute the cosine similarity between the source and summary sheets.
  source_sheet: string representation of the source sheet
  summary_sheet: string representation of the summary sheet
  """
  # Create the TF-IDF vectors
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform([sheet1, sheet2])

  # Compute the cosine similarity
  cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
  if verbose: print(f"Cosine Similarity: {cosine_sim[0][0]}")
  return cosine_sim[0][0]
  
def compute_cosine_similarity_list(chat_summaries, ground_truth, verbose=True):
  """
  Compute the cosine similarity between the source and summary sheets.
  source_sheet: string representation of the source sheet
  summary_sheet: string representation of the summary sheet
  """
  cosine_scores = []
  vectorizer = TfidfVectorizer()
  for summary in chat_summaries:
    tfidf_matrix = vectorizer.fit_transform([summary, ground_truth])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    cosine_scores.append(cosine_sim[0][0])
  return cosine_scores
