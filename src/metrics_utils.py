import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ['compute_cosine_similarity']

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