import joblib
import numpy as np
import bm25s
import os
from sentence_transformers import SentenceTransformer

from utils import (
    read_dataframe,
    pprint, 
    generate_with_single_input, 
    cosine_similarity,
    display_widget
)
import unittests

NEWS_DATA = read_dataframe("news_data_dedup.csv")

def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    dataset (list or sequence): The dataset from which elements are to be retrieved. It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
     
    output = [NEWS_DATA[index] for index in indices]

    return output


# The corpus used will be the title appended with the description
corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]

# Instantiate the retriever by passing the corpus data
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)

# Tokenize the chunks
tokenized_data = bm25s.tokenize(corpus)

# Index the tokenized chunks within the retriever
BM25_RETRIEVER.index(tokenized_data)

# Tokenize the same query used in the previous exercise
sample_query = "What are the recent news about GDP?"
tokenized_sample_query = bm25s.tokenize(sample_query)

# Get the retrieved results and their respective scores
results, scores = BM25_RETRIEVER.retrieve(tokenized_sample_query, k=3)

print(f"Results for query: {sample_query}\n")
for doc in results[0]:
  print(f"Document retrieved {corpus.index(doc)} : {doc}\n")