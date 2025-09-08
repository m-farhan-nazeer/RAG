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

# Use these as a global defined BM25 retriever objects

corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)
TOKENIZED_DATA = bm25s.tokenize(corpus)
BM25_RETRIEVER.index(TOKENIZED_DATA)



def bm25_retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top k relevant documents for a given query using the BM25 algorithm.

    This function tokenizes the input query and uses a pre-indexed BM25 retriever to
    search through a collection of documents. It returns the indices of the top k documents
    that are most relevant to the query.

    Args:
        query (str): The search query for which documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k relevant documents
        within the corpus.
    """
    ### START CODE HERE ###

    # Tokenize the query using the 'tokenize' function from the 'bm25s' module
    tokenized_query =bm25s.tokenize(query)
    
    # Use the 'BM25_RETRIEVER' to retrieve documents and their scores based on the tokenized query
    # Retrieve the top 'k' documents
    results, scores = BM25_RETRIEVER.retrieve(tokenized_query, k=top_k)

    # Extract the first element from 'results' to get the list of retrieved documents
    retrieved_docs = results[0]

    # Convert the retrieved documents into their corresponding indices in the results list
    top_k_indices = [corpus.index(doc) for doc in retrieved_docs]

    ### END CODE HERE ###
    
    return top_k_indices


    # Load the pre-computed embeddings with joblib
EMBEDDINGS = joblib.load("embeddings.joblib")

model_name = os.path.join(os.environ['MODEL_PATH'],"BAAI/bge-base-en-v1.5" )
model = SentenceTransformer(model_name)


# Example usage
query = "RAG is awesome"
# Using, but truncating the result to not pollute the output, don't truncate it in the exercise.
model.encode(query)[:40]
    