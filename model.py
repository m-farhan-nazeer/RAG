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
    

query1 = "What are the primary colors"
query2 = "Yellow, red and blue"
query3 = "Cats are friendly animals"

query1_embed = model.encode(query1)
query2_embed = model.encode(query2)
query3_embed = model.encode(query3)

print(f"Similarity between '{query1}' and '{query2}' = {cosine_similarity(query1_embed, query2_embed)[0]}")
print(f"Similarity between '{query1}' and '{query3}' = {cosine_similarity(query1_embed, query3_embed)[0]}")


query = "Taylor Swift"
query_embed = model.encode(query)
# The result is a matrix with one matrix per sample. Since there is only one sample (the query), it is a matrix with one matrix within.
# This is why you need to get the first element
similarity_scores = cosine_similarity(query_embed, EMBEDDINGS)
similarity_indices = np.argsort(-similarity_scores) # Sort on decreasing order (sort the negative on increasing order), but return the indices
# Top 2 indices
top_2_indices = similarity_indices[:2]
print(top_2_indices)

# GRADED CELL 

def semantic_search_retrieve(query, top_k=5):
    """
    Retrieves the top k relevant documents for a given query using semantic search and cosine similarity.

    This function generates an embedding for the input query and compares it against pre-computed document
    embeddings using cosine similarity. The indices of the top k most similar documents are returned.

    Args:
        query (str): The search query for which relevant documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default value is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k most relevant documents in the corpus.
    """
    ### START CODE HERE ###
    # Generate the embedding for the query using the pre-trained model
    query_embedding = model.encode(query)
    
    # Calculate the cosine similarity scores between the query embedding and the pre-computed document embeddings
    similarity_scores = cosine_similarity(query_embedding, EMBEDDINGS)
    
    # Sort the similarity scores in descending order and get the indices
    similarity_indices = np.argsort(-similarity_scores)

    # Select the indices of the top k documents as a numpy array
    top_k_indices_array =similarity_indices[:top_k]

    ### END CODE HERE ###
    
    # Cast them to int 
    top_k_indices = [int(x) for x in top_k_indices_array]
    
    return top_k_indices


def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.
    """

    # Create a dictionary to store the RRF scores for each document index
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        for rank, item in enumerate(lst, start=1):
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Use rank directly instead of lst[rank]
            rrf_scores[item] += 1 / (K + rank)

    # Sort items by score (highest first)
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # Take top-k
    top_k_indices = sorted_items[:top_k]

    return top_k_indices
list1 = semantic_search_retrieve('What are the recent news about GDP?')
list2 = bm25_retrieve('What are the recent news about GDP?')
rrf_list = reciprocal_rank_fusion(list1, list2)
print(f"Semantic Search List: {list1}")
print(f"BM25 List: {list2}")
print(f"RRF List: {rrf_list}")