from sentence_transformers import SentenceTransformer

# Load the pre-trained sentence transformer model using the method .encode
model_name =  "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(os.path.join(os.environ['MODEL_PATH'],model_name))
# To get a string embedded, just pass it to the model.
res = model.encode("RAG is awesome")
print(res.shape)
model.encode(['apple', 'car'])
words = ['apple', 'car', 'fruit', 'automobile', 'love', 'sentiment']
vectorized_words = model.encode(words)
word = 'fruit'
print(f"{word}:")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tCosine Similarity: {cosine_similarity(vectorized_word, vectorized_words[i])[0]:.4f}")
print("\n\n\n")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)]
    print(f"\t{w}:\t\tEuclidean Distance: {euclidean_distance(vectorized_word, vectorized_words[i])[0]:.4f}")


def retrieve_relevant(query, documents, metric='cosine_similarity'):
    """
    Retrieves and ranks documents based on their similarity to a given query using the specified metric.
    
    Parameters:
    query (str): The query string for which relevant documents are to be retrieved.
    documents (list of str): A list of documents to be compared against the query.
    metric (str, optional): The similarity measurement metric to be used. It supports 'cosine_similarity'
                            and 'euclidean'. Defaults to 'cosine_similarity'.
    
    Returns:
    list of tuples: A list of tuples where each tuple contains a document and its similarity or distance
                    score with respect to the query. The list is sorted based on these scores, with
                    descending order for 'cosine_similarity' and ascending order for 'euclidean'.
    """
    query_emb = model.encode(query)
    documents_emb = model.encode(documents)
    vals = []

    if metric == 'cosine_similarity':
        distances = cosine_similarity(query_emb, documents_emb)
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in descending order
        vals.sort(reverse=True, key=lambda x: x[1])
        
    elif metric == 'euclidean':
        distances = euclidean_distance(query_emb, documents_emb)
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in ascending order
        vals.sort(key=lambda x: x[1])
        
    return vals