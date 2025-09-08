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