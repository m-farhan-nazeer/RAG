# GRADED CELL 

client = weaviate.connect_to_local(port=8079, grpc_port=50050)
collection = client.collections.get("bbc_collection")


def filter_by_metadata(metadata_property: str, 
                       values: list[str], 
                       collection: "weaviate.collections.collection.sync.Collection" , 
                       limit: int = 5) -> list:
    """
    Retrieves objects from a specified collection based on metadata filtering criteria.

    This function queries a collection within the specified client to fetch objects that match 
    certain metadata criteria. It uses a filter to find objects whose specified 'property' contains 
    any of the given 'values'. The number of objects retrieved is limited by the 'limit' parameter.

    Args:
    metadata_property (str): The name of the metadata property to filter on.
    values (List[str]): A list of values to be matched against the specified property.
    collection_name (weaviate.collections.collection.sync.Collection): The collection to query.
    limit (int, optional): The maximum number of objects to retrieve. Defaults to 5.

    Returns:
    List[Object]: A list of objects from the collection that match the filtering criteria.
    """
    ### START CODE HERE ###
    # Retrieve using collection.query.fetch_objects
    
    response = collection.query.fetch_objects(limit=limit,filters = Filter.by_property(metadata_property).contains_any(values))

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects


# Let's get an example
res = filter_by_metadata('title', ['Taylor Swift'], collection, limit = 2)
for x in res:
    print_object_properties(x)




# GRADED CELL 

def semantic_search_retrieve(query: str,
                             collection: "weaviate.collections.collection.sync.Collection" , 
                             top_k: int = 5) -> list:
    """
    Performs a semantic search on a collection and retrieves the top relevant chunks.

    This function executes a semantic search query on a specified collection to find text chunks 
    that are most relevant to the input 'query'. The search retrieves a limited number of top 
    matching objects, as specified by 'top_k'. The function returns the 'chunk' property of 
    each of the top matching objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the semantic search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    ### START CODE HERE ###
    

    # Retrieve using collection.query.near_text
    response = collection.query.near_text(query=query,limit=top_k)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects


# GRADED CELL 

def bm25_retrieve(query: str, 
                  collection: "weaviate.collections.collection.sync.Collection" , 
                  top_k: int = 5) -> list:
    """
    Performs a BM25 search on a collection and retrieves the top relevant chunks.

    This function executes a BM25-based search query on a specified collection to identify text 
    chunks that are most relevant to the provided 'query'. It retrieves a limited number of the 
    top matching objects, as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the BM25 search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    
    ### START CODE HERE ###

    # Retrieve using collection.query.bm25
    response = collection.query.bm25(query=query,limit=top_k)

    ### END CODE HERE ### 
    
    response_objects = [x.properties for x in response.objects]
    return response_objects 


# GRADED CELL 

def hybrid_retrieve(query: str, 
                    collection: "weaviate.collections.collection.sync.Collection" , 
                    alpha: float = 0.5,
                    top_k: int = 5
                   ) -> list:
    """
    Performs a hybrid search on a collection and retrieves the top relevant chunks.

    This function executes a hybrid search that combines semantic vector search and traditional 
    keyword-based search on a specified collection to find text chunks most relevant to the 
    input 'query'. The relevance of results is influenced by 'alpha', which balances the weight 
    between vector and keyword matches. It retrieves a limited number of top matching objects, 
    as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the hybrid search is performed.
    alpha (float, optional): A weighting factor that balances the contribution of semantic 
    and keyword matches. Defaults to 0.5.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    ### START CODE HERE ### 

    # Retrieve using collection.query.hybrid
    response = collection.query.hybrid(query=query,alpha=alpha,limit=top_k)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 



# GRADED CELL 

def semantic_search_with_reranking(query: str, 
                                   rerank_property: str,
                                   collection: "weaviate.collections.collection.sync.Collection" , 
                                   rerank_query: str = None,
                                   top_k: int = 5
                                   ) -> list:
    """
    Performs a semantic search and reranks the results based on a specified property.

    Args:
        query (str): The search query to perform the initial search.
        rerank_property (str): The property used for reranking the search results.
        collection (weaviate.collections.collection.sync.Collection): The collection to search within.
        rerank_query (str, optional): The query to use specifically for reranking. If not provided, 
                                      the original query is used for reranking.
        top_k (int, optional): The maximum number of top results to return. Defaults to 5.

    Returns:
        list: A list of properties from the reranked search results, where each item corresponds to 
              an object in the collection.
    """
    ### START CODE HERE ### 
    


    # Set the rerank_query to be the same as the query if rerank_query is not passed (don't change this line)
    if rerank_query is None: 
        rerank_query = query 
        
    # Define the reranker with rerank_query and rerank_property
    reranker = Rerank(prop=rerank_property,query=rerank_query)

    # Retrieve using collection.query.near_text with the appropriate parameters (do not forget the rerank!)
    response = collection.query.near_text(query=query,limit=top_k,rerank=reranker)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 


# GRADED CELL 

def semantic_search_with_reranking(query: str, 
                                   rerank_property: str,
                                   collection: "weaviate.collections.collection.sync.Collection" , 
                                   rerank_query: str = None,
                                   top_k: int = 5
                                   ) -> list:
    """
    Performs a semantic search and reranks the results based on a specified property.

    Args:
        query (str): The search query to perform the initial search.
        rerank_property (str): The property used for reranking the search results.
        collection (weaviate.collections.collection.sync.Collection): The collection to search within.
        rerank_query (str, optional): The query to use specifically for reranking. If not provided, 
                                      the original query is used for reranking.
        top_k (int, optional): The maximum number of top results to return. Defaults to 5.

    Returns:
        list: A list of properties from the reranked search results, where each item corresponds to 
              an object in the collection.
    """
    ### START CODE HERE ### 
    


    # Set the rerank_query to be the same as the query if rerank_query is not passed (don't change this line)
    if rerank_query is None: 
        rerank_query = query 
        
    # Define the reranker with rerank_query and rerank_property
    reranker = Rerank(prop=rerank_property,query=rerank_query)

    # Retrieve using collection.query.near_text with the appropriate parameters (do not forget the rerank!)
    response = collection.query.near_text(query=query,limit=top_k,rerank=reranker)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 