# GRADED CELL 

client = weaviate.connect_to_local(port=8079, grpc_port=50050)


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