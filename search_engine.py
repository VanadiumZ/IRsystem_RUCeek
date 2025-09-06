from query import QueryEngine

# Global QueryEngine instance - initialize only once
_global_query_engine = None

def get_query_engine():
    """
    Get global QueryEngine instance, implementing singleton pattern
    
    Advantages:
    1. Avoid repeatedly loading large data files (inverted index, document length, ID mapping, etc.)
    2. Avoid repeatedly initializing jieba dictionary and stopword list
    3. Significantly improve query response speed
    
    Returns:
        QueryEngine: Global unique QueryEngine instance
    """
    global _global_query_engine
    
    if _global_query_engine is None:
        print("Initializing QueryEngine (executed only once)...")
        # Use the same path configuration as app_jieba.py
        DATA_DIR = "./data"
        TXT_DIR = "./final_txt"
        
        _global_query_engine = QueryEngine(thu=False, data_dir=DATA_DIR, txt_dir=TXT_DIR)
        print("QueryEngine initialization completed!")
    
    return _global_query_engine

def evaluate(query: str) -> list:
    '''
    Optimized evaluate function - using global QueryEngine instance
    
    Performance optimization description:
    - Original version: Creates new QueryEngine instance for each query, causing repeated data file loading
    - Optimized version: Uses global singleton, initializes only once for 20 queries, greatly improves performance
    
    Parameters: query, string type, represents the query
    Return value: url_list, a list of 20 urls
    '''
    # Get global QueryEngine instance (initialize on first call)

    # reset_query_engine()
    qp = get_query_engine()
    
    # Execute query
    url_list = qp.retrieval_url(query)
    # url_list = qp.retrieval_by_enhanced_score(query)

    return url_list

def reset_query_engine():
    """
    Reset global QueryEngine instance (for testing or configuration changes)
    """
    global _global_query_engine
    _global_query_engine = None
    print("QueryEngine has been reset")

