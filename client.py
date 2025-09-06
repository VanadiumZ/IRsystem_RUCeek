import json
import requests
import getpass
import time
from urllib.parse import urljoin

## TODO: Students need to implement the evaluate function
from search_engine import evaluate, reset_query_engine

# The TA will announce the base_url to be used after debugging is complete
base_url = 'http://123.57.79.100:8080'  # Local server address, modify according to actual situation

def input_idx():
    idx = input('idx: ')
    # maybe some restrictions
    return idx

def input_passwd():
    passwd = getpass.getpass('passwd for final submission (None for debug mode): ')
    if passwd == '':
        print('=== DEBUG MODE ===')
    return passwd

def login(idx, passwd):
    url = urljoin(base_url, 'login')
    r = requests.post(url, data={'idx': idx, 'passwd': passwd})
    r_dct = eval(r.text)
    queries = r_dct['queries']
    if r_dct['mode'] == 'illegal':
        raise ValueError('illegal password!')
    elif r_dct['mode'] == 'debug':
        print(f"queries: {queries}")
    print(f'{len(queries)} queries.')
    return queries

def send_ans(idx, passwd, urls):
    url = urljoin(base_url, 'mrr')
    r = requests.post(url, data={'idx': idx, 'passwd': passwd, 'urls': json.dumps(urls)})
    r_dct = eval(r.text)
    if r_dct['mode'] == 'illegal':
        raise ValueError('illegal password!')
    return r_dct['mode'], r_dct['mrr']


# # ===========================================

# def debug_q(queries, index, tot_urls):
#     query = queries[index]
#     print(f"degbug: query: {query}")
#     urls = evaluate(query)
#     print(f"Returned {len(urls)} results")
#     tot_urls.append(urls)
#     print_urls(tot_urls)

#     return tot_urls

# def print_urls(tot_urls):
#     for i, urls in enumerate(tot_urls):
#         print(f"query {i+1}:")
#         for url in urls:
#             print(f"  {url}")
#         print()

# # ============================================

def main():
    print("=== Information Retrieval System Evaluation Client ===")
    print(f"Connecting to server: {base_url}")
    
    # Record program start time
    start_time = time.time()
    
    idx = input_idx()
    passwd = input_passwd()
    queries = login(idx, passwd)
    print(f"Obtained {len(queries)} queries")

    reset_query_engine()

    print("\nStarting to process queries...")
    # Record query processing start time
    query_start_time = time.time()
    tot_urls = []

    for index, query in enumerate(queries):
        print(f"Processing query {index+1}/{len(queries)}")
        query_time_start = time.time()
        urls = evaluate(query)
        query_time_end = time.time()
        print(f"Returned {len(urls)} results (time: {query_time_end - query_time_start:.3f}s)")
        tot_urls.append(urls)

    # Record query processing end time
    query_end_time = time.time()
    query_total_time = query_end_time - query_start_time
    print(f"\nAll queries processed, total time: {query_total_time:.3f}s")
    print(f"Average time per query: {query_total_time/len(queries):.3f}s")

    # tot_urls = debug_q(queries, 0, tot_urls)
    
    print("\nSubmitting results to server...")
    mode, mrr = send_ans(idx, passwd, tot_urls)
    print(f'MRR@20: [{mrr}], [{mode}] mode')
    
    # Record program end time
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Runtime Statistics ===")
    print(f"Total program runtime: {total_time:.3f}s ({total_time/60:.2f}min)")
    print(f"Query processing time: {query_total_time:.3f}s ({query_total_time/60:.2f}min)")
    print(f"Other operations time: {total_time - query_total_time:.3f}s")
    
    if mrr == -1:
        print("Error: You have already submitted a test, each person can only test once!")
    elif mrr > 0:
        print(f"Test completed! Your MRR score: {mrr:.4f}")
    else:
        print("Test encountered problems, please check.")

if __name__ == '__main__':
    main()
