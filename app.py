# app.py (Jieba + Dill Version)
import time
import re
from html import escape
from functools import lru_cache
from flask import Flask, render_template, request
from query import QueryEngine  # Import QueryEngine class
from query_optimized import QueryEngineOptimized  # Import optimized version of QueryEngine class
from smart_snippet import SmartSnippetGenerator  # Import smart snippet generator

# --- 1. Global Configuration and Resource Loading ---
DATA_DIR = "./data"
TXT_DIR = "./final_txt"

print("Initializing Optimized QueryEngine...")
# Use optimized version of QueryEngine class to handle search, passing path parameters
query_engine = QueryEngineOptimized(k1=1.5, b=0.75, thu=False, data_dir=DATA_DIR, txt_dir=TXT_DIR)
# Keep original version as backup
query_engine_original = QueryEngine(k1=1.5, b=0.75, thu=False, data_dir=DATA_DIR, txt_dir=TXT_DIR)

# Get necessary data from QueryEngine
inverted_index = query_engine.inverted_index
doc_id_pair = query_engine.id_pair
id_doc_pair = {v: k for k, v in doc_id_pair.items()}
doc_length = query_engine.doc_length

# Initialize smart snippet generator
smart_snippet_gen = SmartSnippetGenerator(query_engine.stop_list, inverted_index)

# --- 2. Helper Functions ---
import os

def get_txt_path_by_filename(filename: str) -> str:
    """Get full path by filename"""
    if filename: return os.path.join(TXT_DIR, filename)
    return None

def get_txt_path(doc_id: int) -> str:
    """Get full path by doc_id (maintain compatibility)"""
    filename = id_doc_pair.get(doc_id)
    if filename: return os.path.join(TXT_DIR, filename)
    return None

@lru_cache(maxsize=1024)
def extract_metadata(txt_path: str) -> dict:
    try:
        with open(txt_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        # Extract URL, remove 'URL: ' prefix
        url_line = lines[0].strip() if lines else ""
        if url_line.startswith("URL: "):
            url = url_line[5:]  # Remove 'URL: ' prefix
        else:
            url = url_line
        
        body = " ".join([ln.strip() for ln in lines[1:] if ln.strip()])
        title = "Untitled"
        for line in lines[1:]:
            clean_line = line.strip()
            if clean_line:
                title = clean_line[:60] + "..." if len(clean_line) > 60 else clean_line
                break
        return {"url": url, "body": body, "title": title}
    except (IOError, IndexError):
        return {"url": "", "body": "", "title": "Error Reading File"}

def build_snippet_html(body_text: str, query: str, window_chars=200) -> str:
    """Use smart snippet generator to generate high-quality snippets"""
    return smart_snippet_gen.generate_smart_snippet(body_text, query, window_chars)

# --- 3. Initialize Flask Application ---
app = Flask(__name__)
print("Flask App with Jieba ready.")

# --- 4. Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    t0 = time.time()
    query = request.args.get('q', '').strip()
    try:
        page = int(request.args.get('page', 1))
    except (ValueError, TypeError):
        page = 1
    page_size = 10
    
    results_list = []
    total_results = 0

    if query:
        # Detect if contains advanced search syntax
        has_advanced_syntax = any([
            'inurl:' in query.lower(),
            '"' in query,
            'site:' in query.lower(),
            ' OR ' in query.upper()
        ])
        
        if has_advanced_syntax:
            # Use optimized version of advanced search functionality
            try:
                url_results = query_engine.retrieval_url_optimized(query, k=100)
                # Convert URL results to (filename, score) format
                ranked_results = []
                for url in url_results:
                    # Find corresponding filename from URL
                    for filename in query_engine.id_pair.values():
                        file_url = query_engine.get_url(filename)
                        if file_url == url:
                            # Assign base score for advanced search results
                            score = 1.0 - len(ranked_results) * 0.01  # Decreasing score
                            ranked_results.append((filename, score))
                            break
            except Exception as e:
                print(f"Optimized search failed, fallback to original version: {e}")
                # Fallback to original version of advanced search
                url_results = query_engine_original.retrieval_url(query, k=100)
                ranked_results = []
                for url in url_results:
                    for filename in query_engine_original.id_pair.values():
                        file_url = query_engine_original.get_url(filename)
                        if file_url == url:
                            score = 1.0 - len(ranked_results) * 0.01
                            ranked_results.append((filename, score))
                            break
        else:
            # Use traditional search
            ranked_results = query_engine.retrieval_by_score(
                query_engine.inverted_index, 
                query_engine.id_pair, 
                query, 
                k=100  # Get more results for pagination
            )
        
        total_results = len(ranked_results)

        start_index = (page - 1) * page_size
        paginated_results = ranked_results[start_index : start_index + page_size]

        for filename, score in paginated_results:
            txt_path = get_txt_path_by_filename(filename)
            if not txt_path: continue
            
            metadata = extract_metadata(txt_path)
            snippet_html = build_snippet_html(metadata["body"], query)
            
            results_list.append({
                "filename": filename, "score": round(score, 4), "title": metadata["title"],
                "url": metadata["url"], "snippet_html": snippet_html,
            })

    time_ms = round((time.time() - t0) * 1000, 2)
    
    return render_template(
        'search_results.html',
        query=query,
        results=results_list,
        total_results=total_results,
        time_ms=time_ms,
        page=page,
        page_size=page_size
    )

if __name__ == '__main__':
    app.run(debug=True)