from collections import defaultdict, Counter
import jieba
import numpy as np
import dill
import os
import math
import thulac
import re
from urllib.parse import urlparse
from functools import lru_cache

class AdvancedQueryParser:
    """
    Advanced Query Parser - Optimized Version
    Supports parsing various advanced query syntax including URL filtering, exact phrase matching, site restrictions, etc.
    """
    
    def __init__(self):
        # Compile regular expressions for better performance
        self.inurl_pattern = re.compile(r'inurl:([^\s]+)', re.IGNORECASE)
        self.exact_phrase_pattern = re.compile(r'"([^"]+)"')
        self.site_pattern = re.compile(r'site:([^\s]+)', re.IGNORECASE)
        self.site_or_pattern = re.compile(r'site:([^\s]+)\s+OR\s+site:([^\s]+)', re.IGNORECASE)
    
    @lru_cache(maxsize=1000)
    def parse_query(self, query):
        """
        Parse query string and extract advanced syntax
        Use LRU cache to avoid repeatedly parsing the same query
        """
        # Extract inurl: keywords
        inurl_matches = self.inurl_pattern.findall(query)
        inurl_terms = [match for match in inurl_matches]
        
        # Extract exact phrases
        exact_phrases = self.exact_phrase_pattern.findall(query)
        
        # Extract multi-site OR queries
        site_or_matches = self.site_or_pattern.findall(query)
        site_or_filters = [list(match) for match in site_or_matches]
        
        # Extract single site filters (excluding those already in OR)
        all_site_matches = self.site_pattern.findall(query)
        or_sites = set()
        for pair in site_or_filters:
            or_sites.update(pair)
        site_filters = [site for site in all_site_matches if site not in or_sites]
        
        # Separate directory filters (site: containing paths)
        directory_filters = []
        regular_site_filters = []
        
        for site in site_filters:
            if '/' in site and not site.startswith('http'):
                directory_filters.append('http://' + site)
            elif site.startswith('http'):
                directory_filters.append(site)
            else:
                regular_site_filters.append(site)
        
        # Build base query (remove all advanced syntax)
        base_query = query
        base_query = self.inurl_pattern.sub('', base_query)
        base_query = self.exact_phrase_pattern.sub('', base_query)
        base_query = self.site_pattern.sub('', base_query)
        base_query = re.sub(r'\s+OR\s+', ' ', base_query, flags=re.IGNORECASE)
        base_query = re.sub(r'\s+', ' ', base_query).strip()
        
        return {
            'base_query': base_query,
            'inurl_terms': inurl_terms,
            'exact_phrases': exact_phrases,
            'site_filters': regular_site_filters,
            'site_or_filters': site_or_filters,
            'directory_filters': directory_filters
        }
    
    @lru_cache(maxsize=5000)
    def check_url_filter(self, url, inurl_terms_tuple):
        """
        Check if URL matches inurl filter conditions
        Use cache and tuple parameters to support caching
        """
        if not inurl_terms_tuple:
            return True
            
        url_lower = url.lower()
        parsed_url = urlparse(url_lower)
        path = parsed_url.path
        
        for term in inurl_terms_tuple:
            if term.lower() in path:
                return True
        
        return False
    
    def check_exact_phrase(self, title, content, exact_phrases):
        """
        Check if title and content contain exact phrases
        """
        if not exact_phrases:
            return True, 0
            
        title_lower = title.lower()
        content_lower = content.lower()
        
        match_count = 0
        for phrase in exact_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in title_lower or phrase_lower in content_lower:
                match_count += 1
        
        return match_count > 0, match_count
    
    @lru_cache(maxsize=5000)
    def check_site_filter(self, url, site_filters_tuple, site_or_filters_tuple):
        """
        Check if URL matches site filter conditions
        Use cache to optimize performance
        """
        if not site_filters_tuple and not site_or_filters_tuple:
            return True
            
        parsed_url = urlparse(url.lower())
        host = parsed_url.netloc
        
        # Check single site filters
        for site in site_filters_tuple:
            site_lower = site.lower()
            if site_lower.startswith('.'):
                if host.endswith(site_lower):
                    return True
            else:
                if host.endswith(site_lower):
                    return True
        
        # Check multi-site OR filters
        for site_pair in site_or_filters_tuple:
            for site in site_pair:
                site_lower = site.lower()
                if site_lower.startswith('.'):
                    if host.endswith(site_lower):
                        return True
                else:
                    if host.endswith(site_lower):
                        return True
        
        return len(site_filters_tuple) == 0 and len(site_or_filters_tuple) == 0
    
    @lru_cache(maxsize=5000)
    def check_directory_filter(self, url, directory_filters_tuple):
        """
        Check if URL matches directory filter conditions
        Use cache to optimize performance
        """
        if not directory_filters_tuple:
            return True
            
        url_lower = url.lower()
        
        for prefix in directory_filters_tuple:
            prefix_lower = prefix.lower()
            if not prefix_lower.endswith('/'):
                prefix_lower += '/'
            
            if url_lower.startswith(prefix_lower):
                return True
        
        return False


class QueryEngineOptimized:
    """
    Optimized version of query engine
    Main optimizations:
    1. Document content caching
    2. URL parsing cache
    3. Reduce unnecessary file I/O
    4. Optimize filter logic order
    """

    def __init__(self, k1=1.5, b=0.75, thu=False, data_dir="./data", txt_dir="./final_txt"):
        '''
        initialize the QueryEngine, load necessary data files
        '''
        self.thu = None
        self.txt_dir = txt_dir
        self.query_parser = AdvancedQueryParser()
        
        # Document content cache
        self._content_cache = {}
        self._url_cache = {}
        
        # Choose different data subdirectories based on thu parameter
        if thu:
            if thulac is None:
                raise ImportError("thulac module is required when thu=True")
            
            data_subdir = os.path.join(data_dir, 'thu')
            jieba_dir = os.path.join(data_dir, 'jieba')
            
            jieba.load_userdict(os.path.join(jieba_dir, 'jieba_dict.txt'))
            with open(os.path.join(jieba_dir, 'baidu_stopwords.txt'), 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
                
            with open(os.path.join(data_subdir, 'inverted_index_thu.pkl'), 'rb') as fin:
                restore_inverted_index = dill.load(fin)
            self.inverted_index = dict(restore_inverted_index)

            with open(os.path.join(data_subdir, 'doc_length_thu.pkl'), 'rb') as fin:
                restore_doc_length = dill.load(fin)
            self.doc_length = dict(restore_doc_length)

            with open(os.path.join(data_subdir, 'doc_id_pair_thu.pkl'), 'rb') as fin:
                restore_id_pair = dill.load(fin)
            self.id_pair = dict(restore_id_pair)
            
            self.thu = thulac.thulac(seg_only=True)
        else:
            data_subdir = os.path.join(data_dir, 'jieba')
            
            jieba.load_userdict(os.path.join(data_subdir, 'jieba_dict.txt'))
            with open(os.path.join(data_subdir, 'baidu_stopwords.txt'), 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
                
            with open(os.path.join(data_subdir, 'inverted_index.pkl'), 'rb') as fin:
                restore_inverted_index = dill.load(fin)
            self.inverted_index = dict(restore_inverted_index)

            with open(os.path.join(data_subdir, 'doc_length.pkl'), 'rb') as fin:
                restore_doc_length = dill.load(fin)
            self.doc_length = dict(restore_doc_length)

            with open(os.path.join(data_subdir, 'doc_id_pair.pkl'), 'rb') as fin:
                restore_id_pair = dill.load(fin)
            self.id_pair = dict(restore_id_pair)
        
        self.k1 = k1
        self.b = b
        self.N = len(self.doc_length)
        self.avgdl = sum(self.doc_length.values()) / self.N
    
    def get_url(self, file):
        """Get the URL corresponding to the file, optimized with cache"""
        if file in self._url_cache:
            return self._url_cache[file]
            
        # Actual URL retrieval logic (read from file)
        try:
            file_path = os.path.join(self.txt_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith("URL: "):
                    url = first_line[5:]  # Remove 'URL: ' prefix
                else:
                    url = first_line
                self._url_cache[file] = url
                return url
        except Exception:
            # If reading fails, return default URL
            url = f"http://example.com/{file}"
            self._url_cache[file] = url
            return url
    
    def get_posting_list(self, inverted_index, query_term):
        try:
            return inverted_index[query_term][1:]
        except KeyError:
            return []
    
    def _tokenize_query_with_thulac(self, query: str):
        """
        align with thulac processor
        """
        seg_text = self.thu.cut(query, text=True)
        tokens = [t for t in seg_text.split() if len(t) > 1 and t not in self.stop_list]
        return tokens
    
    def _compute_ordered_window_tf(self, positions1, positions2, window_size=8):
        """
        Calculate term pair frequency within ordered windows
        
        Args:
            positions1: Position list of the first term
            positions2: Position list of the second term  
            window_size: Window size
            
        Returns:
            int: Term pair frequency within ordered windows
        """
        count = 0
        i, j = 0, 0
        
        while i < len(positions1) and j < len(positions2):
            pos1, pos2 = positions1[i], positions2[j]
            
            # Check if within ordered window (pos1 < pos2 and distance <= window_size)
            if pos1 < pos2 and pos2 - pos1 <= window_size:
                count += 1
                i += 1
            elif pos1 < pos2:
                i += 1
            else:
                j += 1
                
        return count
    
    def _compute_unordered_window_tf(self, positions1, positions2, window_size=8):
        """
        Calculate term pair frequency within unordered windows
        
        Args:
            positions1: Position list of the first term
            positions2: Position list of the second term
            window_size: Window size
            
        Returns:
            int: Term pair frequency within unordered windows
        """
        count = 0
        
        for pos1 in positions1:
            for pos2 in positions2:
                # Check if within unordered window (distance <= window_size)
                if abs(pos1 - pos2) <= window_size:
                    count += 1
                    
        return count
    
    def _compute_window_features(self, query_terms, window_size=8):
        """
        Calculate window features of query terms
        
        Args:
            query_terms: List of query terms
            window_size: Window size
            
        Returns:
            tuple: (ordered_features, unordered_features)
        """
        ordered_features = defaultdict(lambda: defaultdict(int))
        unordered_features = defaultdict(lambda: defaultdict(int))
        
        # Generate all term pair combinations
        for i in range(len(query_terms)):
            for j in range(i + 1, len(query_terms)):
                term1, term2 = query_terms[i], query_terms[j]
                
                # Get inverted lists of both terms
                postings1 = self.get_posting_list(self.inverted_index, term1)
                postings2 = self.get_posting_list(self.inverted_index, term2)
                
                # Organize position information by document ID
                doc_positions1 = defaultdict(list)
                doc_positions2 = defaultdict(list)
                
                for posting in postings1:
                    doc_positions1[posting.docid] = posting.positions
                    
                for posting in postings2:
                    doc_positions2[posting.docid] = posting.positions
                
                # Calculate window features in common documents
                common_docs = set(doc_positions1.keys()) & set(doc_positions2.keys())
                
                for docid in common_docs:
                    positions1 = doc_positions1[docid]
                    positions2 = doc_positions2[docid]
                    
                    # Ordered window features (term1 < term2)
                    ordered_tf = self._compute_ordered_window_tf(positions1, positions2, window_size)
                    if ordered_tf > 0:
                        ordered_features[(term1, term2)][docid] = ordered_tf
                    
                    # Unordered window features
                    unordered_tf = self._compute_unordered_window_tf(positions1, positions2, window_size)
                    if unordered_tf > 0:
                        unordered_features[(term1, term2)][docid] = unordered_tf
        
        return ordered_features, unordered_features
    
    def enhanced_bm25_scores(self, inverted_index, doc_length, query, N, k=3, k1=1.3, b=0.75, 
                           lambda_t=0.7, lambda_o=0.2, lambda_u=0.1, window_size=8, thu=False):
        """
        Enhanced BM25 scoring based on SDM model
        
        Args:
            inverted_index: Inverted index
            doc_length: Document length dictionary
            query: Query string
            N: Total number of documents
            k: Number of results to return
            k1, b: BM25 parameters
            lambda_t, lambda_o, lambda_u: SDM model weight parameters
            window_size: Window size
            thu: Whether to use Thulac tokenization
            
        Returns:
            List[(docid, score)]: Sorted list of document IDs and scores
        """
        scores = defaultdict(float)
        
        # 1. Tokenization
        if thu:
            query_terms = [term for term in self._tokenize_query_with_thulac(query)]
        else:
            query_terms = [term for term in jieba.cut(query) if len(term.strip()) > 0]
        
        if not query_terms:
            return []
        
        # 2. Unigram features (traditional BM25)
        query_term_counts = Counter(query_terms)
        for term, qtf in query_term_counts.items():
            postings_list = self.get_posting_list(inverted_index, term)
            df = len(postings_list)
            if df == 0:
                continue
                
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1)
            
            for posting in postings_list:
                tf = posting.tf
                docid = posting.docid
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length[docid] / self.avgdl)))
                scores[docid] += lambda_t * bm25_score
        
        # 3. Calculate window features (only when there are multiple query terms)
        if len(query_terms) > 1:
            ordered_features, unordered_features = self._compute_window_features(query_terms, window_size)
            
            # 4. Ordered Window features
            for (term1, term2), doc_tfs in ordered_features.items():
                # Calculate document frequency for this term pair
                df_od = len(doc_tfs)
                if df_od == 0:
                    continue
                    
                idf_od = math.log(((N - df_od + 0.5) / (df_od + 0.5)) + 1)
                
                for docid, tf_od in doc_tfs.items():
                    # Use BM25-style scoring
                    bm25_od = idf_od * ((tf_od * (k1 + 1)) / (tf_od + k1 * (1 - b + b * doc_length[docid] / self.avgdl)))
                    scores[docid] += lambda_o * math.log(bm25_od + 1)
            
            # 5. Unordered Window features  
            for (term1, term2), doc_tfs in unordered_features.items():
                # 计算该词对的文档频率
                df_uw = len(doc_tfs)
                if df_uw == 0:
                    continue
                    
                idf_uw = math.log(((N - df_uw + 0.5) / (df_uw + 0.5)) + 1)
                
                for docid, tf_uw in doc_tfs.items():
                    # 使用BM25风格的评分
                    bm25_uw = idf_uw * ((tf_uw * (k1 + 1)) / (tf_uw + k1 * (1 - b + b * doc_length[docid] / self.avgdl)))
                    scores[docid] += lambda_u * math.log(bm25_uw + 1)
        
        # Sort and return results
        results = [(docid, score) for docid, score in scores.items()]
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    def retrieval_by_score(self, inverted_index, id_pair, query, k=3):
        """Original score-based retrieval method"""
        if not query.strip():
            return []
            
        # Use enhanced_bm25_scores for retrieval
        scores = self.enhanced_bm25_scores(
            inverted_index, self.doc_length, query, self.N, k=k,
            k1=self.k1, b=self.b, thu=(self.thu is not None)
        )
        
        # Convert to (filename, score) format
        results = []
        for doc_id, score in scores:
            if doc_id in id_pair:
                filename = id_pair[doc_id]
                results.append((filename, score))
        
        return results[:k]
    
    def retrieval_url_optimized(self, query, k=20):
        """
        Optimized version of advanced search method
        Main optimizations:
        1. Early lightweight filtering
        2. Lazy loading of document content
        3. Caching mechanism
        4. Optimized filtering order
        """
        if not query:
            return []
            
        # Parse advanced query syntax (using cache)
        parsed_query = self.query_parser.parse_query(query)
        base_query = parsed_query['base_query']
        
        # Convert to tuples to support caching
        inurl_terms_tuple = tuple(parsed_query['inurl_terms'])
        site_filters_tuple = tuple(parsed_query['site_filters'])
        site_or_filters_tuple = tuple(tuple(pair) for pair in parsed_query['site_or_filters'])
        directory_filters_tuple = tuple(parsed_query['directory_filters'])
        
        # Get candidate results
        if not base_query.strip() and any([
            parsed_query['inurl_terms'],
            parsed_query['exact_phrases'], 
            parsed_query['site_filters'],
            parsed_query['site_or_filters'],
            parsed_query['directory_filters']
        ]):
            # If only filter conditions exist, get all documents
            results = [(filename, 1.0) for filename in self.id_pair.values()]
        else:
            # Use base query for retrieval
            results = self.retrieval_by_score(self.inverted_index, self.id_pair, base_query, k=k*2)
        
        # Optimization: sort by filtering cost, execute lightweight filtering first
        filtered_results = []
        
        for filename, score in results:
            url = self.get_url(filename)
            
            # 1. URL filtering (lightweight, using cache)
            if not self.query_parser.check_url_filter(url, inurl_terms_tuple):
                continue
                
            # 2. Site filtering (lightweight, using cache)
            if not self.query_parser.check_site_filter(url, site_filters_tuple, site_or_filters_tuple):
                continue
                
            # 3. Directory filtering (lightweight, using cache)
            if not self.query_parser.check_directory_filter(url, directory_filters_tuple):
                continue
            
            # 4. Exact phrase filtering (heavyweight, execute last)
            if parsed_query['exact_phrases']:
                # Lazy load document content
                title, content = self._load_document_content_cached(filename)
                phrase_match, phrase_count = self.query_parser.check_exact_phrase(
                    title, content, parsed_query['exact_phrases']
                )
                
                if not phrase_match:
                    continue
                    
                # Exact phrase bonus
                score += phrase_count * 2.0
            
            # URL hit bonus
            if parsed_query['inurl_terms']:
                url_lower = url.lower()
                parsed_url = urlparse(url_lower)
                path = parsed_url.path
                
                for term in parsed_query['inurl_terms']:
                    if term.lower() in path:
                        score += 0.5
            
            filtered_results.append((filename, score))
            
            # Early termination: if we have enough results, we can stop early
            if len(filtered_results) >= k * 2:
                break
        
        # Re-sort by score
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return URLs of top k results
        final_results = filtered_results[:k]
        url_list = [self.get_url(result[0]) for result in final_results]
        
        return url_list
    
    def _load_document_content_cached(self, filename):
        """
        Cached version of document content loading
        Avoid repeatedly reading the same file
        """
        if filename in self._content_cache:
            return self._content_cache[filename]
            
        try:
            file_path = os.path.join(self.txt_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n', 1)
            title = lines[0] if lines else ''
            body = lines[1] if len(lines) > 1 else ''
            
            # Cache results (limit cache size)
            if len(self._content_cache) < 1000:
                self._content_cache[filename] = (title, body)
            
            return title, body
        except Exception as e:
            return '', ''
    
    def clear_cache(self):
        """Clear cache"""
        self._content_cache.clear()
        self._url_cache.clear()
        # Clear LRU cache
        self.query_parser.parse_query.cache_clear()
        self.query_parser.check_url_filter.cache_clear()
        self.query_parser.check_site_filter.cache_clear()
        self.query_parser.check_directory_filter.cache_clear()