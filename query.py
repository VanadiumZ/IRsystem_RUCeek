from collections import defaultdict, Counter
import jieba
import numpy as np
import dill
import os
import math
import thulac
import re
from urllib.parse import urlparse


class AdvancedQueryParser:
    """
    Advanced query parser supporting the following syntax:
    - inurl:keyword - URL hit filtering
    - "exact phrase" - Exact phrase matching
    - site:domain - Site restriction
    - site:A OR site:B - Multi-site OR query
    - site:http://domain/path/ - Directory restriction query
    """
    
    def __init__(self):
        # Compile regular expressions for better performance
        self.inurl_pattern = re.compile(r'inurl:([^\s]+)', re.IGNORECASE)
        self.exact_phrase_pattern = re.compile(r'"([^"]+)"')
        self.site_pattern = re.compile(r'site:([^\s]+)', re.IGNORECASE)
        self.site_or_pattern = re.compile(r'site:([^\s]+)\s+OR\s+site:([^\s]+)', re.IGNORECASE)
    
    def parse_query(self, query):
        """
        Parse query string and extract advanced query syntax
        
        Returns:
            dict: Dictionary containing parsing results
            {
                'base_query': str,  # Base query after removing advanced syntax
                'inurl_terms': list,  # List of inurl keywords
                'exact_phrases': list,  # List of exact phrases
                'site_filters': list,  # List of site filters
                'site_or_filters': list,  # List of multi-site OR filters
                'directory_filters': list  # List of directory filters
            }
        """
        result = {
            'base_query': query,
            'inurl_terms': [],
            'exact_phrases': [],
            'site_filters': [],
            'site_or_filters': [],
            'directory_filters': []
        }
        
        # Extract inurl keywords
        inurl_matches = self.inurl_pattern.findall(query)
        result['inurl_terms'] = inurl_matches
        
        # Extract exact phrases
        phrase_matches = self.exact_phrase_pattern.findall(query)
        result['exact_phrases'] = phrase_matches
        
        # Extract multi-site OR queries
        site_or_matches = self.site_or_pattern.findall(query)
        for match in site_or_matches:
            result['site_or_filters'].append([match[0], match[1]])
        
        # Extract single site filters (excluding already matched OR queries)
        remaining_query = self.site_or_pattern.sub('', query)
        site_matches = self.site_pattern.findall(remaining_query)
        
        for site in site_matches:
            if site.startswith('http://') or site.startswith('https://'):
                # URLs with protocol are treated as directory restrictions
                result['directory_filters'].append(site)
            else:
                # Regular domain filtering
                result['site_filters'].append(site)
        
        # Generate base query (remove all advanced syntax)
        base_query = query
        base_query = self.inurl_pattern.sub('', base_query)
        base_query = self.exact_phrase_pattern.sub('', base_query)
        base_query = self.site_pattern.sub('', base_query)
        base_query = re.sub(r'\s+OR\s+', ' ', base_query, flags=re.IGNORECASE)
        base_query = re.sub(r'\s+', ' ', base_query).strip()
        
        result['base_query'] = base_query
        
        return result
    
    def check_url_filter(self, url, inurl_terms):
        """
        Check if URL matches inurl filter conditions
        
        Args:
            url: URL to check
            inurl_terms: List of inurl keywords
            
        Returns:
            bool: Whether it matches
        """
        if not inurl_terms:
            return True
            
        url_lower = url.lower()
        parsed_url = urlparse(url_lower)
        path = parsed_url.path
        
        for term in inurl_terms:
            if term.lower() in path:
                return True
        return False
    
    def check_exact_phrase(self, title, content, exact_phrases):
        """
        Check if title and content contain exact phrases
        
        Args:
            title: Document title
            content: Document content
            exact_phrases: List of exact phrases
            
        Returns:
            tuple: (whether matched, number of matched phrases)
        """
        if not exact_phrases:
            return True, 0
            
        # Whitespace normalization
        title_normalized = re.sub(r'\s+', ' ', title.strip()) if title else ''
        content_normalized = re.sub(r'\s+', ' ', content.strip()) if content else ''
        
        matched_count = 0
        for phrase in exact_phrases:
            phrase_normalized = re.sub(r'\s+', ' ', phrase.strip())
            if (phrase_normalized in title_normalized or 
                phrase_normalized in content_normalized):
                matched_count += 1
        
        return matched_count > 0, matched_count
    
    def check_site_filter(self, url, site_filters, site_or_filters):
        """
        Check if URL matches site filter conditions
        
        Args:
            url: URL to check
            site_filters: List of single site filters
            site_or_filters: List of multi-site OR filters
            
        Returns:
            bool: Whether it matches
        """
        if not site_filters and not site_or_filters:
            return True
            
        parsed_url = urlparse(url.lower())
        host = parsed_url.netloc
        
        # Check single site filters
        for site in site_filters:
            site_lower = site.lower()
            if site_lower.startswith('.'):
                # Suffix matching, e.g. .edu
                if host.endswith(site_lower):
                    return True
            else:
                # Domain matching
                if host.endswith(site_lower):
                    return True
        
        # Check multi-site OR filters
        for site_pair in site_or_filters:
            for site in site_pair:
                site_lower = site.lower()
                if site_lower.startswith('.'):
                    if host.endswith(site_lower):
                        return True
                else:
                    if host.endswith(site_lower):
                        return True
        
        return len(site_filters) == 0 and len(site_or_filters) == 0
    
    def check_directory_filter(self, url, directory_filters):
        """
        Check if URL matches directory filter conditions
        
        Args:
            url: URL to check
            directory_filters: List of directory filters
            
        Returns:
            bool: Whether it matches
        """
        if not directory_filters:
            return True
            
        url_lower = url.lower()
        
        for prefix in directory_filters:
            prefix_lower = prefix.lower()
            # Ensure ending with /, avoid false matches
            if not prefix_lower.endswith('/'):
                prefix_lower += '/'
            
            if url_lower.startswith(prefix_lower):
                return True
        
        return False


class QueryEngine:

    def __init__(self, k1=1.5, b=0.75, thu=False, data_dir="./data", txt_dir="./final_txt"):
        '''
        initialize the QueryEngine, load necessary data files
        Args:
            k1, b: BM25 parameters
            thu: whether to use THU segmentation
            data_dir: base directory for data files
            txt_dir: directory for text files
        '''
        self.thu = None
        self.txt_dir = txt_dir
        
        # Choose different data subdirectories based on thu parameter
        if thu:
            if thulac is None:
                raise ImportError("thulac module is required when thu=True")
            
            data_subdir = os.path.join(data_dir, 'thu')
            jieba_dir = os.path.join(data_dir, 'jieba')  # jieba词典仍在jieba目录
            
            # Load jieba user dictionary and stopwords
            jieba.load_userdict(os.path.join(jieba_dir, 'jieba_dict.txt'))
            with open(os.path.join(jieba_dir, 'baidu_stopwords.txt'), 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
                
            # Load THU data files
            with open(os.path.join(data_subdir, 'inverted_index_thu.pkl'), 'rb') as fin:
                restore_inverted_index = dill.load(fin)
            self.inverted_index = dict(restore_inverted_index)

            with open(os.path.join(data_subdir, 'doc_length_thu.pkl'), 'rb') as fin:
                restore_doc_length = dill.load(fin)
            self.doc_length = dict(restore_doc_length)

            with open(os.path.join(data_subdir, 'doc_id_pair_thu.pkl'), 'rb') as fin:
                restore_id_pair = dill.load(fin)
            self.id_pair = dict(restore_id_pair)
            
            # Initialize thulac
            self.thu = thulac.thulac(
                        seg_only=True,
                        user_dict=os.path.join(jieba_dir, 'jieba_dict.txt'),
                        T2S=False,
                        filt=False)
        else:
            data_subdir = os.path.join(data_dir, 'jieba')
            
            # Load jieba user dictionary and stopwords
            jieba.load_userdict(os.path.join(data_subdir, 'jieba_dict.txt'))
            with open(os.path.join(data_subdir, 'baidu_stopwords.txt'), 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
                
            # Load jieba data files
            with open(os.path.join(data_subdir, 'inverted_index.pkl'), 'rb') as fin:
                restore_inverted_index = dill.load(fin)
            self.inverted_index = dict(restore_inverted_index)

            with open(os.path.join(data_subdir, 'doc_length.pkl'), 'rb') as fin:
                restore_doc_length = dill.load(fin)
            self.doc_length = dict(restore_doc_length)

            with open(os.path.join(data_subdir, 'doc_id_pair.pkl'), 'rb') as fin:
                restore_id_pair = dill.load(fin)
            self.id_pair = dict(restore_id_pair)

        self.doc_lengths = [doclength for doclength in self.doc_length.values()]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.N = len(self.doc_length)
        self.k1 = k1
        self.b = b
        
        # Initialize advanced query parser
        self.query_parser = AdvancedQueryParser()
        self.thucut = thu
        

    def get_url(self, file):
        with open(os.path.join(self.txt_dir, file), 'r', encoding='utf-8') as f:
            url = (f.readlines()[0].split())[-1]
        return url
    
    def get_posting_list(self, inverted_index, query_term):
        try:
            return inverted_index[query_term][1:]
        except KeyError:
            return []
        
    def cosine_scores(self, inverted_index, doc_length, query, k=3):
        scores = defaultdict(lambda: 0.0)
        query_terms = Counter(term for term in jieba.cut(query))
        for q in query_terms:
            log_func = np.vectorize(lambda x: 1.0 + np.log10(x) if x > 0 else 0.0)
            w_tq = log_func(query_terms[q])
            postings_list = self.get_posting_list(inverted_index, q)
            for posting in postings_list:
                w_td = log_func(posting.tf)
                scores[posting.docid] += w_td * w_tq
        results = [(docid, score / doc_length[docid]) for docid, score in scores.items()]
        results.sort(key=lambda x: -x[1])
        return results[0:k]
    
    def _tokenize_query_with_thulac(self, query: str):
        """
        align with thulac processor
        """
        seg_text = self.thu.cut(query, text=True)
        tokens = [t for t in seg_text.split() if len(t) > 1 and t not in self.stop_list]
        return tokens
    
    def bm25_scores(self, inverted_index, doc_length, query, N, k=3, k1=1.3, b=0.75, thu=False):
        scores =defaultdict(lambda: 0.0)
        if thu:
            query_terms =Counter(term for term in self._tokenize_query_with_thulac(query))
        else:
            query_terms =Counter(term for term in jieba.cut(query))

        for q in query_terms:
            postings_list = self.get_posting_list(inverted_index, q)
            df = len(postings_list) # document frequency
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1)
            for posting in postings_list:
                tf = posting.tf
                docid = posting.docid
                scores[docid] += idf * (
                    (tf * (k1 +1)) / (tf + k1 * (1 - b + b * doc_length[docid]  / self.avg_length))
                )
        results = [(docid, score) for docid, score in scores.items()]
        results.sort(key=lambda x: -x[1])
        return results[0:k]
    
    def rank_by_tfidf(self, inverted_index, N, query, topk=3, use_cosine=True, stopwords=None):
        """
        Rank with tf-idf
        - TF: 1 + log(tf)
        - IDF: log(1 + N/(df+1))  # avoid df = 0 / infinite
        - similarity: dot product or cosine
        
        Params:
            inverted_index: term -> postings
            N: number of documents
            query: string of query
            topk: return top k results
            use_cosine: whether to use cosine similarity
        Returns:
            List[(docid, score)] sorted by score in descending order
        """
        # Split query and compute tf
        tokens = [t for t in jieba.cut(query) if not stopwords or t not in stopwords]
        q_tf = Counter(tokens)
        
        dot = defaultdict(float)
        doc_norm_sq = defaultdict(float)
        q_norm_sq = 0.0
        
        for term, tfq in q_tf.items():
            postings = self.get_posting_list(inverted_index, term)
            df = len(postings)
            if df == 0:
                continue
            
            # Log tf + idf
            idf = math.log(1.0 + N / (df + 1.0))
            w_tq = (1.0 + math.log(tfq)) * idf
            q_norm_sq += w_tq * w_tq
            
            for p in postings:
                w_td = (1.0 + math.log(p.tf)) * idf
                dot[p.docid] += w_td * w_tq
                if use_cosine:
                    doc_norm_sq[p.docid] += w_td * w_td
        
        if not dot:
            return []
        
        if use_cosine:
            q_norm = math.sqrt(q_norm_sq) or 1.0
            scores = {doc: dot[doc] / ( (math.sqrt(doc_norm_sq[doc]) or 1.0) * q_norm + 1e-12 )
                    for doc in dot}
        else:
            scores = dot
        
        return sorted(scores.items(), key=lambda x: -x[1])[:topk]
    
    def retrieval_by_score(self, inverted_index, id_pair, query, k=3):
        # top_scores = self.bm25_scores(inverted_index, self.doc_length, query, self.N, k=k, thu=self.thucut)
        # top_scores = self.rank_by_tfidf(inverted_index, self.N, query, topk=k, use_cosine=True)
        top_scores = self.enhanced_bm25_scores(inverted_index, self.doc_length, query, self.N, k=k, thu=self.thu)
        results = [(id_pair[docid], score) for docid, score in top_scores]
        return results
    
    def retrieval_url(self, query, k=20):
        if not query:
            return []
            
        # Parse advanced query syntax
        parsed_query = self.query_parser.parse_query(query)
        base_query = parsed_query['base_query']
        
        # If base query is empty but has advanced filter conditions, use all documents as candidates
        if not base_query.strip() and any([
            parsed_query['inurl_terms'],
            parsed_query['exact_phrases'], 
            parsed_query['site_filters'],
            parsed_query['site_or_filters'],
            parsed_query['directory_filters']
        ]):
            # Get all documents as candidates
            results = [(filename, 1.0) for filename in self.id_pair.values()]
        else:
            # Use base query for retrieval, get more candidate results for subsequent filtering
            results = self.retrieval_by_score(self.inverted_index, self.id_pair, base_query, k=k*3)
        
        # Apply advanced filtering and score adjustment
        filtered_results = []
        
        for filename, score in results:
            url = self.get_url(filename)
            
            # URL filtering
            if not self.query_parser.check_url_filter(url, parsed_query['inurl_terms']):
                continue
                
            # Site filtering
            if not self.query_parser.check_site_filter(url, parsed_query['site_filters'], parsed_query['site_or_filters']):
                continue
                
            # Directory filtering
            if not self.query_parser.check_directory_filter(url, parsed_query['directory_filters']):
                continue
            
            # Exact phrase filtering and score adjustment
            if parsed_query['exact_phrases']:
                # Read document content for exact phrase matching
                title, content = self._load_document_content(filename)
                phrase_match, phrase_count = self.query_parser.check_exact_phrase(
                    title, content, parsed_query['exact_phrases']
                )
                
                if not phrase_match:
                    continue  # Must hit exact phrase
                    
                # Exact phrase bonus
                score += phrase_count * 2.0
            
            # URL hit bonus
            if parsed_query['inurl_terms']:
                url_lower = url.lower()
                parsed_url = urlparse(url_lower)
                path = parsed_url.path
                
                for term in parsed_query['inurl_terms']:
                    if term.lower() in path:
                        score += 0.5  # Small bonus
            
            filtered_results.append((filename, score))
        
        # 按评分重新排序
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果的URL
        final_results = filtered_results[:k]
        url_list = [self.get_url(result[0]) for result in final_results]
        
        return url_list
    
    def _load_document_content(self, filename):
        """
        加载文档内容用于精确短语匹配
        
        Args:
            filename: 文档文件名
            
        Returns:
            tuple: (title, content)
        """
        try:
            file_path = os.path.join(self.txt_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 简单地将第一行作为标题，其余作为内容
            lines = content.split('\n', 1)
            title = lines[0] if lines else ''
            body = lines[1] if len(lines) > 1 else ''
            
            return title, body
        except Exception as e:
            # 如果读取失败，返回空内容
            return '', ''
    
    def _compute_ordered_window_tf(self, positions1, positions2, window_size=8):
        """
        计算有序窗口词频 (Ordered Window TF)
        使用双指针算法找到满足 0 < pos2 - pos1 <= window_size 的位置对数量
        
        Args:
            positions1: 第一个词的位置列表
            positions2: 第二个词的位置列表  
            window_size: 窗口大小，默认为8
        
        Returns:
            int: 有序窗口匹配次数
        """
        if not positions1 or not positions2:
            return 0
            
        count = 0
        i, j = 0, 0
        
        while i < len(positions1) and j < len(positions2):
            pos1, pos2 = positions1[i], positions2[j]
            
            if 0 < pos2 - pos1 <= window_size:
                count += 1
                i += 1
                j += 1
            elif pos2 <= pos1:
                j += 1
            else:  # pos2 - pos1 > window_size
                i += 1
                
        return count
    
    def _compute_unordered_window_tf(self, positions1, positions2, window_size=8):
        """
        计算无序窗口词频 (Unordered Window TF)
        找到最小跨度 <= window_size 的片段个数
        
        Args:
            positions1: 第一个词的位置列表
            positions2: 第二个词的位置列表
            window_size: 窗口大小，默认为8
        
        Returns:
            int: 无序窗口匹配次数
        """
        if not positions1 or not positions2:
            return 0
            
        # 合并并排序所有位置，同时记录来源
        all_positions = []
        for pos in positions1:
            all_positions.append((pos, 1))  # 标记为第一个词
        for pos in positions2:
            all_positions.append((pos, 2))  # 标记为第二个词
            
        all_positions.sort()
        
        count = 0
        i = 0
        
        while i < len(all_positions):
            # 寻找包含两个词的最小窗口
            seen_words = set()
            j = i
            
            while j < len(all_positions) and len(seen_words) < 2:
                seen_words.add(all_positions[j][1])
                j += 1
                
            if len(seen_words) == 2:  # 找到包含两个词的窗口
                span = all_positions[j-1][0] - all_positions[i][0]
                if span <= window_size:
                    count += 1
                    
            i += 1
            
        return count
    
    def _compute_window_features(self, query_terms, window_size=8):
        """
        Compute window features for all adjacent term pairs in the query
        
        Args:
            query_terms: List of query terms
            window_size: Window size
            
        Returns:
            tuple: (ordered_features, unordered_features)
                   Each is a dictionary of {(term1, term2): {docid: tf}}
        """
        ordered_features = defaultdict(lambda: defaultdict(int))
        unordered_features = defaultdict(lambda: defaultdict(int))
        
        # Iterate through all adjacent term pairs
        for i in range(len(query_terms) - 1):
            term1, term2 = query_terms[i], query_terms[i + 1]
            
            # Get inverted lists for both terms
            postings1 = self.get_posting_list(self.inverted_index, term1)
            postings2 = self.get_posting_list(self.inverted_index, term2)
            
            # Build mapping from document ID to positions
            doc_positions1 = {p.docid: p.positions for p in postings1 if hasattr(p, 'positions')}
            doc_positions2 = {p.docid: p.positions for p in postings2 if hasattr(p, 'positions')}
            
            # Find documents containing both terms
            common_docs = set(doc_positions1.keys()) & set(doc_positions2.keys())
            
            for docid in common_docs:
                positions1 = doc_positions1[docid]
                positions2 = doc_positions2[docid]
                
                # Compute ordered window features
                ordered_tf = self._compute_ordered_window_tf(positions1, positions2, window_size)
                if ordered_tf > 0:
                    ordered_features[(term1, term2)][docid] = ordered_tf
                    
                # Compute unordered window features
                unordered_tf = self._compute_unordered_window_tf(positions1, positions2, window_size)
                if unordered_tf > 0:
                    unordered_features[(term1, term2)][docid] = unordered_tf
                    
        return ordered_features, unordered_features
    
    def enhanced_bm25_scores(self, inverted_index, doc_length, query, N, k=3, k1=1.3, b=0.75, 
                           lambda_t=0.7, lambda_o=0.2, lambda_u=0.1, window_size=6, thu=False):
        """
        Sequential Dependence Model (SDM) enhanced BM25 algorithm
        
        Implementation principle:
        SDM scores query-document relevance through three types of evidence:
        1. Unigram: Traditional BM25 scoring, weight λ_T
        2. Ordered Window: Adjacent query terms maintain order and distance ≤ w, weight λ_O  
        3. Unordered Window: Adjacent query terms don't need order but fall within same window ≤ w, weight λ_U
        
        Final score = λ_T * Σlog(f_T) + λ_O * Σlog(f_O) + λ_U * Σlog(f_U)
        where λ_T + λ_O + λ_U = 1
        
        Args:
            inverted_index: Inverted index (must contain position information)
            doc_length: Document length dictionary
            query: Query string
            N: Total number of documents
            k: Number of results to return
            k1, b: BM25 parameters
            lambda_t: Unigram feature weight (default 0.8)
            lambda_o: Ordered window feature weight (default 0.15)
            lambda_u: Unordered window feature weight (default 0.05)
            window_size: Window size (default 8)
            thu: Whether to use thulac tokenization
            
        Returns:
            List[(docid, score)]: Document list sorted by score in descending order
        """
        # Ensure weights sum to 1
        total_weight = lambda_t + lambda_o + lambda_u
        lambda_t /= total_weight
        lambda_o /= total_weight  
        lambda_u /= total_weight
        
        scores = defaultdict(lambda: 0.0)
        thu = self.thu
        # Tokenization
        if thu:
            query_terms = self._tokenize_query_with_thulac(query)
        else:
            query_terms = [term for term in jieba.cut(query) if term not in self.stop_list]
            
        if not query_terms:
            return []
        
        # print(query_terms)
            
        # 1. Unigram features (traditional BM25)
        query_term_counts = Counter(query_terms)
        for term, count in query_term_counts.items():
            postings_list = self.get_posting_list(inverted_index, term)
            if not postings_list:
                continue
                
            df = len(postings_list)
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1)
            
            for posting in postings_list:
                tf = posting.tf
                docid = posting.docid
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length[docid] / self.avg_length)))
                scores[docid] += lambda_t * math.log(bm25_score + 1)  # Add 1 to avoid log(0)
        
        # 2. Compute window features
        if len(query_terms) > 1:  # Only compute window features when there are multiple terms
            ordered_features, unordered_features = self._compute_window_features(query_terms, window_size)
            
            # 3. Ordered Window features
            for (term1, term2), doc_tfs in ordered_features.items():
                # Calculate document frequency for this term pair
                df_od = len(doc_tfs)
                if df_od == 0:
                    continue
                    
                idf_od = math.log(((N - df_od + 0.5) / (df_od + 0.5)) + 1)
                
                for docid, tf_od in doc_tfs.items():
                    # Use BM25-style scoring
                    bm25_od = idf_od * ((tf_od * (k1 + 1)) / (tf_od + k1 * (1 - b + b * doc_length[docid] / self.avg_length)))
                    scores[docid] += lambda_o * math.log(bm25_od + 1)
            
            # 4. Unordered Window features  
            for (term1, term2), doc_tfs in unordered_features.items():
                # Calculate document frequency for this term pair
                df_uw = len(doc_tfs)
                if df_uw == 0:
                    continue
                    
                idf_uw = math.log(((N - df_uw + 0.5) / (df_uw + 0.5)) + 1)
                
                for docid, tf_uw in doc_tfs.items():
                    # Use BM25-style scoring
                    bm25_uw = idf_uw * ((tf_uw * (k1 + 1)) / (tf_uw + k1 * (1 - b + b * doc_length[docid] / self.avg_length)))
                    scores[docid] += lambda_u * math.log(bm25_uw + 1)
        
        
        # Sort and return results
        results = [(docid, score) for docid, score in scores.items()]
        results.sort(key=lambda x: -x[1])
        return results[:k]
