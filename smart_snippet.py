# smart_snippet.py - 智能摘要生成器
import re
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
from html import escape
import jieba

class SmartSnippetGenerator:
    """Smart snippet generator for implementing high-quality search result summaries"""
    
    def __init__(self, stop_list: List[str], inverted_index: Dict = None):
        """
        Initialize smart snippet generator
        Args:
            stop_list: Stop words list
            inverted_index: Inverted index for calculating IDF weights
        """
        self.stop_list = set(stop_list)
        self.inverted_index = inverted_index
        self.chinese_punctuation = set('。！？；，、：')
        
        # Low information words extension (common function words besides stop words)
        self.low_info_words = set(['上', '下', '中', '内', '外', '前', '后', '左', '右', 
                                  '东', '西', '南', '北', '里', '间', '时', '候', '等'])
    
    def preprocess_query(self, query: str) -> Dict[str, any]:
        """
        A. Preprocess query: tokenization and deduplication, stop word filtering, n-gram construction
        Args:
            query: Original query string
        Returns:
            Dict containing: terms, ngrams, full_phrase, term_weights
        """
        # 1. Tokenize and deduplicate
        raw_terms = list(jieba.cut(query.strip(), cut_all=False))
        terms = list(dict.fromkeys([t for t in raw_terms if t.strip()]))
        
        # 2. Filter stop words and low information words
        filtered_terms = [t for t in terms 
                         if t not in self.stop_list and t not in self.low_info_words and len(t) > 0]
        
        # 3. Keep complete phrase (remove whitespace)
        full_phrase = re.sub(r'\s+', '', query)
        
        # 4. Construct n-grams (2-gram and 3-gram)
        ngrams = []
        for i in range(len(terms) - 1):
            # 2-gram
            bigram = terms[i] + terms[i + 1]
            if len(bigram) > 1:
                ngrams.append(bigram)
            
            # 3-gram
            if i < len(terms) - 2:
                trigram = terms[i] + terms[i + 1] + terms[i + 2]
                if len(trigram) > 2:
                    ngrams.append(trigram)
        
        # 5. Calculate weights
        term_weights = self._calculate_term_weights(filtered_terms, ngrams, full_phrase)
        
        return {
            'terms': filtered_terms,
            'ngrams': ngrams,
            'full_phrase': full_phrase,
            'term_weights': term_weights,
            'all_patterns': filtered_terms + ngrams + [full_phrase] if full_phrase else filtered_terms + ngrams
        }
    
    def _calculate_term_weights(self, terms: List[str], ngrams: List[str], full_phrase: str) -> Dict[str, float]:
        """Calculate term weights"""
        weights = {}
        
        # Base weight: term length
        for term in terms:
            weights[term] = len(term)
        
        # N-gram weights are higher
        for ngram in ngrams:
            weights[ngram] = len(ngram) * 1.5
        
        # Complete phrase has highest weight
        if full_phrase:
            weights[full_phrase] = len(full_phrase) * 2.0
        
        return weights
    
    def generate_smart_snippet(self, text: str, query: str, window_size: int = 200) -> str:
        """Generate smart snippet"""
        if not text or not query:
            return ""
        
        # Preprocess query
        query_data = self.preprocess_query(query)
        patterns = query_data['all_patterns']
        
        if not patterns:
            # If no valid query terms, return beginning of text
            snippet = text[:window_size]
            return f"<p>{escape(snippet)}...</p>"
        
        # Find best matching position
        best_pos = 0
        max_matches = 0
        
        for i in range(len(text) - window_size + 1):
            window_text = text[i:i + window_size]
            matches = sum(1 for pattern in patterns if pattern in window_text)
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        # If no matches found, use beginning of text
        if max_matches == 0:
            snippet = text[:window_size]
        else:
            snippet = text[best_pos:best_pos + window_size]
        
        # Highlighting processing
        highlighted_snippet = self._highlight_text(snippet, patterns)
        
        return f"<p>{highlighted_snippet}...</p>"
    
    def _highlight_text(self, text: str, patterns: List[str]) -> str:
        """Highlight matching words in text"""
        # Sort by length, prioritize matching longer words
        sorted_patterns = sorted(patterns, key=len, reverse=True)
        
        highlighted = escape(text)
        
        for pattern in sorted_patterns:
            if pattern in text:
                escaped_pattern = escape(pattern)
                highlighted = highlighted.replace(escaped_pattern, f"<mark>{escaped_pattern}</mark>")
        
        return highlighted