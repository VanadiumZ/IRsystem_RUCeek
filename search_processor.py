import os
import re
import jieba
import dill
import numpy as np
from collections import Counter, defaultdict
from bs4 import BeautifulSoup, Comment


class Posting(object):
    """Posting item in inverted index with position information"""
    special_doc_id = -1
    
    def __init__(self, docid, tf=0, positions=None):
        self.docid = docid
        self.tf = tf
        self.positions = positions if positions is not None else []
    
    def __repr__(self):
        return "<docid: %s, tf: %d, positions: %s>" % (self.docid, self.tf, self.positions)


class SearchEngineProcessor:
    """Search engine data processor
    
    Integrates HTML to text conversion, document ID mapping, and inverted index construction
    """
    
    def __init__(self, 
                 html_dir='tempHTML', 
                 txt_dir='final_txt',
                 stopwords_file='baidu_stopwords.txt',
                 jieba_dict_file='jieba_dict.txt',
                 batch_size=300,
                 segmenter='jieba'):
        """
        Initialize the processor
        
        Args:
            html_dir: HTML files directory
            txt_dir: Text files directory
            stopwords_file: Stopwords file
            jieba_dict_file: Jieba dictionary file
            batch_size: Batch processing size
            segmenter: Segmenter type ('jieba' or 'thulac')
        """
        self.html_dir = html_dir
        self.txt_dir = txt_dir
        self.stopwords_file = stopwords_file
        self.jieba_dict_file = jieba_dict_file
        self.batch_size = batch_size
        self.segmenter = segmenter
        
        # Initialize data structures
        self.id_pairs = {}
        self.term_docid_position_pairs = []  # 存储(term, docid, position)三元组
        self.doc_length = []
        self.inverted_index = defaultdict(lambda: [Posting(Posting.special_doc_id, 0, [])])
        self.stop_list = []
        
        # Initialize thulac if needed
        self.thulac_seg = None
        if segmenter == 'thulac':
            try:
                import thulac
                self.thulac_seg = thulac.thulac(seg_only=True)
            except ImportError:
                print("Warning: thulac not installed, falling back to jieba")
                self.segmenter = 'jieba'
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load jieba dictionary and stopwords"""
        # Load jieba dictionary
        if os.path.exists(self.jieba_dict_file):
            jieba.load_userdict(self.jieba_dict_file)
        
        # Load stopwords
        if os.path.exists(self.stopwords_file):
            with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
    
    def html_to_txt(self, file):
        """Convert a single HTML file to text file"""
        html_path = os.path.join(self.html_dir, file)
        filename = file.split('.')[0]
        txt_path = os.path.join(self.txt_dir, filename + '.txt')
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                url = soup.find(string=lambda text: isinstance(text, Comment))
            
            # Ensure output directory exists
            os.makedirs(self.txt_dir, exist_ok=True)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                if url:
                    f.write(url + '\n')
                f.write(text)
            
            return True
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return False
    
    def batch_html_to_txt(self, file_list=None):
        """Batch convert HTML files to text files"""
        if file_list is None:
            if not os.path.exists(self.html_dir):
                print(f"HTML directory {self.html_dir} does not exist")
                return
            file_list = os.listdir(self.html_dir)
        
        batch = []
        count = 0
        total_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        for file in file_list:
            batch.append(file)
            if len(batch) == self.batch_size:
                count += 1
                success_count = 0
                for f in batch:
                    if self.html_to_txt(f):
                        success_count += 1
                print(f'Batch ({count}/{total_batches}) converted, success: {success_count}/{len(batch)}')
                batch = []
        
        # Process remaining files
        if batch:
            count += 1
            success_count = 0
            for f in batch:
                if self.html_to_txt(f):
                    success_count += 1
            print(f'Batch ({count}/{total_batches}) converted, success: {success_count}/{len(batch)}')
    
    def extract_doc_id(self, filename):
        """Extract document ID from filename"""
        return re.split(r'[_\\.]+', filename)[1]
    
    def build_id_pairs(self, file_list=None):
        """Build mapping from document ID to filename"""
        if file_list is None:
            if not os.path.exists(self.txt_dir):
                print(f"Text directory {self.txt_dir} does not exist")
                return
            file_list = os.listdir(self.txt_dir)
        
        batch = []
        count = 0
        total_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        for file in file_list:
            batch.append(file)
            if len(batch) == self.batch_size:
                count += 1
                for f in batch:
                    try:
                        docid = self.extract_doc_id(f)
                        self.id_pairs[docid] = f
                    except Exception as e:
                        print(f"Error processing ID for file {f}: {e}")
                print(f'Batch ({count}/{total_batches}) ID mapping saved')
                batch = []
        
        # Process remaining files
        if batch:
            count += 1
            for f in batch:
                try:
                    docid = self.extract_doc_id(f)
                    self.id_pairs[docid] = f
                except Exception as e:
                    print(f"Error processing ID for file {f}: {e}")
            print(f'Batch ({count}/{total_batches}) ID mapping saved')
        
        # Sort ID mapping
        self.id_pairs = dict(sorted(self.id_pairs.items()))
    
    def process_text_file(self, file):
        """Process a single text file, extract terms with positions and calculate document length"""
        filepath = os.path.join(self.txt_dir, file)
        docid = self.extract_doc_id(file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()[1:]  # Skip first line URL
                contents = " ".join(line.strip() for line in content)
                
                # Tokenization with position tracking
                if self.segmenter == 'thulac' and self.thulac_seg:
                    # Use thulac segmentation
                    seg_result = self.thulac_seg.cut(contents, text=True)
                    raw_terms = seg_result.split()
                else:
                    # Use jieba segmentation
                    raw_terms = list(jieba.cut(contents))
                
                # Filter terms and track positions
                terms_with_positions = []
                position = 0
                for term in raw_terms:
                    if (len(term.strip()) > 1) and (term not in self.stop_list):
                        terms_with_positions.append((term, position))
                    position += 1
                
                if not terms_with_positions:
                    terms_with_positions = [('empty_file', 0)]
                
                # Add term-document ID-position triplets
                for term, pos in terms_with_positions:
                    self.term_docid_position_pairs.append((term, docid, pos))
                
                # Calculate document length
                terms_only = [term for term, _ in terms_with_positions]
                term_counts = np.array(list(Counter(terms_only).values()))
                log_func = np.vectorize(lambda x: 1.0 + np.log10(x) if x > 0 else 0.0)
                log_tf = log_func(term_counts)
                doc_len = np.sqrt(np.sum(log_tf**2))
                self.doc_length.append((docid, doc_len))
                
                return True
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return False
    
    def build_inverted_index(self, file_list=None):
        """Build inverted index"""
        if file_list is None:
            if not os.path.exists(self.txt_dir):
                print(f"Text directory {self.txt_dir} does not exist")
                return
            file_list = os.listdir(self.txt_dir)
        
        # Process files in batches
        batch = []
        count = 0
        total_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        for f in file_list:
            batch.append(f)
            if len(batch) >= self.batch_size:
                count += 1
                success_count = 0
                for f in batch:
                    if self.process_text_file(f):
                        success_count += 1
                print(f'Batch ({count}/{total_batches}) processed, success: {success_count}/{len(batch)}')
                batch = []
        
        # Process remaining files
        if batch:
            count += 1
            success_count = 0
            for f in batch:
                if self.process_text_file(f):
                    success_count += 1
            print(f'Batch ({count}/{total_batches}) processed, success: {success_count}/{len(batch)}')
        
        # Sort and build inverted index
        print("Building inverted index with position information...")
        self.term_docid_position_pairs = sorted(self.term_docid_position_pairs)
        self.doc_length = sorted(self.doc_length, key=lambda x: x[0])
        
        # Build inverted index with positions
        self.inverted_index = defaultdict(lambda: [Posting(Posting.special_doc_id, 0, [])])
        
        for term, docid, position in self.term_docid_position_pairs:
            postings_list = self.inverted_index[term]
            if postings_list[-1].docid == docid:
                postings_list[-1].tf += 1
                postings_list[-1].positions.append(position)
            else:
                postings_list.append(Posting(docid, 1, [position]))
        
        print("Inverted index construction completed")
    
    def save_data(self, 
                  doc_id_pair_file='doc_id_pair.pkl',
                  doc_length_file='doc_length.pkl',
                  inverted_index_file='inverted_index.pkl'):
        """Save processing results to files"""
        try:
            # Save document ID mapping
            with open(doc_id_pair_file, 'wb') as f:
                dill.dump(self.id_pairs, f)
            print(f"Document ID mapping saved to {doc_id_pair_file}")
            
            # Save document lengths
            with open(doc_length_file, 'wb') as f:
                dill.dump(self.doc_length, f)
            print(f"Document lengths saved to {doc_length_file}")
            
            # Save inverted index
            with open(inverted_index_file, 'wb') as f:
                dill.dump(self.inverted_index, f)
            print(f"Inverted index saved to {inverted_index_file}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_data(self,
                  doc_id_pair_file='doc_id_pair.pkl',
                  doc_length_file='doc_length.pkl',
                  inverted_index_file='inverted_index.pkl'):
        """Load data from files"""
        try:
            # Load document ID mapping
            if os.path.exists(doc_id_pair_file):
                with open(doc_id_pair_file, 'rb') as f:
                    self.id_pairs = dill.load(f)
                print(f"Document ID mapping loaded from {doc_id_pair_file}")
            
            # Load document lengths
            if os.path.exists(doc_length_file):
                with open(doc_length_file, 'rb') as f:
                    self.doc_length = dill.load(f)
                print(f"Document lengths loaded from {doc_length_file}")
            
            # Load inverted index
            if os.path.exists(inverted_index_file):
                with open(inverted_index_file, 'rb') as f:
                    self.inverted_index = dill.load(f)
                print(f"Inverted index loaded from {inverted_index_file}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def process_all(self):
        """Execute complete processing pipeline"""
        print("Starting search engine data processing...")
        
        # 1. HTML to text conversion
        print("\nStep 1: HTML to text conversion")
        self.batch_html_to_txt()
        
        # 2. Build ID mapping
        print("\nStep 2: Build document ID mapping")
        self.build_id_pairs()
        
        # 3. Build inverted index
        print("\nStep 3: Build inverted index")
        self.build_inverted_index()
        
        # 4. Save data
        print("\nStep 4: Save processing results")
        self.save_data()
        
        print("\nSearch engine data processing completed!")
        print(f"Processed {len(self.id_pairs)} documents")
        print(f"Built inverted index with {len(self.inverted_index)} terms")
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            'total_documents': len(self.id_pairs),
            'total_terms': len(self.inverted_index),
            'total_term_doc_position_pairs': len(self.term_docid_position_pairs),
            'avg_doc_length': np.mean([length for _, length in self.doc_length]) if self.doc_length else 0,
            'segmenter_used': self.segmenter
        }


if __name__ == "__main__":
    # Usage example
    processor = SearchEngineProcessor()
    
    # Execute complete processing pipeline
    processor.process_all()
    
    # Display statistics
    stats = processor.get_stats()
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")