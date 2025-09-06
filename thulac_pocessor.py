# filename: search_processor_thulac.py
import os
import re
import dill
import numpy as np
from collections import Counter, defaultdict
from bs4 import BeautifulSoup, Comment

import thulac  # <-- 使用 THULAC

class Posting(object):
    """Posting item in inverted index with position information"""
    special_doc_id = -1

    def __init__(self, docid, tf=0, positions=None):
        self.docid = docid
        self.tf = tf
        self.positions = positions if positions is not None else []

    def __repr__(self):
        return "<docid: %s, tf: %d, positions: %s>" % (self.docid, self.tf, self.positions)


class SearchEngineProcessorTHU:
    """
    Search engine data processor (THULAC version)

    - Maintains the same four-step process as the original file:
      1) (Optional) HTML->TXT
      2) Build docid mapping
      3) Build inverted index
      4) Serialize to disk

    - Only replaces tokenization with THULAC, supports user_dict and custom stopwords.
    """

    def __init__(self,
                 html_dir='tempHTML',
                 txt_dir='final_txt',
                 stopwords_file='baidu_stopwords.txt',
                 user_dict_file='thulac_user_dict.txt',
                 batch_size=300,
                 seg_only=True,
                 t2s=False,
                 filt=False):
        """
        Args:
            html_dir: HTML file directory
            txt_dir: Plain text directory
            stopwords_file: Stopwords file (one per line)
            user_dict_file: THULAC user dictionary (UTF-8, one word per line)
            batch_size: Batch processing size
            seg_only: Whether THULAC only segments (no POS tags), recommended True
            t2s: Whether to convert traditional to simplified Chinese
            filt: Whether to enable THULAC built-in filtering (remove punctuation/misc symbols)
        """
        self.html_dir = html_dir
        self.txt_dir = txt_dir
        self.stopwords_file = stopwords_file
        self.user_dict_file = user_dict_file
        self.batch_size = batch_size

        # Internal data structures
        self.id_pairs = {}  # {docid: filename}
        self.term_docid_position_pairs = []  # [(term, docid, position), ...]
        self.doc_length = []  # [(docid, length), ...]
        self.inverted_index = defaultdict(lambda: [Posting(Posting.special_doc_id, 0, [])])
        self.stop_list = []

        # THULAC tokenizer instance
        self.thu = None
        self.seg_only = seg_only
        self.t2s = t2s
        self.filt = filt

        self._load_config()

    # -----------------------------
    # Configuration loading (equivalent to original, changed to THULAC)
    # -----------------------------
    def _load_config(self):
        """Load THULAC and stopwords"""
        # Initialize THULAC
        user_dict = self.user_dict_file if (self.user_dict_file and os.path.exists(self.user_dict_file)) else None
        self.thu = thulac.thulac(
            seg_only=self.seg_only,
            user_dict=user_dict,
            T2S=self.t2s,
            filt=self.filt
        )

        # Load stopwords
        if os.path.exists(self.stopwords_file):
            with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                self.stop_list = [line.rstrip() for line in f.readlines()]
        else:
            self.stop_list = []

    # -----------------------------
    # HTML -> TXT
    # -----------------------------
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

        if batch:
            count += 1
            success_count = 0
            for f in batch:
                if self.html_to_txt(f):
                    success_count += 1
            print(f'Batch ({count}/{total_batches}) converted, success: {success_count}/{len(batch)}')

    # -----------------------------
    # Document ID mapping
    # -----------------------------
    def extract_doc_id(self, filename):
        """Extract document ID from filename, e.g. xxx_1234_xxx.txt -> 1234"""
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

        if batch:
            count += 1
            for f in batch:
                try:
                    docid = self.extract_doc_id(f)
                    self.id_pairs[docid] = f
                except Exception as e:
                    print(f"Error processing ID for file {f}: {e}")
            print(f'Batch ({count}/{total_batches}) ID mapping saved')

        self.id_pairs = dict(sorted(self.id_pairs.items()))

    # -----------------------------
    # Tokenization and inverted index (core change: thulac)
    # -----------------------------
    def _tokenize(self, text):
        """
        Uniformly returns List[str] (words only, no POS tags), compatible with seg_only True/False.
        THULAC returns [[word, tag], ...] when text=False (default),
        and returns space-separated string when text=True.
        """
        if self.seg_only:
            # Get plain text string directly, then split by space
            seg_text = self.thu.cut(text, text=True)  # e.g. "我 来到 北京 清华大学"
            tokens = seg_text.split()
        else:
            # Get 2D array, take the 0th column (word) of each entry
            raw = self.thu.cut(text, text=False)      # e.g. [[word, tag], ...]
            tokens = []
            for item in raw:
                if isinstance(item, (list, tuple)) and item:
                    tokens.append(item[0])
                elif isinstance(item, str):
                    tokens.append(item)
                # Ignore other abnormal formats
        return tokens

    def process_text_file(self, file):
        """Process a single text file, extract terms with positions and calculate document length"""
        filepath = os.path.join(self.txt_dir, file)
        docid = self.extract_doc_id(file)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()[1:]  # Skip first line URL
                contents = " ".join(line.strip() for line in content)

                # === Tokenization & filtering (same filtering criteria as original), while recording positions ===
                raw_terms = self._tokenize(contents)
                terms_with_positions = []
                position = 0
                for term in raw_terms:
                    if (len(term.strip()) > 1) and (term not in self.stop_list):
                        terms_with_positions.append((term, position))
                    position += 1

                if not terms_with_positions:
                    terms_with_positions = [('empty_file', 0)]

                # term-doc-position triplets
                for term, pos in terms_with_positions:
                    self.term_docid_position_pairs.append((term, docid, pos))

                # Document length (consistent with original logic: log tf + L2)
                terms_only = [term for term, _ in terms_with_positions]
                term_counts = np.array(list(Counter(terms_only).values()))
                log_func = np.vectorize(lambda x: 1.0 + np.log10(x) if x > 0 else 0.0)
                log_tf = log_func(term_counts)
                doc_len = np.sqrt(np.sum(log_tf ** 2))
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

        if batch:
            count += 1
            success_count = 0
            for f in batch:
                if self.process_text_file(f):
                    success_count += 1
            print(f'Batch ({count}/{total_batches}) processed, success: {success_count}/{len(batch)}')

        print("Building inverted index with position information...")
        self.term_docid_position_pairs = sorted(self.term_docid_position_pairs)
        self.doc_length = sorted(self.doc_length, key=lambda x: x[0])

        self.inverted_index = defaultdict(lambda: [Posting(Posting.special_doc_id, 0, [])])
        for term, docid, position in self.term_docid_position_pairs:
            postings_list = self.inverted_index[term]
            if postings_list[-1].docid == docid:
                postings_list[-1].tf += 1
                postings_list[-1].positions.append(position)
            else:
                postings_list.append(Posting(docid, 1, [position]))
        print("Inverted index construction completed")

    # -----------------------------
    # Serialization / Deserialization
    # -----------------------------
    def save_data(self,
                  doc_id_pair_file='doc_id_pair_thu.pkl',
                  doc_length_file='doc_length_thu.pkl',
                  inverted_index_file='inverted_index_thu.pkl'):
        try:
            with open(doc_id_pair_file, 'wb') as f:
                dill.dump(self.id_pairs, f)
            print(f"Document ID mapping saved to {doc_id_pair_file}")

            with open(doc_length_file, 'wb') as f:
                dill.dump(self.doc_length, f)
            print(f"Document lengths saved to {doc_length_file}")

            with open(inverted_index_file, 'wb') as f:
                dill.dump(self.inverted_index, f)
            print(f"Inverted index saved to {inverted_index_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self,
                  doc_id_pair_file='doc_id_pair.pkl',
                  doc_length_file='doc_length.pkl',
                  inverted_index_file='inverted_index.pkl'):
        try:
            if os.path.exists(doc_id_pair_file):
                with open(doc_id_pair_file, 'rb') as f:
                    self.id_pairs = dill.load(f)
                print(f"Document ID mapping loaded from {doc_id_pair_file}")

            if os.path.exists(doc_length_file):
                with open(doc_length_file, 'rb') as f:
                    self.doc_length = dill.load(f)
                print(f"Document lengths loaded from {doc_length_file}")

            if os.path.exists(inverted_index_file):
                with open(inverted_index_file, 'rb') as f:
                    self.inverted_index = dill.load(f)
                print(f"Inverted index loaded from {inverted_index_file}")
        except Exception as e:
            print(f"Error loading data: {e}")

    # -----------------------------
    # One-click process & statistics
    # -----------------------------
    def process_all(self):
        print("Starting search engine data processing (THULAC)...")

        # (If HTML->TXT is needed, uncomment)
        # print("\nStep 1: HTML to text conversion")
        # self.batch_html_to_txt()

        print("\nStep 2: Build document ID mapping")
        self.build_id_pairs()

        print("\nStep 3: Build inverted index")
        self.build_inverted_index()

        print("\nStep 4: Save processing results")
        self.save_data()

        print("\nSearch engine data processing completed!")
        print(f"Processed {len(self.id_pairs)} documents")
        print(f"Built inverted index with {len(self.inverted_index)} terms")

    def get_stats(self):
        return {
            'total_documents': len(self.id_pairs),
            'total_terms': len(self.inverted_index),
            'total_term_doc_position_pairs': len(self.term_docid_position_pairs),
            'avg_doc_length': np.mean([length for _, length in self.doc_length]) if self.doc_length else 0,
            'segmenter_used': 'thulac'
        }


if __name__ == "__main__":
    # Usage example
    processor = SearchEngineProcessorTHU(
        html_dir='tempHTML',
        txt_dir='final_txt',
        stopwords_file='baidu_stopwords.txt',
        user_dict_file='thulac_user_dict.txt',  # Optional: can run even if not exists
        batch_size=300,
        seg_only=True,   # Only tokenization, no POS tags
        t2s=False,
        filt=False
    )

    processor.process_all()
    stats = processor.get_stats()
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
