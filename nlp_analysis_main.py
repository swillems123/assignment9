#!/usr/bin/env python3
"""
NLP Analysis Tool
=================

This script performs natural language processing analysis on text files,
including tokenization, stemming, lemmatization, named entity recognition,
n-gram analysis, and authorship attribution.

Usage:
    python nlp_analysis_main.py [--data_dir DATA_DIR]

Arguments:
    --data_dir DATA_DIR  Directory containing text files (default: current directory)
"""

import nltk
import os
import sys
import argparse
import re
import string
from collections import Counter
import math
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from nltk import ne_chunk, pos_tag

# --- NLTK Resource Management ---

# List of required NLTK resources
REQUIRED_RESOURCES = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words'
]

def ensure_nltk_resources():
    """Check if required NLTK resources are downloaded, and download if missing."""
    print("Checking NLTK resources...")
    all_downloaded = True
    for resource in REQUIRED_RESOURCES:
        try:
            # Use nltk.data.find to check if the resource is available
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif resource == 'averaged_perceptron_tagger':
                 nltk.data.find('taggers/averaged_perceptron_tagger')
            elif resource == 'maxent_ne_chunker':
                 nltk.data.find('chunkers/maxent_ne_chunker')
            elif resource == 'words':
                 nltk.data.find('corpora/words')
            # If no LookupError, the resource is found
            print(f"  [+] Resource '{resource}' found.")
        except LookupError:
            print(f"  [-] Resource '{resource}' not found. Downloading...")
            all_downloaded = False
            try:
                nltk.download(resource, quiet=False) # Set quiet=False for visibility
                print(f"      Downloaded '{resource}' successfully.")
            except Exception as e:
                 print(f"      Error downloading '{resource}': {e}")
                 sys.exit(f"Failed to download critical NLTK resource: {resource}. Please check internet connection or NLTK setup.")
    if all_downloaded:
        print("All required NLTK resources are available.")
    else:
        print("Finished checking/downloading resources.")
    print("-" * 30) # Separator

# --- TextProcessor Class ---

class TextProcessor:
    """Class for text processing operations."""

    def __init__(self):
        """Initialize the text processor."""
        # Stopwords should be available due to check at script start
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        # Use Treebank tokenizer to avoid potential punkt lookup errors during word tokenization
        self.word_tokenizer = TreebankWordTokenizer()

    def read_file(self, file_path):
        """
        Read text file with error handling.

        Args:
            file_path (str): Path to the text file

        Returns:
            str: Content of the file, or None on error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"  Error reading file {os.path.basename(file_path)} with latin-1 encoding: {e}")
                return None
        except FileNotFoundError:
             print(f"  Error: File not found at {file_path}")
             return None
        except Exception as e:
            print(f"  Error reading file {os.path.basename(file_path)}: {e}")
            return None

    def preprocess_text(self, text):
        """
        Clean and preprocess text.

        Args:
            text (str): Raw text

        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
             return ""

        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        """
        Split text into tokens using TreebankWordTokenizer.

        Args:
            text (str): Preprocessed text

        Returns:
            list: List of tokens
        """
        if not isinstance(text, str):
            return []
        return self.word_tokenizer.tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.

        Args:
            tokens (list): List of tokens

        Returns:
            list: Tokens with stopwords removed
        """
        if not isinstance(tokens, list):
            return []
        # Filter out stopwords and short tokens
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def stem_tokens(self, tokens):
        """
        Apply stemming to token list.

        Args:
            tokens (list): List of tokens

        Returns:
            list: Stemmed tokens
        """
        if not isinstance(tokens, list):
            return []
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to token list.

        Args:
            tokens (list): List of tokens

        Returns:
            list: Lemmatized tokens
        """
        if not isinstance(tokens, list):
            return []
        return [self.lemmatizer.lemmatize(token) for token in tokens]

# --- TextAnalyzer Class ---

class TextAnalyzer:
    """Class for text analysis operations."""

    def __init__(self):
        """Initialize the text analyzer."""
        # Resources should be available due to check at script start
        pass

    def find_most_common(self, items, n=20):
        """
        Find the most common items in a list.

        Args:
            items (list or iterable): List of items to count
            n (int): Number of most common items to return

        Returns:
            list: List of (item, count) tuples for the most common items
        """
        if not items:
            return []
        try:
            # Use Counter for efficient counting
            return Counter(items).most_common(n)
        except TypeError:
             print(f"  Warning: Attempted to count unhashable items. Returning empty list.")
             return []

    def extract_named_entities(self, text):
        """
        Extract named entities from text.

        Args:
            text (str): Raw text

        Returns:
            list: List of named entity strings
        """
        if not isinstance(text, str) or not text.strip():
            return []

        entities = []
        try:
            # Process sentence by sentence
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tagged = pos_tag(tokens)
                chunked = ne_chunk(tagged)
                # Extract entities (trees with labels)
                for chunk in chunked:
                    if hasattr(chunk, 'label'):
                        entity = ' '.join(c[0] for c in chunk)
                        entities.append(entity)
        except Exception as e:
             print(f"  Error during named entity extraction: {e}")
             return [] # Return empty on error

        return entities

    def generate_ngrams(self, tokens, n=3):
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens (list): List of tokens
            n (int): Size of n-grams (e.g., 3 for trigrams)

        Returns:
            list: List of n-gram tuples
        """
        if not isinstance(tokens, list) or len(tokens) < n:
            return []
        return list(ngrams(tokens, n))

# --- AuthorshipAnalyzer Class ---

class AuthorshipAnalyzer:
    """Class for authorship attribution."""

    def __init__(self):
        """Initialize the authorship analyzer."""
        pass

    def calculate_similarity(self, ngrams1, ngrams2):
        """
        Calculate similarity between two sets of n-grams using Jaccard similarity.

        Args:
            ngrams1 (list): First list of n-grams
            ngrams2 (list): Second list of n-grams

        Returns:
            float: Similarity score between 0 and 1
        """
        if not ngrams1 or not ngrams2:
            return 0.0
        set1, set2 = set(ngrams1), set(ngrams2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def calculate_style_similarity(self, text_data1, text_data2):
        """
        Calculate style similarity between two texts based on extracted features.

        Args:
            text_data1 (dict): First text data dictionary
            text_data2 (dict): Second text data dictionary

        Returns:
            float: Style similarity score (cosine similarity) between 0 and 1
        """
        if not text_data1 or not text_data2:
            return 0.0
        features1 = self._extract_stylistic_features(text_data1)
        features2 = self._extract_stylistic_features(text_data2)
        return self._cosine_similarity(features1, features2)

    def _extract_stylistic_features(self, text_data):
        """
        Extract stylistic features from text data.

        Args:
            text_data (dict): Text data dictionary containing 'tokens'

        Returns:
            dict: Dictionary of stylistic features (e.g., avg word length, richness)
        """
        features = {}
        tokens = text_data.get('tokens', []) # Use raw tokens for some style features
        clean_tokens = text_data.get('clean_tokens', []) # Use clean tokens for others

        if tokens:
            # Average word length
            word_lengths = [len(token) for token in tokens]
            if word_lengths:
                features['avg_word_length'] = sum(word_lengths) / len(word_lengths)

            # Vocabulary richness (Type-Token Ratio on clean tokens)
            if clean_tokens:
                 features['vocabulary_richness'] = len(set(clean_tokens)) / len(clean_tokens) if clean_tokens else 0.0

            # Function word usage (example set)
            function_words = {'the', 'a', 'an', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'is', 'was', 'and', 'but'}
            func_word_count = sum(1 for token in tokens if token in function_words)
            features['func_word_ratio'] = func_word_count / len(tokens) if tokens else 0.0

        # Add sentence length from stats if available
        stats = text_data.get('stats', {})
        if 'avg_sentence_length' in stats:
             features['avg_sentence_length'] = stats['avg_sentence_length']

        return features

    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two feature vectors (dictionaries).

        Args:
            vec1 (dict): First feature vector
            vec2 (dict): Second feature vector

        Returns:
            float: Cosine similarity between 0 and 1
        """
        all_keys = set(vec1.keys()).union(vec2.keys())
        dot_product = sum(vec1.get(key, 0) * vec2.get(key, 0) for key in all_keys)
        mag1 = math.sqrt(sum(vec1.get(key, 0) ** 2 for key in all_keys))
        mag2 = math.sqrt(sum(vec2.get(key, 0) ** 2 for key in all_keys))
        return dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0

# --- Main Execution Logic ---

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NLP Text Analysis Tool')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing text files (default: current directory)')
    parser.add_argument('--unknown_file', type=str, default='RJ_Lovecraft.txt', # Default based on previous context
                        help='Filename of the text with unknown authorship (default: RJ_Lovecraft.txt)')
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='Size of n-grams for analysis (default: 3)')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top common items (tokens, n-grams) to display (default: 20)')
    return parser.parse_args()

def main():
    """Main entry point for the program."""
    # 0. Ensure NLTK resources are ready
    ensure_nltk_resources()

    # 1. Parse Arguments
    args = parse_arguments()

    # 2. Validate Data Directory
    if not os.path.isdir(args.data_dir): # Check if it's a directory
        print(f"Error: Path '{args.data_dir}' is not a valid directory.")
        return 1

    # 3. Find Text Files
    try:
        all_files = os.listdir(args.data_dir)
        text_files = [f for f in all_files if f.endswith('.txt') and os.path.isfile(os.path.join(args.data_dir, f))]
    except OSError as e:
        print(f"Error accessing directory '{args.data_dir}': {e}")
        return 1

    if not text_files:
        print(f"No .txt files found in '{args.data_dir}'. Please check the directory path.")
        return 1

    print(f"Found {len(text_files)} text files in '{args.data_dir}'.")
    unknown_text_path = os.path.join(args.data_dir, args.unknown_file)
    if args.unknown_file not in text_files:
         print(f"Warning: Unknown file '{args.unknown_file}' not found in the directory. Authorship analysis will be skipped.")
         args.unknown_file = None # Disable authorship part if file missing

    # 4. Initialize Processors/Analyzers
    processor = TextProcessor()
    analyzer = TextAnalyzer()
    authorship_analyzer = AuthorshipAnalyzer()

    # 5. Process Each File
    processed_texts = {}
    print("\n===== PART 1: TEXT PROCESSING AND BASIC ANALYSIS =====\n")
    for filename in text_files:
        file_path = os.path.join(args.data_dir, filename)
        print(f"Processing {filename}...")

        # Read and process
        text = processor.read_file(file_path)
        if text is None: # Skip file if read failed
             print(f"  Skipping {filename} due to read error.")
             continue
        processed_text = processor.preprocess_text(text)
        tokens = processor.tokenize_text(processed_text) # Use Treebank tokenizer result
        clean_tokens = processor.remove_stopwords(tokens)
        stemmed_tokens = processor.stem_tokens(clean_tokens)
        lemmatized_tokens = processor.lemmatize_tokens(clean_tokens)

        # Basic Analysis
        common_lemmas = analyzer.find_most_common(lemmatized_tokens, args.top_n)
        named_entities = analyzer.extract_named_entities(text) # Use raw text for NER

        # Store results
        processed_texts[filename] = {
            'raw_text': text,
            'processed_text': processed_text,
            'tokens': tokens, # Store raw tokens from Treebank
            'clean_tokens': clean_tokens,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'common_lemmas': common_lemmas,
            'named_entities': named_entities
            # N-grams will be added later
        }

        # Print basic analysis results
        print(f"\nResults for {filename}:")
        print(f"Top {args.top_n} most common lemmatized tokens (excluding stopwords):")
        if common_lemmas:
            for token, count in common_lemmas:
                print(f"  {token}: {count}")
        else:
            print("  No common lemmas found (or text was empty/short).")

        print(f"\nNumber of unique named entities found: {len(set(named_entities))}")
        if named_entities:
            unique_entities = sorted(list(set(named_entities)))
            print(f"Sample named entities: {unique_entities[:10]}") # Show first 10 unique
        else:
            print("  No named entities found.")

        print("\n" + "-" * 50)

    # 6. N-gram Analysis and Authorship Attribution
    print("\n===== PART 2: N-GRAM ANALYSIS AND AUTHORSHIP ATTRIBUTION =====\n")

    # Generate n-grams for all processed texts
    for filename, data in processed_texts.items():
        if 'clean_tokens' in data:
            ngrams_list = analyzer.generate_ngrams(data['clean_tokens'], args.ngram_size)
            processed_texts[filename]['ngrams'] = ngrams_list
            common_ngrams = analyzer.find_most_common(ngrams_list, args.top_n)
            processed_texts[filename]['common_ngrams'] = common_ngrams

            # Print top n-grams
            print(f"\nTop {args.top_n} {args.ngram_size}-grams for {filename}:")
            if common_ngrams:
                for ngram_tuple, count in common_ngrams:
                    print(f"  {' '.join(ngram_tuple)}: {count}")
            else:
                print(f"  No {args.ngram_size}-grams generated (text likely too short).")
        else:
             print(f"Skipping n-gram generation for {filename} as 'clean_tokens' are missing.")

    # Perform Authorship Attribution if unknown file exists
    if args.unknown_file and args.unknown_file in processed_texts:
        print(f"\nPerforming authorship analysis for {args.unknown_file}...")
        unknown_data = processed_texts[args.unknown_file]

        # Ensure unknown text has n-grams
        if 'ngrams' not in unknown_data:
            print(f"  Cannot perform authorship analysis: N-grams missing for {args.unknown_file}.")
        else:
            comparisons = []
            # Compare with all *other* processed texts that have n-grams
            for author_fname, author_data in processed_texts.items():
                if author_fname == args.unknown_file or 'ngrams' not in author_data:
                    continue # Skip self-comparison or texts without n-grams

                # Calculate similarities
                ngram_similarity = authorship_analyzer.calculate_similarity(
                    author_data['ngrams'],
                    unknown_data['ngrams']
                )
                style_similarity = authorship_analyzer.calculate_style_similarity(
                    author_data, # Pass the whole dict for feature extraction
                    unknown_data
                )
                # Simple average for combined score (can be weighted)
                combined_similarity = (ngram_similarity + style_similarity) / 2.0

                comparisons.append((author_fname, ngram_similarity, style_similarity, combined_similarity))

            # Sort by combined similarity (highest first)
            comparisons.sort(key=lambda x: x[3], reverse=True)

            # Print results
            print("\nAuthorship attribution results (compared to other files):")
            if comparisons:
                for author, ngram_sim, style_sim, combined_sim in comparisons:
                    print(f"{author}:")
                    print(f"  N-gram Similarity (Jaccard): {ngram_sim:.4f}")
                    print(f"  Stylistic Similarity (Cosine): {style_sim:.4f}")
                    print(f"  Combined Score (Average): {combined_sim:.4f}")

                most_likely_author_file = comparisons[0][0]
                print(f"\n---> Based on combined score, the style of '{args.unknown_file}' is most similar to '{most_likely_author_file}'.")
            else:
                print("  No other suitable texts found for comparison.")
    elif args.unknown_file:
         print(f"\nSkipping authorship analysis because '{args.unknown_file}' was not successfully processed or found.")


    # 7. Subject Determination (Based on assignment instructions)
    print("\n===== PART 3: SUBJECT DETERMINATION =====\n")
    print("Attempting to determine the subject based on common tokens and named entities:")

    # Define keywords/entities related to Romeo and Juliet
    rj_keywords = {'romeo', 'juliet', 'montague', 'capulet', 'verona', 'tybalt', 'mercutio', 'nurse', 'friar'}
    # Define keywords for other potential themes based on filenames
    martin_keywords = {'lord', 'house', 'keep', 'realm', 'iron', 'throne', 'winter'} # Example
    tolkien_keywords = {'elves', 'shire', 'ring', 'hobbit', 'gandalf', 'middle-earth', 'orc'} # Example
    lovecraft_keywords = {'eldritch', 'cosmic', 'ancient', 'madness', 'cthulhu', 'arkham', 'forbidden'} # Example

    for filename, data in processed_texts.items():
        print(f"\nSubject analysis for {filename}:")
        common_lemmas_set = {lemma for lemma, count in data.get('common_lemmas', [])}
        entities_set = {entity.lower() for entity in data.get('named_entities', [])}

        # Check for Romeo and Juliet theme
        if any(keyword in common_lemmas_set for keyword in rj_keywords) or \
           any(keyword in entities_set for keyword in rj_keywords):
            print("  -> Subject appears to be related to 'Romeo and Juliet'.")

            # Guess style based on filename convention (as per original context)
            if "martin" in filename.lower():
                print("     Style convention suggests: George R.R. Martin")
            elif "tolkien" in filename.lower(): # Corrected spelling
                print("     Style convention suggests: J.R.R. Tolkien")
            elif "lovecraft" in filename.lower():
                print("     Style convention suggests: H.P. Lovecraft")
            else:
                 print("     Style convention not recognized from filename.")

        # Check for other themes based on keywords (example)
        elif any(keyword in common_lemmas_set for keyword in martin_keywords) and "martin" in filename.lower():
             print("  -> Subject might be related to a 'Game of Thrones'-like fantasy setting.")
        elif any(keyword in common_lemmas_set for keyword in tolkien_keywords) and "tolkien" in filename.lower():
             print("  -> Subject might be related to a 'Lord of the Rings'-like fantasy setting.")
        elif any(keyword in common_lemmas_set for keyword in lovecraft_keywords) and "lovecraft" in filename.lower():
             print("  -> Subject might be related to 'Lovecraftian Cosmic Horror'.")
        else:
            # Generic fallback
            top_5_lemmas = [lemma for lemma, count in data.get('common_lemmas', [])[:5]]
            top_5_entities = sorted(list(set(data.get('named_entities', []))))[:5]
            print(f"  -> Subject seems distinct. Top lemmas: {top_5_lemmas}. Sample entities: {top_5_entities}")


    print("\nAnalysis complete!")
    return 0

# --- Script Entry Point ---

if __name__ == "__main__":
    # Ensure the script exits with the return code from main()
    sys.exit(main())
