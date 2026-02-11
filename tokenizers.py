import re
from collections import defaultdict, Counter
from typing import List, Tuple
import pickle


class WhitespaceTokenizer:
    """
    Whitespace-based tokenizer that treats whitespace as separator
    and special characters as individual tokens.
    """
    
    def __init__(self):
        self.vocab = set()
    
    def train(self, text: str):
        """Build vocabulary from training data."""
        tokens = self.tokenize(text)
        self.vocab = set(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize by splitting on whitespace and separating special characters.
        """
        tokens = []
        words = text.split()
        
        for word in words:
            # Separate alphanumeric from non-alphanumeric
            parts = re.findall(r'\w+|[^\w\s]', word)
            tokens.extend(parts)
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        if not tokens:
            return ""
        
        result = [tokens[0]]
        for token in tokens[1:]:
            # Don't add space before punctuation
            if re.match(r'^[^\w\s]$', token):
                result.append(token)
            else:
                result.append(' ' + token)
        
        return ''.join(result)


class RegexTokenizer:
    """Regex-based tokenizer with language-specific patterns."""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.vocab = set()
        
        # Language-specific patterns
        if language == 'english':
            # Handles contractions, words, punctuation
            self.pattern = r"\w+(?:'\w+)?|[^\w\s]"
        elif language == 'mongolian':
            # Cyrillic script (U+0400 to U+04FF)
            self.pattern = r"[\u0400-\u04FF\w]+|[^\w\s]"
        else:
            self.pattern = r"\w+|[^\w\s]"
    
    def train(self, text: str):
        """Build vocabulary from training data."""
        tokens = self.tokenize(text)
        self.vocab = set(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize using regex pattern."""
        return re.findall(self.pattern, text)
    
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        if not tokens:
            return ""
        
        result = [tokens[0]]
        for i, token in enumerate(tokens[1:], 1):
            # No space before punctuation or after apostrophe
            if re.match(r'^[^\w\s]$', token) or (i > 0 and tokens[i-1] == "'"):
                result.append(token)
            else:
                result.append(' ' + token)
        
        return ''.join(result)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer implemented from scratch."""
    
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2, unk_token: str = '<UNK>'):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.merges = []  # List of (pair, merged) tuples
        self.vocab = set([unk_token])  # Initialize with UNK token
    
    def train(self, text: str):
        """
        Train BPE on text.
        Start with character-level and iteratively merge frequent pairs.
        """
        # Get word frequencies
        words = text.split()
        word_freqs = Counter(words)
        
        # Initialize with character-level tokens
        # Add </w> as end-of-word marker
        vocab = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + ['</w>'])
            vocab[chars] = freq
        
        # CRITICAL: Add all individual characters to base vocabulary
        # This ensures characters seen in training won't be marked as UNK
        base_chars = set()
        for word in word_freqs.keys():
            base_chars.update(list(word))
        base_chars.add('</w>')  ## adding end of word marker because it prevents - 
           ## - model to merging the end of the word with the next word.
        
        self.vocab.update(base_chars)
        
        # Perform BPE merges
        for iteration in range(self.vocab_size):
            # Count all adjacent pairs
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Merge the pair
            merged = best_pair[0] + best_pair[1]
            new_vocab = {}
            
            for word, freq in vocab.items():
                new_word = []
                i= 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            
            vocab = new_vocab
            self.merges.append((best_pair, merged))
        
        # Build final vocabulary
        self.vocab = set()
        for word in vocab.keys():
            self.vocab.update(word)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using learned BPE merges.
        Unknown characters are replaced with UNK token.
        """
        words = text.split()
        tokens = []
        
        for word in words:
            # Start with character-level, replace unknown chars with UNK
            chars = [c if c in self.vocab or c == '</w>' else self.unk_token for c in list(word)] + ['</w>']
            
            # Apply merges in order
            for pair, merged in self.merges:
                i = 0
                new_chars = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == pair[0] and chars[i+1] == pair[1]:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            
            tokens.extend(chars)
        
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from BPE tokens."""
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()


def save_tokenizer(tokenizer, filepath: str):
    """Save tokenizer to file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(filepath: str):
    """Load tokenizer from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
