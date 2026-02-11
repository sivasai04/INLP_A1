import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import pickle


class NGramLM:
    """4-Gram Language Model with configurable smoothing."""
    
    def __init__(self, n: int = 4, smoothing: str = 'none'):
        """
        Initialize n-gram language model.
        
        Args:
            n: N-gram order (default 4)
            smoothing: 'none', 'witten-bell', or 'kneser-ney'
        """
        self.n = n
        self.smoothing = smoothing
        
        # N-gram counts for all orders
        self.ngram_counts = [defaultdict(int) for _ in range(n + 1)]
        self.context_counts = [defaultdict(int) for _ in range(n)]
        
        # For Kneser-Ney: track unique contexts
        self.continuation_counts = [defaultdict(set) for _ in range(n)]
        
        # Vocabulary
        self.vocab = set()
        self.vocab_size = 0
        
        # Special tokens
        self.start_token = '<s>'
        self.end_token = '</s>'
        self.unk_token = '<unk>'
    
    def train(self, tokens: List[str]):
        """
        Train model on tokenized text.
        
        Args:
            tokens: List of tokens
        """
        # Build vocabulary
        self.vocab = set(tokens)
        self.vocab.add(self.start_token)
        self.vocab.add(self.end_token)
        self.vocab.add(self.unk_token)
        self.vocab_size = len(self.vocab)
        
        # Add boundary markers
        padded = [self.start_token] * (self.n - 1) + tokens + [self.end_token]
        
        # Count all n-grams
        for i in range(len(padded)):
            for order in range(1, self.n + 1):
                if i + order <= len(padded):
                    ngram = tuple(padded[i:i + order])
                    self.ngram_counts[order][ngram] += 1
                    
                    if order > 1:
                        context = ngram[:-1]
                        self.context_counts[order - 1][context] += 1
                        self.continuation_counts[order - 1][context].add(ngram[-1])
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get probability of n-gram using selected smoothing.
        
        Args:
            ngram: Tuple of tokens
        
        Returns:
            Probability
        """
        if self.smoothing == 'none':
            return self._prob_no_smoothing(ngram)
        elif self.smoothing == 'witten-bell':
            return self._prob_witten_bell(ngram)
        elif self.smoothing == 'kneser-ney':
            return self._prob_kneser_ney(ngram)
        else:
            raise ValueError(f"Unknown smoothing: {self.smoothing}")
    
    def _prob_no_smoothing(self, ngram: Tuple[str, ...]) -> float:
        """No smoothing with backoff."""
        order = len(ngram)
        
        if order == 1:
            count = self.ngram_counts[1][(ngram[0],)]
            total = sum(self.ngram_counts[1].values())
            return count / total if total > 0 else 0.0
        
        context = ngram[:-1]
        count = self.ngram_counts[order][ngram]
        context_count = self.context_counts[order - 1][context]
        
        if context_count > 0:
            return count / context_count
        else:
            # Backoff
            return self._prob_no_smoothing(ngram[1:])
    
    def _prob_witten_bell(self, ngram: Tuple[str, ...]) -> float:
        """
        Witten-Bell smoothing.
        Formula: P(w|h) = C(hw) / (C(h) + T(h))
        where T(h) = number of unique continuations of h
        """
        order = len(ngram)
        
        if order == 1:
            count = self.ngram_counts[1][(ngram[0],)]
            total = sum(self.ngram_counts[1].values())
            return (count + 1) / (total + self.vocab_size)
        
        context = ngram[:-1]
        word = ngram[-1]
        
        ngram_count = self.ngram_counts[order][ngram]
        context_count = self.context_counts[order - 1][context]
        
        if context_count == 0:
            return self._prob_witten_bell(ngram[1:])
        
        # Number of unique words after context
        T = len(self.continuation_counts[order - 1][context])
        
        # Witten-Bell formula
        prob = ngram_count / (context_count + T)
        
        if ngram_count == 0:
            # Backoff with interpolation
            backoff_prob = self._prob_witten_bell(ngram[1:])
            lambda_weight = T / (context_count + T)
            prob = lambda_weight * backoff_prob
        
        return prob
    
    def _prob_kneser_ney(self, ngram: Tuple[str, ...], d: float = 0.75) -> float:
        """
        Kneser-Ney smoothing with absolute discounting.
        
        Args:
            ngram: N-gram tuple
            d: Discount (default 0.75)
        
        Formula:
        P_KN(w|h) = max(C(hw) - d, 0) / C(h) + λ(h) * P_cont(w)
        """
        order = len(ngram)
        
        if order == 1:
            # Continuation probability for unigram
            word = ngram[0]
            num_contexts = sum(1 for ctx in self.continuation_counts[0].values() 
                             if word in ctx)
            total_contexts = sum(len(ctx) for ctx in self.continuation_counts[0].values())
            return num_contexts / total_contexts if total_contexts > 0 else 1.0 / self.vocab_size
        
        context = ngram[:-1]
        word = ngram[-1]
        
        ngram_count = self.ngram_counts[order][ngram]
        context_count = self.context_counts[order - 1][context]
        
        if context_count == 0:
            return self._prob_kneser_ney(ngram[1:], d)
        
        # Discounted probability
        discounted = max(ngram_count - d, 0)
        prob = discounted / context_count
        
        # Interpolation weight
        num_unique = len(self.continuation_counts[order - 1][context])
        lambda_weight = (d * num_unique) / context_count
        
        # Backoff
        continuation_prob = self._prob_kneser_ney(ngram[1:], d)
        
        prob += lambda_weight * continuation_prob
        
        return prob
    
    def sentence_probability(self, tokens: List[str]) -> float:
        """
        Calculate log probability of sentence.
        
        Args:
            tokens: Sentence tokens
        
        Returns:
            Log probability
        """
        padded = [self.start_token] * (self.n - 1) + tokens + [self.end_token]
        
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            ngram = tuple(padded[i - self.n + 1:i + 1])
            prob = self.get_probability(ngram)
            
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # Avoid log(0)
        
        return log_prob
    
    def autocomplete(self, prefix_tokens: List[str], max_length: int = 100) -> List[str]:
        """
        Complete sentence from prefix using greedy search.
        
        Args:
            prefix_tokens: Initial tokens
            max_length: Maximum tokens to generate
        
        Returns:
            Completed sentence tokens
        """
        current = [self.start_token] * (self.n - 1) + prefix_tokens
        
        for _ in range(max_length):
            context = tuple(current[-(self.n - 1):])
            
            # Find best next token
            best_token = None
            best_prob = -1
            
            for token in self.vocab:
                if token in [self.start_token, self.unk_token]:
                    continue
                
                ngram = context + (token,)
                prob = self.get_probability(ngram)
                
                if prob > best_prob:
                    best_prob = prob
                    best_token = token
            
            if best_token is None or best_token == self.end_token:
                break
                
            current.append(best_token)
        
        # Return without start markers
        return current[self.n - 1:]
    
    def perplexity(self, tokens: List[str]) -> float:
        """
        Calculate perplexity on test data.
        PP(W) = exp(-1/N * Σ log P(w_i | context))
        
        Args:
            tokens: Test tokens
        
        Returns:
            Perplexity score
        """
        log_prob = self.sentence_probability(tokens)
        N = len(tokens) + 1  # +1 for end token
        
        return math.exp(-log_prob / N)


def save_model(model, filepath: str):
    """Save language model to file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str):
    """Load language model from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
