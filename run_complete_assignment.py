import os
from tokenizers import WhitespaceTokenizer, RegexTokenizer, BPETokenizer, save_tokenizer
from language_models import NGramLM, save_model
from utils import read_text

def main():
    
    # Load preprocessed data
    if not os.path.exists('data/en/train.txt') or not os.path.exists('data/mn/train.txt'):
        print("ERROR: Preprocessed data not found!")
        print("Please run 'python utils.py' first.")
        return
    
    en_train = read_text('data/en/train.txt')
    en_val = read_text('data/en/val.txt')
    en_test = read_text('data/en/test.txt')
    
    mn_train = read_text('data/mn/train.txt')
    mn_val = read_text('data/mn/val.txt')
    mn_test = read_text('data/mn/test.txt')

    # TRAIN TOKENIZERS
    
    print("\n  -----ENGLISH TOKENIZERS------")
    print("  - Whitespace tokenizer...")
    ws_en = WhitespaceTokenizer()
    ws_en.train(en_train)
    
    print("  - Regex tokenizer...")
    regex_en = RegexTokenizer('english')
    regex_en.train(en_train)
    
    print("  - BPE tokenizer (vocab_size=5000, min_frequency=3)...")
    bpe_en = BPETokenizer(vocab_size=5000, min_frequency=3)
    bpe_en.train(en_train)
    
    print("\n  -----MONGOLIAN TOKENIZERS----")
    print("  - Whitespace tokenizer...")
    ws_mn = WhitespaceTokenizer()
    ws_mn.train(mn_train)
    
    print("  - Regex tokenizer...")
    regex_mn = RegexTokenizer('mongolian')
    regex_mn.train(mn_train)
    
    print("  - BPE tokenizer (vocab_size=5000, min_frequency=3)...")
    bpe_mn = BPETokenizer(vocab_size=5000, min_frequency=3)
    bpe_mn.train(mn_train)
    
    # Save all tokenizers for future use
    print("\n  Saving all tokenizers...")
    os.makedirs('models/tokenizers', exist_ok=True)
    save_tokenizer(ws_en, 'models/tokenizers/whitespace_en.pkl')
    save_tokenizer(regex_en, 'models/tokenizers/regex_en.pkl')
    save_tokenizer(bpe_en, 'models/tokenizers/bpe_en.pkl')
    save_tokenizer(ws_mn, 'models/tokenizers/whitespace_mn.pkl')
    save_tokenizer(regex_mn, 'models/tokenizers/regex_mn.pkl')
    save_tokenizer(bpe_mn, 'models/tokenizers/bpe_mn.pkl')
    print("All 6 tokenizers saved to models/tokenizers/")
    
    # LANGUAGE MODEL TRAINING
    
    # Define tokenizers to process sequentially
    tokenizer_configs = [
        ('whitespace', ws_en),
        ('regex', regex_en),
        ('bpe', bpe_en)
    ]
    
    smoothing_methods = ['none', 'witten-bell', 'kneser-ney']
    
    for tok_name, tokenizer in tokenizer_configs:
        print(f"Processing {tok_name} tokenizer...")
        
        # Tokenize data (only for this tokenizer)
        print(f"  Tokenizing data...")
        train_tokens = tokenizer.tokenize(en_train)
        val_tokens = tokenizer.tokenize(en_val)
        test_tokens = tokenizer.tokenize(en_test)
        
        # Train models for this tokenizer
        for smoothing in smoothing_methods:
            model_name = f"{tok_name}_{smoothing}"
            print(f"\n    - Training {model_name}...")
            
            lm = NGramLM(n=4, smoothing=smoothing)
            lm.train(train_tokens)
            
            # Evaluate Perplexity
            val_perplexity = lm.perplexity(val_tokens)
            test_perplexity = lm.perplexity(test_tokens)
            
            print(f"      Val Perplexity:  {val_perplexity:.2f}")
            print(f"      Test Perplexity: {test_perplexity:.2f}")
            
            #  Save the trained model
            os.makedirs('models/language_models', exist_ok=True)
            model_path = f'models/language_models/{model_name}.pkl'
            save_model(lm, model_path)
            print(f"      Model saved to {model_path}")
        
        # 5. Clear memory for this tokenizer's data
        del train_tokens
        del val_tokens
        del test_tokens
        import gc
        gc.collect()
        print(f"\n  Finished {tok_name} processing. Memory cleared.")


if __name__ == "__main__":
    main()
