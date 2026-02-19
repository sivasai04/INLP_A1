from tokenizers import load_tokenizer
from language_models import load_model

def main():
    test_prefixes = [
        "There are many",
    ]

    import gc
    tokenizers_config = [
        ('whitespace', 'models/tokenizers/whitespace_en.pkl'),
        ('regex', 'models/tokenizers/regex_en.pkl'),
        ('bpe', 'models/tokenizers/bpe_en.pkl')
    ]

    smoothing_methods = ['none', 'witten-bell', 'kneser-ney']

    for tok_name, tok_path in tokenizers_config:
        print(f"Loading Tokenizer: {tok_name.upper()}")
        try:
            tokenizer = load_tokenizer(tok_path)
        except Exception as e:
            print(f"Failed to load tokenizer {tok_name}: {e}")
            continue

        for smoothing in smoothing_methods:
            model_path = f"models/language_models/{tok_name}_{smoothing}.pkl"
            print(f"\n  Testing Model: {tok_name} + {smoothing}")
            try:
                lm = load_model(model_path)
                print("  Running autocomplete examples...")
                for prefix in test_prefixes:
                    prefix_tokens = tokenizer.tokenize(prefix)
                    completed_tokens = lm.autocomplete(prefix_tokens, max_length=15)
                    completed_text = tokenizer.detokenize(completed_tokens)
                    
                    print(f"    Input:  {prefix}")
                    print(f"    Output: {completed_text}")
                # Explicitly free memory
                del lm
                gc.collect()
                
            except Exception as e:
                print(f"    Error loading/testing model {model_path}: {e}")


if __name__ == "__main__":
    main()
