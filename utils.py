import json
import random
import re
from typing import List, Tuple


def load_jsonl_corpus(filepath: str) -> str:
    """
    Load corpus from JSONL file format.
    Each line is a JSON object with 'text' field.
    
    Args:
        filepath: Path to JSONL file
    
    Returns:
        Combined text from all lines
    """
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data and data['text']:
                    texts.append(data['text'].strip())
            except json.JSONDecodeError:
                continue
    
    return '\n'.join(texts)


def clean_text(text: str) -> str:
    """
    Clean text by removing detritus like excessive spacing, bad alignment,
    and weird unicode artifacts.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width unicode characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
    
    # Fix excessive spacing - multiple spaces to single
    text = re.sub(r' +', ' ', text)
    
    # Fix excessive newlines - max 2 consecutive
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # Remove trailing/leading spaces on each line
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines at start/end
    text = text.strip()
    
    return text


def partition_data(text: str, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, test_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[str, str, str]:
    """
    Split corpus into train/validation/test partitions.
    Splits at line level to avoid cutting sentences.
    
    Args:
        text: Text to partition
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    
    Returns:
        (train_text, val_text, test_text)
    """
    # Split into lines
    lines = [line for line in text.split('\n') if line.strip()]
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(lines)
    
    # Calculate split points
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    return '\n'.join(train_lines), '\n'.join(val_lines), '\n'.join(test_lines)


def get_subsample(text: str, num_lines: int = 50000, seed: int = 42) -> str:
    """
    Extract a random sample of lines from text for computationally 
    expensive tasks like BPE training.
    
    Args:
        text: The source text (e.g., training partition)
        num_lines: Number of lines to extract
        seed: Random seed for reproducibility
        
    Returns:
        A string containing the subsampled lines
    """
    lines = text.split('\n')
    if len(lines) <= num_lines:
        return text
        
    random.seed(seed)
    # Using random.sample ensures we get a diverse set of lines 
    # rather than just the first num_lines
    sampled_lines = random.sample(lines, num_lines)
    return '\n'.join(sampled_lines)


def write_text(filepath: str, text: str):
    """Write text to file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


def read_text(filepath: str) -> str:
    """Read text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == "__main__":
    # This block only runs if you execute 'python utils.py'
    # It will NOT run if you import utils in another file.
    
    import os
    
    datasets = ['en', 'mn']  # English and Mongolian
    
    for lang in datasets:
        input_file = f'cc100_{lang}.jsonl'
        output_dir = f'data/{lang}'
        
        print(f"--- Processing {lang.upper()} Dataset ---")
        
        if os.path.exists(input_file):
            # 1. Load
            raw_text = load_jsonl_corpus(input_file)
            
            # 2. Clean
            cleaned = clean_text(raw_text)
            
            # 3. Partition
            train, val, test = partition_data(cleaned)
            
            # 4. Save
            write_text(os.path.join(output_dir, 'train.txt'), train)
            write_text(os.path.join(output_dir, 'val.txt'), val)
            write_text(os.path.join(output_dir, 'test.txt'), test)
            
            print(f"Successfully saved partitions to {output_dir}\n")
        else:
            print(f"Error: {input_file} not found. Skipping...")
