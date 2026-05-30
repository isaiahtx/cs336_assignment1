from src.assignment1.profiler import run_with_monitor
from src.assignment1.bpe.train import train_bpe
from pathlib import Path
import pickle

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    vocab, merges = run_with_monitor(
        train_bpe,
        './data/TinyStoriesV2-GPT4-valid.txt',
        10_000,
        special_tokens,
        pretokenizer_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        num_processes = 6,
        desired_num_chunks = 6 * 4,
    )

    results_folder = Path("results/bpe_train_tiny_stories_valid")
    results_folder.mkdir(parents=True,exist_ok=True)
    
    with open(results_folder / 'vocab.pkl','wb') as f:
        pickle.dump(vocab,f)
    
    with open(results_folder / 'merges.pkl','wb') as f:
        pickle.dump(merges,f)
    
    with open(results_folder / 'special_tokens.pkl','wb') as f:
        pickle.dump(special_tokens,f)
    
    print(f"longest token: {max(vocab.values(),key=len)}")