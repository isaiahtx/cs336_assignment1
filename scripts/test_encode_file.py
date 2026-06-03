from src.assignment1.bpe.tokenizer import encode_file_to_chunks, Tokenizer
import pickle

ts_special_tokens_path = "results/bpe_train_tiny_stories/special_tokens.pkl"
with open(ts_special_tokens_path, "rb") as f:
    ts_special_tokens = pickle.load(f)

ts_tokenizer = Tokenizer.from_files(
    "results/bpe_train_tiny_stories/vocab.pkl",
    "results/bpe_train_tiny_stories/merges.pkl",
    ts_special_tokens
)

encode_file_to_chunks(ts_tokenizer,"data/owt_valid.txt",'test123')