from src.assignment1.bpe.tokenizer import Tokenizer, encode_file_to_chunks
from src.assignment1.bpe.utils import sample_documents
import time
import os
import shutil
import pickle
from pathlib import Path


def get_dir_size(dir_path: str | Path):
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            total_size += os.path.getsize(file_path)
    return total_size


def parts_a_and_b():
    seed = 42

    ts_special_tokens_path = "results/bpe_train_tiny_stories/special_tokens.pkl"
    with open(ts_special_tokens_path, "rb") as f:
        ts_special_tokens = pickle.load(f)

    owt_special_tokens_path = "results/bpe_train_owt/special_tokens.pkl"
    with open(owt_special_tokens_path, "rb") as f:
        owt_special_tokens = pickle.load(f)

    ts_tokenizer = Tokenizer.from_files(
        "results/bpe_train_tiny_stories/vocab.pkl",
        "results/bpe_train_tiny_stories/merges.pkl",
        ts_special_tokens
    )
    
    owt_tokenizer = Tokenizer.from_files(
        "results/bpe_train_owt/vocab.pkl",
        "results/bpe_train_owt/merges.pkl",
        owt_special_tokens
    )

    ts_stories = sample_documents(
        "./data/TinyStoriesV2-GPT4-valid.txt",
        seed=seed,
        special_tokens=ts_tokenizer.special_tokens
    )

    owt_stories = sample_documents(
        "./data/owt_valid.txt",
        seed=seed,
        special_tokens=owt_tokenizer.special_tokens
    )
    
    for name,stories,tokenizer in [
            ("Tokenized 10 ts stories with ts tokenizer",ts_stories,ts_tokenizer),
            ("Tokenized 10 ts stories with owt tokenizer",ts_stories,owt_tokenizer),
            ("Tokenized 10 owt stories with ts tokenizer",owt_stories,ts_tokenizer),
            ("Tokenized 10 owt stories with owt tokenizer",owt_stories,owt_tokenizer)
        ]:
        encodings = [tokenizer.encode(story) for story in stories]
        num_tokens = sum(map(len,encodings))
        num_bytes = sum(map(len,stories))
        print(f"{name}, got {num_bytes / num_tokens:.2f} bytes/token ({num_bytes} bytes; {num_tokens} tokens)")

def part_c():
    special_tokens_path = "results/bpe_train_owt/special_tokens.pkl"
    with open(special_tokens_path, "rb") as f:
        special_tokens = pickle.load(f)

    tokenizer = Tokenizer.from_files(
        "results/bpe_train_owt/vocab.pkl",
        "results/bpe_train_owt/merges.pkl",
        special_tokens
    )

    target_size_gb = 825

    output = "/tmp/test_owt_valid"

    with open("data/owt_valid.txt","rb") as f:
        f.seek(0,os.SEEK_END)
        file_size_gb = f.tell()/ (1024 ** 3)

    print(f"file size: {file_size_gb:.3f} GB")
    print(f"Encoding...")

    t0 = time.perf_counter()
    nids = sum(n for _,_,n in encode_file_to_chunks(
        tokenizer,
        "data/owt_valid.txt",
        output,
    ))
    tot = time.perf_counter() - t0

    print(f"Encoded {nids:,} tokens in {tot:,.3f} seconds: {round(nids/tot):,} tk/s. Total disk space: {get_dir_size(output) / (1024 * 1024):,.2f} MB.")
    ext_total_seconds = tot * target_size_gb / file_size_gb
    ext_hours = int(ext_total_seconds / 3600)
    ext_minutes = int(ext_total_seconds / 60 - ext_hours * 60)
    ext_seconds = int(ext_total_seconds - 3600 * ext_hours - 60 * ext_minutes)
    print(f"Extrapolating, it would take {ext_hours}h{ext_minutes}m{ext_seconds}s to encode {target_size_gb} GB of text.")

    shutil.rmtree(output)


def part_d():
    ts_special_tokens_path = "results/bpe_train_tiny_stories/special_tokens.pkl"
    with open(ts_special_tokens_path, "rb") as f:
        ts_special_tokens = pickle.load(f)

    owt_special_tokens_path = "results/bpe_train_owt/special_tokens.pkl"
    with open(owt_special_tokens_path, "rb") as f:
        owt_special_tokens = pickle.load(f)

    ts_tokenizer = Tokenizer.from_files(
        "results/bpe_train_tiny_stories/vocab.pkl",
        "results/bpe_train_tiny_stories/merges.pkl",
        ts_special_tokens
    )
    
    owt_tokenizer = Tokenizer.from_files(
        "results/bpe_train_owt/vocab.pkl",
        "results/bpe_train_owt/merges.pkl",
        owt_special_tokens
    )

    for desc, input_path, output_path,tokenizer in [
            ("Encoding TS valid","data/TinyStoriesV2-GPT4-valid.txt", "results/encode_ts_valid", ts_tokenizer),
            ("Encoding TS train","data/TinyStoriesV2-GPT4-train.txt", "results/encode_ts_train", ts_tokenizer),
            ("Encoding OWT valid","data/owt_valid.txt", "results/encode_owt_valid", owt_tokenizer),
            ("Encoding OWT train","data/owt_train.txt", "results/encode_owt_train", owt_tokenizer),
        ]:
        with open(input_path,"rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

        print(f"{desc} ({file_size / (1024 ** 3):,.3f} GB of text)")
        t0 = time.perf_counter()
        nids = sum(n for _,_,n in encode_file_to_chunks(
            tokenizer,
            input_path,
            output_path,
        ))
        tot = time.perf_counter() - t0

        print(f"Encoded {nids:,} tokens in {tot:,.3f} seconds: {round(nids/tot):,} tk/s. Total disk space: {get_dir_size(output_path) / (1024 * 1024):,.2f} MB.")
        print()
    

if __name__ == "__main__":
    print("Parts (a) and (b):")
    print()
    parts_a_and_b()
    print()
    print()

    print("Part (c):")
    print()
    part_c()
    print()
    print()

    print("Part (d)")
    print()
    part_d()
