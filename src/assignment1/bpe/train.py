from typing import List, Iterable, BinaryIO, Dict, Tuple, Optional
import regex as re
from collections import Counter, defaultdict
import os
from multiprocessing import Pool

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> list[int]:
    assert special_tokens, "Need at least one special token to align boundaries"

    pattern = re.compile(b"|".join(re.escape(st.encode()) for st in sorted(special_tokens,key=len,reverse=True)))
    max_tok_len = max(len(st.encode()) for st in special_tokens)
    overlap = max_tok_len - 1

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        # use a small overlap for each mini chunk to avoid missing special tokens which fall on boundaries
        pos = chunk_boundaries[bi]
        file.seek(pos)
        carry = b""
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break
            buf = carry + mini_chunk
            m = pattern.search(buf)
            if m is not None:
                chunk_boundaries[bi] = pos - len(carry) + m.start()
                break
            carry = buf[-overlap:] if overlap > 0 else b""
            pos += len(mini_chunk)

    return sorted(set(chunk_boundaries))


def process_chunk(input_path, start, end, special_tokens, pretokenizer_pattern):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    delim = "|".join(re.escape(st) for st in special_tokens)
    pieces = re.split(delim, chunk)
    pretokens = [
        ptk.group()
        for piece in pieces
        for ptk in re.finditer(pretokenizer_pattern, piece)
    ]
    return Counter(pretokens)


def pretokenize_for_training(
    input_path: str,
    special_tokens: List[str],
    pretokenizer_pattern: str = r"...",
    num_processes: int = 4,
    desired_num_chunks: Optional[int] = None,
) -> Dict[Tuple[bytes, ...], int]:
    if desired_num_chunks is None:
        desired_num_chunks = num_processes 

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens)

    args = [
        (input_path, start, end, special_tokens, pretokenizer_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, args)

    out = defaultdict(int)
    for result in results:
        for k, v in result.items():
            out[tuple(bytes([b]) for b in k.encode())] += v
    return out


def pretokenize_for_training_slow(
        input_path: str,
        special_tokens: List[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ) -> Dict[Tuple[bytes,...],int]:
    with open(input_path,'r') as f:
        text = f.read()

    if special_tokens:
        delim = f"{"|".join(re.escape(st) for st in special_tokens)}"
        pieces = re.split(delim,text)
    else:
        pieces = [text]
    
    pretokens = [ptk.group() for piece in pieces for ptk in re.finditer(pretokenizer_pattern,piece)]
    
    return {tuple(bytes([b]) for b in k.encode()):v for k,v in Counter(pretokens).items()}


def train_bpe_slow(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ):
    vocab = {id:bytes([id]) for id in range(256)}

    for st in special_tokens:
        vocab[len(vocab)] = st.encode()

    frequencies = pretokenize_for_training_slow(input_path,special_tokens,pretokenizer_pattern)

    merges = []
    
    while len(vocab) < vocab_size:
        pairs = defaultdict(lambda: 0)
        for ptk, n in frequencies.items():
            for i in range(1,len(ptk)):
                pairs[(ptk[i-1],ptk[i])] += n
            
        if not pairs:
            break
        to_merge,_ = max(pairs.items(),key=lambda x:(x[1],x[0]))
        merged = to_merge[0] + to_merge[1]
        pairs.pop(to_merge)
        
        vocab[len(vocab)] = merged
        
        new_frequencies = frequencies.copy()
        for ptk,n in frequencies.items():
            i = 1
            while i < len(ptk):
                if (ptk[i-1],ptk[i]) == to_merge:
                    new_frequencies.pop(ptk)
                    ptk = ptk[:i-1] + (merged,) + ptk[i+1:]
                    new_frequencies[ptk] = n
                else:
                    i += 1
        
        frequencies = new_frequencies
        merges.append(to_merge)
    
    return vocab, merges


def merge_pretoken(ptk: Tuple[bytes,...], to_merge: Tuple[bytes,bytes]) -> Tuple[bytes,...]:
    merged = to_merge[0] + to_merge[1]
    out = []
    i = 0
    while i < len(ptk):
        if i+1 == len(ptk):
            out.append(ptk[-1])
            break
        a = ptk[i]
        b = ptk[i+1]
        if (a,b) == to_merge:
            out.append(merged)
            i += 2
        else:
            out.append(a)
            i += 1
    return tuple(out)

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ):
    vocab = {id:bytes([id]) for id in range(256)}

    for st in special_tokens:
        vocab[len(vocab)] = st.encode()

    if special_tokens:
        frequencies = pretokenize_for_training(input_path,special_tokens,pretokenizer_pattern)
    else:
        frequencies = pretokenize_for_training_slow(input_path,special_tokens,pretokenizer_pattern)
    
    merges = []

    pairs = defaultdict(lambda: 0)
    inverted_index = defaultdict(lambda: set())
    for old_ptk, n in frequencies.items():
        for i in range(1,len(old_ptk)):
            pair = (old_ptk[i-1],old_ptk[i])
            pairs[pair] += n
            inverted_index[pair].add(old_ptk)

    while len(vocab) < vocab_size:
        if not pairs:
            break

        to_merge,_ = max(pairs.items(),key=lambda x:(x[1],x[0]))
        merged = to_merge[0] + to_merge[1]
        pairs.pop(to_merge)

        ptks_to_update = inverted_index[to_merge].copy()

        for old_ptk in ptks_to_update:
            n = frequencies.pop(old_ptk)
            new_ptk = merge_pretoken(old_ptk,to_merge)
            old_pairs = Counter(zip(old_ptk[:-1],old_ptk[1:]))
            new_pairs = Counter(zip(new_ptk[:-1],new_ptk[1:]))

            for p,c in old_pairs.items():
                pairs[p] -= n * c
                if pairs[p] <= 0:
                    del pairs[p]
            for p,c in new_pairs.items():
                pairs[p] += n * c
            
            frequencies[new_ptk] = frequencies.get(new_ptk,0) + n
            
            for pair in old_pairs:
                inverted_index[pair].remove(old_ptk)
            for pair in new_pairs:
                inverted_index[pair].add(new_ptk)

        vocab[len(vocab)] = merged
        merges.append(to_merge)
    
    # print(merges)
    return vocab, merges


if __name__ == "__main__":
    # train_bpe('./data/train_eg.txt',1000,["<|endoftext|>"])
    train_bpe('./data/train_eg.txt',500,[" ","<|endoftext|>"], pretokenizer_pattern=r".+")