import regex as re
from collections import Counter, defaultdict
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
from src.assignment1.bpe.utils import find_chunk_boundaries
from pathlib import Path

def process_chunk(
        input_path: str | Path,
        start: int,
        end: int,
        special_tokens: list[str],
        pretokenizer_pattern: str,
    ):
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

def process_chunk_star(arg: tuple[str|Path,int,int,list[str],str]):
    return process_chunk(*arg)

def pretokenize_for_training(
    input_path: str | Path,
    special_tokens: list[str],
    pretokenizer_pattern: str = r"...",
    num_processes: int = 8,
    desired_num_chunks: int | None = None,
) -> dict[tuple[bytes, ...], int]:
    if desired_num_chunks is None:
        desired_num_chunks = num_processes * 4
    
    print(f"Pretokenizing with {num_processes} processes and {desired_num_chunks} desired chunks.")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, special_tokens, desired_num_chunks=desired_num_chunks)
    
    sizes = [b - a for a, b in zip(boundaries, boundaries[1:])]
    print(f"found {len(sizes)} chunks; min/max chunk sizes: {min(sizes)/1e6:.3f} / {max(sizes)/1e6:.3f} MB")

    args = [
        (input_path, start, end, special_tokens, pretokenizer_pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    results = process_map(
        process_chunk_star,
        args,
        max_workers=num_processes,
        chunksize=10,
    )
    
    print("Combining each chunk's pretoken frequencies")

    out = results[0]
    for result in results[1:]:
        out += result
    
    out = {tuple(bytes([b]) for b in k.encode()):v for k,v in out.items()}

    print(f"Pretokenization finished")
    return out


def pretokenize_for_training_slow(
        input_path: str | Path,
        special_tokens: list[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ) -> dict[tuple[bytes,...],int]:
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
        special_tokens: list[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = {id:bytes([id]) for id in range(256)}

    for st in special_tokens:
        vocab[len(vocab)] = st.encode()

    frequencies = pretokenize_for_training_slow(input_path,special_tokens,pretokenizer_pattern)

    merges: list[tuple[bytes,bytes]] = []
    
    with tqdm(total=vocab_size, initial=len(vocab)) as pbar:
        while len(vocab) < vocab_size:
            pairs: defaultdict[tuple[bytes,bytes],int] = defaultdict(lambda: 0)
            for ptk, n in frequencies.items():
                for i in range(1,len(ptk)):
                    pairs[(ptk[i-1],ptk[i])] += n
                
            if not pairs:
                break
            to_merge,_ = max(pairs.items(),key=lambda x:(x[1],x[0]))
            merged = to_merge[0] + to_merge[1]
            pairs.pop(to_merge)
            
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
            vocab[len(vocab)] = merged
            merges.append(to_merge)

            pbar.update(1)
    
    return vocab, merges


def merge_pretoken(ptk: tuple[bytes,...], to_merge: tuple[bytes,bytes]) -> tuple[bytes,...]:
    merged: bytes = to_merge[0] + to_merge[1]
    out: list[bytes] = []
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
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str],
        pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        num_processes: int = 6,
        desired_num_chunks: int | None = None,
    ) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = {id:bytes([id]) for id in range(256)}

    for st in special_tokens:
        vocab[len(vocab)] = st.encode()

    if special_tokens:
        frequencies = pretokenize_for_training(
            input_path,
            special_tokens,
            pretokenizer_pattern,
            num_processes,
            desired_num_chunks
        )
    else:
        frequencies = pretokenize_for_training_slow(
            input_path,
            special_tokens,
            pretokenizer_pattern
        )
    
    merges: list[tuple[bytes,bytes]] = []

    pairs: defaultdict[tuple[bytes,bytes],int] = defaultdict(lambda: 0)
    inverted_index: defaultdict[tuple[bytes,bytes],set[tuple[bytes,...]]] = defaultdict(lambda: set())
    for old_ptk, n in frequencies.items():
        for i in range(1,len(old_ptk)):
            pair = (old_ptk[i-1],old_ptk[i])
            pairs[pair] += n
            inverted_index[pair].add(old_ptk)

    print(f"Starting merging, current vocab length {len(vocab)}/{vocab_size}")
    with tqdm(total=vocab_size, initial=len(vocab)) as pbar:
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
            pbar.update(1)
    
    return vocab, merges