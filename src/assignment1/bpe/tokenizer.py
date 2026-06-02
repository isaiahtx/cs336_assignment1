from __future__ import annotations
from src.assignment1.bpe.utils import find_chunk_boundaries
import os
from typing import Iterable, Self, Dict, List, Tuple, Optional, Iterator
import regex as re
import pickle
from tqdm.contrib.concurrent import process_map
import numpy as np
from pathlib import Path


class Tokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes,bytes]]
    special_tokens: List[str]

    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: Optional[List[str]] = None,
            pretokenizer_pattern: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ):
        self.vocab = vocab
        self.merges = merges
        self.merges_to_idx = {(a,b):i for i,(a,b) in enumerate(merges)}
        self.special_tokens = None if special_tokens is None else sorted(special_tokens,key=len,reverse=True)
        self.pretokenizer_pattern = pretokenizer_pattern
        self.bytes_to_id = {v:k for k,v in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
        pretokenizer_pattern: Optional[str] = None,
    ) -> Self:
        with open(vocab_filepath,'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath,'rb') as f:
            merges = pickle.load(f)
        
        args = [vocab, merges, special_tokens]
        if pretokenizer_pattern is not None:
            args.append(pretokenizer_pattern)

        return Tokenizer(*args)
    
    def pretokenize(self, text: str) -> Iterator[Tuple[str,bool]]:
        if self.special_tokens:
            delim = re.compile("(" + "|".join(re.escape(st) for st in self.special_tokens) + ")")
            pieces = re.split(delim,text)
        else:
            pieces = [text]
        
        pat = re.compile(self.pretokenizer_pattern)

        for piece in pieces:
            if self.special_tokens is not None and piece in self.special_tokens:
                yield (piece,True)
            else:
                for ptk in map(lambda m:m.group(),re.finditer(pat, piece)):
                    yield (ptk, False)
    
    def encode(self, text: str) -> List[int]:
        return list(self.encode_iterable([text]))
    


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for ptk, is_special in self.pretokenize(text):
                ptk = ptk.encode('utf-8')
                if is_special:
                    yield self.bytes_to_id[ptk]
                else:
                    ptk = tuple(bytes([b]) for b in ptk)
                    while True:
                        replace_indices = []
                        min_idx = None
                        to_merge = None
                        for i in range(len(ptk) - 1):
                            a = ptk[i]; b = ptk[i+1]
                            idx = self.merges_to_idx.get((a,b))
                            if idx is None:
                                continue
                            if min_idx is None or idx < min_idx:
                                min_idx = idx
                                to_merge = a+b
                                replace_indices = [i]
                            elif idx == min_idx:
                                replace_indices.append(i)
                        if len(replace_indices) == 1:
                            i = replace_indices[0]
                            new_ptk = ptk[:i] + (to_merge,) + ptk[i+2:]
                            ptk = new_ptk
                        elif len(replace_indices) > 1:
                            first = replace_indices[0]
                            last = replace_indices[-1]
                            new_ptk = ptk[:first] + (to_merge,)
                            for i,j in zip(replace_indices,replace_indices[1:]):
                                new_ptk += ptk[i+2:j] + (to_merge,)
                            new_ptk += ptk[last+2:]
                            ptk = new_ptk
                        else:
                            for t in ptk:
                                yield(self.bytes_to_id[t])
                            break
                

    def decode(self, ids: List[int]) -> str:
        bts = b"".join(map(self.vocab.get,ids))
        return bts.decode('utf-8',errors='replace')


def _process_chunk(args: Tuple["Tokenizer",str,str,int,int,int]):
    tokenizer, input_path, output_path, idx, start, end = args
    with open(input_path,"rb") as f:
        f.seek(start)
        ids = tokenizer.encode(f.read(end-start).decode('utf-8'))
        npids = np.array(ids,dtype=np.uint16)
        output_file = Path(output_path) / f"{idx}.npy"
        np.save(output_file,npids)
        return (idx,output_file,len(npids))


def encode_file_to_chunks(
    tokenizer: "Tokenizer",
    input_path: str,
    output_path: str,
    *,
    desired_num_chunks: Optional[int] = None,
    desired_chunk_size_mb: Optional[float] = 1.0,
    num_workers: int=16
) -> List[Path]:
    Path(output_path).mkdir(parents=True,exist_ok=False)

    with open(input_path,"rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f,
            special_tokens=tokenizer.special_tokens,
            desired_num_chunks=desired_num_chunks,
            desired_chunk_size_mb=desired_chunk_size_mb,
            with_ending_tokens=True
        )
    chunksize = 10#min(max(20,round((len(chunk_boundaries) - 1) / (num_workers * 10))),100)
    print(f"\tEncoding {len(chunk_boundaries) - 1} chunks across {num_workers} workers, in batches of {chunksize} chunks per worker")
    
    return process_map(
        _process_chunk,
        [
            (tokenizer,input_path,output_path,idx,start,end) 
            for idx,(start,end) in enumerate(zip(chunk_boundaries,chunk_boundaries[1:]))
        ],
        max_workers=num_workers,
        chunksize=chunksize,
    )