from typing import Iterable, Self, Dict, List, Tuple, Optional, Iterator
import regex as re
from itertools import chain
from collections import defaultdict
import pickle
from pathlib import Path
from heapq import heappush, heappop

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
                # print(f'scanning {ptk}')
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
                                # print(f'\tfound new smallest mergeable pair ({a},{b}) with idx {idx} at index {i}')
                                min_idx = idx
                                to_merge = a+b
                                replace_indices = [i]
                            elif idx == min_idx:
                                # print(f'\tfound smallest mergeable pair ({a},{b}) with idx {idx} at index {i}')
                                replace_indices.append(i)
                        if len(replace_indices) == 1:
                            # print(f'\treplacing with {to_merge} at indices {",".join(map(str,replace_indices))}')
                            i = replace_indices[0]
                            new_ptk = ptk[:i] + (to_merge,) + ptk[i+2:]
                            # print(f"\t\treplaced {ptk} with {new_ptk}")
                            ptk = new_ptk
                        elif len(replace_indices) > 1:
                            # print(f'\treplacing with {to_merge} at indices {",".join(map(str,replace_indices))}')
                            first = replace_indices[0]
                            last = replace_indices[-1]
                            new_ptk = ptk[:first] + (to_merge,)
                            for i,j in zip(replace_indices,replace_indices[1:]):
                                new_ptk += ptk[i+2:j] + (to_merge,)
                            new_ptk += ptk[last+2:]
                            # print(f"\t\treplaced {ptk} with {new_ptk}")
                            ptk = new_ptk
                        else:
                            # print(f"\t\tscanned {ptk} found nothing, yielding")
                            for t in ptk:
                                yield(self.bytes_to_id[t])
                            break
                        
                

    def decode(self, ids: List[int]) -> str:
        bts = b"".join(map(self.vocab.get,ids))
        return bts.decode('utf-8',errors='replace')
