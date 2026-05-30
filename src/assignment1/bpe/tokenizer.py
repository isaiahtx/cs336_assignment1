from typing import Iterable, Self, Dict, List, Tuple, Optional, Iterator
import regex as re
from itertools import chain
from collections import defaultdict
import pickle
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
                        found = False
                        for idx,(a,b) in enumerate(self.merges):
                            i = 0
                            while i+1 < len(ptk):
                                if ptk[i] == a and ptk[i+1] == b:
                                    found = True
                                    ptk = ptk[:i] + (a + b,) + ptk[i+2:]
                                i += 1
                        if not found:
                            break
                    
                    for e in ptk:
                        yield self.bytes_to_id[e]

    def decode(self, ids: List[int]) -> str:
        bts = b"".join(map(self.vocab.get,ids))
        return bts.decode('utf-8',errors='replace')
