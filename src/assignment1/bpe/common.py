from typing import Iterable, Self, Dict, List, Tuple, Optional, Iterator
import regex as re
from itertools import chain
from collections import defaultdict

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
        self.token_to_id = {v:k for k,v in vocab}
        self.merges = merges

        self.special_tokens = sorted(special_tokens,key=len,reverse=True)
        self.pretokenizer_pattern = pretokenizer_pattern
        
        return

    def pretokenize(self, text:str) ->  Iterable[Tuple[str,bool]]:
        if self.special_tokens:
            delim = rf"({"|".join(re.escape(st) for st in self.special_tokens)})"
            pieces = re.split(delim,text)
        else:
            pieces = [text]
        
        for piece in pieces:
            if piece in self.special_tokens:
                yield (piece,True)
            else:
                for pretk in re.finditer(self.pretokenizer_pattern,piece):
                    yield (pretk.group(),False)

    def encode_pretoken(self, ptk: str) -> Iterable[int]:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for ptk,is_special in self.pretokenize(text):
                if is_special:
                    yield self.token_to_id[ptk.encode()]
                else:
                    for id in self.encode_pretoken(ptk):
                        yield id
        
    def decode(self, ids: List[int]):
        text = bytes(ids).decode("utf-8")
        return text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[List[int]]:
        for text in iterable:
            ids = self.encode(text)
            yield ids

            