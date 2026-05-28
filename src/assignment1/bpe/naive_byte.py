from typing import Self, List, Tuple, Dict
from src.assignment1.bpe.common import Tokenizer
from collections import Counter, defaultdict

class NaiveByteTokenizer(Tokenizer):
    @staticmethod
    def train(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes,bytes]], List[str]]:

        with open(input_path, "rb") as f:
            input_text = f.read()

        splits = input_text.split()
        
        # Step 1: Count the frequency of each byte in the input text
        frequency_table = Counter(splits)
        
        # Step 2: Count the frequency of each pair of bytes in the input text
        pairs = Counter([(splits[i], splits[i+1]) for i in range(len(splits)-1)])

        return {}, [], special_tokens

    