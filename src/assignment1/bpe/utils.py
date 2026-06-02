from typing import BinaryIO, Union, List, Optional
import os
import regex as re
from pathlib import Path
import random

def find_chunk_boundaries(
    file: BinaryIO,
    special_tokens: list[str],
    *,
    desired_num_chunks: Optional[int] = None,
    desired_chunk_size_mb: Optional[float] = 1.0,
    with_ending_tokens=True,
) -> list[int]:
    if (desired_num_chunks is None) == (desired_chunk_size_mb is None):
        raise ValueError("Exactly one of desired_num_chunks and desired_chunk_size must be None")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if desired_chunk_size_mb is not None:
        desired_num_chunks = int((file_size / (1024 ** 2)) / desired_chunk_size_mb)

    assert special_tokens, "Need at least one special token to align boundaries"

    pattern = re.compile(b"|".join(re.escape(st.encode()) for st in sorted(special_tokens,key=len,reverse=True)))
    max_tok_len = max(len(st.encode()) for st in special_tokens)
    overlap = max_tok_len - 1


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
                bd = pos - len(carry)
                bd += m.end() if with_ending_tokens else m.start()
                chunk_boundaries[bi] = bd
                break
            carry = buf[-overlap:] if overlap > 0 else b""
            pos += len(mini_chunk)

    return sorted(set(chunk_boundaries))


def sample_documents(dataset_path: Union[Path,str], n: int = 10, special_tokens: List[str] = ["<|endoftext|>"], seed = None) -> List[str]:
    with open(dataset_path,"rb") as f:
        f.seek(0,os.SEEK_END)
        file_size = f.tell()
        stories = []
        chunk_size = 1 << 12
        pattern = re.compile(b"|".join(re.escape(st.encode()) for st in special_tokens))
        random.seed(seed)
        
        while len(stories) < n:
            start_pos = random.randint(0,file_size - 1)
            f.seek(start_pos)
            
            cur = b""
            
            # find start
            while not cur:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                m = pattern.search(chunk)
                if m is not None:
                    cur = chunk[m.end():]
            
            # find end
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                cur += chunk
                m = pattern.search(cur)
                if m is not None:
                    story = cur[:m.end()].decode('utf-8')
                    stories.append(story)
                    break
        
        return stories