from pathlib import Path
import token
from src.assignment1.bpe.tokenizer import Tokenizer
import pickle
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

def main():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="./tests/fixtures/gpt2_vocab.json",
        merges_path="./tests/fixtures/gpt2_merges.txt",
    )
    
    merges = tokenizer.merges
    print([(i,a,b) for i,(a,b) in enumerate(merges) if a + b == b'icated'])
    print([(i,a,b) for i,(a,b) in enumerate(merges) if a + b == b'ated'])
    print()
    print()
    
    print(tokenizer.encode("dedicated"))
    print()
    print()

    print(b'icate' in tokenizer.vocab.values())
    print()
    print()
    # print(sorted(tokenizer.vocab.values()))

def main2():
    tokenizer_folder = Path('./results/bpe_train_tiny_stories')
    with open(tokenizer_folder / "special_tokens.pkl",'rb') as f:
        special_tokens = pickle.load(f)
    
    tkz = Tokenizer.from_files(
        vocab_filepath=tokenizer_folder / "vocab.pkl",
        merges_filepath=tokenizer_folder / "merges.pkl",
        special_tokens=special_tokens,
    )
    
    original = "the cat ate"
    encoded = tkz.encode(original)
    decoded = tkz.decode(encoded)
    print(f"original: {original}")
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")
    print(f"match: {original == decoded}")
    print()
    
    original = "I died — then I exploded!!1!!1 🫠"
    encoded = tkz.encode(original)
    decoded = tkz.decode(encoded)
    print(f"original: {original}")
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")
    print(f"match: {original == decoded}")
    print()
    
    original = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded = tkz.encode(original)
    decoded = tkz.decode(encoded)
    print(f"original: {original}")
    print(f"encoded: {encoded}")
    print(f"decoded: {decoded}")
    print(f"match: {original == decoded}")
    print()

if __name__ == "__main__":
    main()