# My Implementation of CS336 Assignment 1: Basics

This repository is my personal, from-scratch implementation of Stanford's
[CS336](https://stanford-cs336.github.io/) Assignment 1 (Basics): building a
Transformer language model from the ground up.
I am not a Stanford student; I'm working through the publicly available course
materials on my own, purely to learn.

My own code lives under [`src/assignment1/`](./src/assignment1/), with the
scripts that drive it under [`scripts/`](./scripts/). The assignment handout
([cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)), the tests, and
the starter file [pretokenization_example.py](./cs336_basics/pretokenization_example.py) are the course's original materials.

In keeping with the course's AI policy, all of the implementation code here is
written by me. None of it is written or autocompleted by AI. I only use an AI
assistant the way the policy permits: to occasionally ask high-level conceptual
or low-level documentation questions when I am truly stuck. The implementations
are my own work.

---

The original assignment setup instructions follow.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

