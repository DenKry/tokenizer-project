# Tokenizer

Custom tokenizer implementations including Space-based, WordPiece, and BERT tokenizers.

## Project Structure

```
tokenizer/
├── src/
│   └── tokenization/
│       ├── bpe_tokenizer.py                # BPETokenizer
│       ├── pretrained_bert_tokenizer.py    # TokenizerBERT wrapper
│       └── scratch_bert_tokenizer.py       # SpaceTokenizer, WordPieceTokenizer
├── main.py                                 # Usage examples
├── vocab.txt                               # BERT vocabulary file
└── README.md
```

## Features

### SpaceTokenizer
Space-based tokenization with customizable settings.

**Features:**
- Text preprocessing (lowercase, punctuation spacing)
- UNK token handling
- Configurable padding and truncation
- Attention mask and token type IDs generation
- PyTorch tensor support


**Parameters:**
- `input_text` (list[str]): Input texts to tokenize
- `punctuation_list` (list[str]): Punctuation marks to separate
- `vocabulary` (list[str]): Token vocabulary
- `max_length` (int, default=128): Maximum sequence length
- `truncation` (bool, default=True): Apply truncation
- `padding` (bool, default=True): Apply padding
- `return_dict` (bool, default=False): Return dictionary or tensor
- `return_tensors` (str, default='pt'): Return format ('pt' for PyTorch, None for lists)

**Methods:**
- `text_preproc()`: Preprocess text (lowercase, punctuation spacing)
- `tokenize()`: Split text into tokens
- `encode()`: Convert tokens to IDs with padding/truncation
- `decode()`: Convert IDs back to tokens
- `get_max_length()`: Get maximum sequence length
- `apply_truncation()`: Truncate sequences
- `apply_padding()`: Pad sequences
- `create_attention_mask()`: Generate attention masks
- `create_token_type_ids()`: Generate token type IDs

---

### WordPieceTokenizer
Subword tokenization using WordPiece algorithm (like BERT).

**Features:**
- Recursive subword splitting
- `##` prefix for continuation tokens
- Automatic merging on decode
- Inherits all SpaceTokenizer functionality


**WordPiece Algorithm:**
- Tries to match longest subword in vocabulary
- Splits unknown words into smaller known pieces
- Adds `##` prefix to continuation pieces
- Falls back to `[UNK]` if no split found

---

### BPETokenizer
Byte-Pair Encoding tokenizer trained from scratch on raw text.

**Features:**
- Trains directly on UTF-8 byte sequences — no external vocabulary needed
- Learns merge rules iteratively from the training corpus
- Configurable vocab size
- PyTorch tensor support on encode/decode


**Parameters:**
- `vocab_size` (int, default=276): Target vocabulary size (must be > 256)
- `return_tensors` (str, default='pt'): Return format ('pt' for PyTorch tensor, None for list)

**Methods:**
- `train(text)`: Learn BPE merge rules from a text string
- `encode(text, return_tensors=None)`: Encode text to token IDs
- `decode(ids, skip_special_tokens=True)`: Decode token IDs back to text
- `get_stats(ids)`: Count byte-pair frequencies in a token sequence
- `merge(ids, pair, idx)`: Apply a single merge rule to a token sequence

**BPE Algorithm:**
- Initialises vocabulary with all 256 raw bytes
- Repeatedly finds the most frequent adjacent byte pair
- Merges that pair into a new token and records the rule
- Continues until `vocab_size` is reached
- Encoding replays learned merges in the same order

**Example:**
```python
from src.tokenization.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train("hello world hello world")

encoded = tokenizer.encode("hello")       # torch.Tensor([...])
decoded = tokenizer.decode(encoded)       # "hello"

# List output
encoded_list = tokenizer.encode("hello", return_tensors=None)  # [...]
```

---

### TokenizerBERT
Wrapper around HuggingFace BERT tokenizer.

**Features:**
- Pre-trained BERT tokenizer and model
- Automatic subword tokenization
- Embedding extraction
- Simplified API


**Parameters:**
- `model_name` (str, default="bert-base-uncased"): HuggingFace model name
- `add_special_tokens` (bool, default=True): Add [CLS] and [SEP]
- `padding` (bool, default=True): Apply padding
- `truncation` (bool, default=True): Apply truncation
- `return_tensors` (str, default='pt'): Return format

**Methods:**
- `tokenize()`: Tokenize text into subwords
- `encode()`: Convert text to input IDs
- `decode()`: Convert IDs to text
- `get_embeddings()`: Extract BERT embeddings

---

## Output Formats

All tokenizers support multiple output formats:

```python
# 1. Tensor only (default)
encoded = tokenizer.encode()
# → torch.Tensor([2, 128])

# 2. List only
tokenizer = SpaceTokenizer(..., return_tensors=None)
encoded = tokenizer.encode()
# → [[101, 2054, 102, 0, ...], ...]

# 3. Dictionary with tensors
encoded = tokenizer.encode(return_dict=True, return_tensors='pt')
# → {'input_ids': Tensor, 'attention_mask': Tensor, 'token_type_ids': Tensor}

# 4. Dictionary with lists
tokenizer = SpaceTokenizer(..., return_dict=True, return_tensors=None)
encoded = tokenizer.encode()
# → {'input_ids': [[...]], 'attention_mask': [[...]], 'token_type_ids': [[...]]}
```

---

## Installation

```bash
pip install torch transformers
```

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers (for BERT tokenizer)
- vocab.txt file with vocabulary

---

## License

MIT
