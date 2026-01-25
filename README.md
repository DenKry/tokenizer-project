# Tokenizer

Custom tokenizer implementations including Space-based, WordPiece, and BERT tokenizers.

## Project Structure

```
tokenizer/
├── src/
│   └── tokenize/
│       ├── tokenizers.py                   # SpaceTokenizer, WordPieceTokenizer
│       └── pretrained_bert_tokenizer.py    # TokenizerBERT wrapper
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
