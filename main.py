
from src.tokenization import SpaceTokenizer, WordPieceTokenizer, TokenizerBERT

with open("vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

input_text = ["Dr. Agne works at N.A.S.A.", "This kiosk is AMAZING!!!"]
punctuation_list = ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', "'", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`', '|']

space_token = SpaceTokenizer(input_text, punctuation_list, vocab)
encoded = space_token.encode()
print(encoded.shape)

wp_token = WordPieceTokenizer(input_text, punctuation_list, vocab)
wp_encoded = wp_token.encode()
print(wp_token.decode(wp_encoded))

bert_token = TokenizerBERT()
bert_encoded = bert_token.encode(input_text)
print(bert_encoded['input_ids'].shape)
