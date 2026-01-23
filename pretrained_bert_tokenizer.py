
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = ["Dr. Agne works at N.A.S.A.","This kiosk is AMAZING!!!"]
tokens = tokenizer.tokenize(text)
tokens_id = tokenizer.convert_tokens_to_ids(tokens)
back_tokens = tokenizer.convert_ids_to_tokens(tokens_id)
encoded = tokenizer.encode(text, add_special_tokens=True)
decoded = tokenizer.decode(encoded)
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
print(f"Pooler output shape: {outputs.pooler_output.shape}")

# print(tokens, tokens_id, back_tokens, tokens == back_tokens, encoded, decoded, inputs, sep="\n")
# print(input, sep="\n")


