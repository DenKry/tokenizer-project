
import torch
from transformers import BertTokenizer, BertModel

class TokenizerBERT:
    def __init__(self, model_name="bert-base-uncased", add_special_tokens=True, 
                 padding=True, truncation=True, return_tensors='pt'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=self.add_special_tokens,
                            padding=self.padding, truncation=self.truncation,
                            return_tensors=self.return_tensors)
    
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_embeddings(self, text):
        inputs = self.encode(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state, outputs.pooler_output


