
import torch

class SpaceTokenizer:
    def __init__(self, input_text, punctuation_list, vocabulary, max_length=128, 
                 truncation=True, padding=True, return_dict=False, return_tensors='pt'):
        self.input = input_text
        self.punct = punctuation_list
        self.vocab = vocabulary
        self.max_len = max_length
        self.truncation = truncation
        self.padding = padding
        self.return_dict = return_dict
        self.return_tensors = return_tensors

    def vocab_dict(self):
        token_to_id = id_to_token = {}
        for id in range(len(self.vocab)):
            token_to_id[self.vocab[id]] = id
            id_to_token[id] = self.vocab[id]  
        return token_to_id, id_to_token

    def text_preproc(self):
        out = []
        for seq in self.input:
            seq_out = seq.lower()

            for sign in self.punct:
                if sign in seq_out:
                    seq_out = seq_out.replace(sign, f" {sign} ")

            double_space = "  "
            while double_space in seq_out:
                seq_out = seq_out.replace(double_space, " ")
            out.append(seq_out.strip())
        return out
    
    def tokenize(self):
        text = self.text_preproc()

        out = []
        for seq in text:
            seq = seq.split()

            for token in seq:
                if token not in self.vocab:
                    index = seq.index(token)
                    seq[index] = "[UNK]"
            out.append(seq)
        return out
    
    def encode(self, return_dict=None, return_tensors=None):
        if return_dict is None:
            return_dict = self.return_dict
        if return_tensors is None:
            return_tensors = self.return_tensors
        
        tokenized_text = self.tokenize()
        token_to_id, _ = self.vocab_dict()

        out = []
        for seq in tokenized_text:
            encoded_seq = [token_to_id["[CLS]"]]
            for token in seq:
                encoded_seq.append(token_to_id[token])
            encoded_seq.append(token_to_id["[SEP]"])
            out.append(encoded_seq)
        
        max_length = self.get_max_length(out)
        truncated = self.truncate(out, max_length)
        padded = self.padd(truncated, max_length)
        
        if not return_dict:
            if return_tensors == 'pt':
                return torch.tensor(padded)
            else:
                return padded
        
        attn_mask = self.create_attention_mask(padded)
        token_type_ids = self.create_token_type_ids(padded)
        
        out = {
            'input_ids': padded,
            'attention_mask': attn_mask,
            'token_type_ids': token_type_ids
        }
        
        if return_tensors == 'pt':
            out['input_ids'] = torch.tensor(out['input_ids'])
            out['attention_mask'] = torch.tensor(out['attention_mask'])
            out['token_type_ids'] = torch.tensor(out['token_type_ids'])
        
        return out
    
    def get_max_length(self, sequences):
        if self.max_len is not None:
            return self.max_len
        return max(len(seq) for seq in sequences)
    
    def truncate(self, sequences, max_length):
        truncated = []
        for seq in sequences:
            if self.truncation and len(seq) > max_length:
                seq = seq[:max_length-1] + [seq[-1]]
            truncated.append(seq)
        return truncated
    
    def padd(self, sequences, max_length):
        padded = []
        for seq in sequences:
            if self.padding and len(seq) < max_length:
                padding_tokens = [0] * (max_length - len(seq))
                seq = seq + padding_tokens
            padded.append(seq)
        return padded
    
    def decode(self, token_ids=None, skip_special_tokens=True):
        token_to_id, id_to_token = self.vocab_dict()

        if token_ids is None:
            token_ids = self.seq_pad()
        
        if isinstance(token_ids, dict):
            token_ids = token_ids['input_ids']
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        special_ids = [token_to_id.get("[PAD]"), token_to_id.get("[CLS]"), token_to_id.get("[SEP]")]
        
        if token_ids and isinstance(token_ids[0], int):
            decoded_seq = []
            for id in token_ids:
                if skip_special_tokens and id in special_ids:
                    continue
                decoded_seq.append(id_to_token[id])
            return decoded_seq
        
        out = []
        for seq in token_ids:
            decoded_seq = []
            for id in seq:
                if skip_special_tokens and id in special_ids:
                    continue
                decoded_seq.append(id_to_token[id])
            out.append(decoded_seq)
        return out
    
    def seq_pad(self):
        encoded_text = self.encode(return_dict=False, return_tensors='')
        max_length = self.get_max_length(encoded_text)
        
        truncated = self.truncate(encoded_text, max_length)
        padded = self.padd(truncated, max_length)
        
        return padded
    
    def create_attention_mask(self, padded_text):
        token_to_id, _ = self.vocab_dict()

        out = []
        for seq in padded_text:
            for id in range(len(seq)):
                if seq[-id-1] == token_to_id["[SEP]"]:
                    break
            
            attn_seq = [1] * (len(seq) - id) + [0] * id
            out.append(attn_seq)

        return out
    
    def create_token_type_ids(self, padded_text):
        token_to_id, _ = self.vocab_dict()
        sep_id = token_to_id["[SEP]"]
        
        out = []
        for seq in padded_text:
            type_seq = []
            current_type = 0
            
            for token_id in seq:
                type_seq.append(current_type)
                
                if token_id == sep_id and current_type == 0:
                    current_type = 1
            
            out.append(type_seq)
        
        return out


class WordPieceTokenizer(SpaceTokenizer):
    def __init__(self, input_text, punctuation_list, vocabulary, max_length=128, 
                 truncation=True, padding=True, return_dict=False, return_tensors='pt'):
        super().__init__(input_text, punctuation_list, vocabulary, max_length, truncation, padding, return_dict, return_tensors)

    def tokenize(self):
        return self.word_piece_tokenize()

    def reccursive_token_check(self, token, prefix=""):
        if token in self.vocab:
            return [prefix + token]
        
        for ch in range(1, len(token)):
            if token[:-ch] in self.vocab:
                result = [prefix + token[:-ch]]
                remainder = self.reccursive_token_check(token[-ch:], prefix="##")
                result.extend(remainder)
                return result
        return ["[UNK]"]
    
    def word_piece_tokenize(self):
        text = self.text_preproc()
        
        out = []
        for seq in text:
            seq = seq.split()

            tokenized_seq = []
            for token in seq:
                token_res = self.reccursive_token_check(token)
                tokenized_seq.extend(token_res)

            out.append(tokenized_seq)
        return out
    
    def decode(self, token_ids=None, skip_special_tokens=True):
        token_to_id, id_to_token = self.vocab_dict()
        
        if token_ids is None:
            encoded_text = self.seq_pad()
        else:
            if isinstance(token_ids, dict):
                encoded_text = token_ids['input_ids']
            else:
                encoded_text = token_ids
        
        if isinstance(encoded_text, torch.Tensor):
            encoded_text = encoded_text.tolist()
        
        special_ids = [token_to_id.get("[PAD]"), token_to_id.get("[CLS]"), token_to_id.get("[SEP]")]

        out = []
        for seq in encoded_text:
            decoded_seq = []
            for id in seq:
                if skip_special_tokens and id in special_ids:
                    continue
                token = id_to_token[id]
                
                if token.startswith("##"):
                    if decoded_seq:
                        decoded_seq[-1] += token[2:]
                else:
                    decoded_seq.append(token)
            
            out.append(decoded_seq)
        return out
    