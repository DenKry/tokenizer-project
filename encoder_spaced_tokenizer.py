
import torch

with open("vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

input_text = ["Dr. Agne works at N.A.S.A.", "This kiosk is AMAZING!!!"]
punctuation_list = ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', "'", '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`', '|']

class SpaceTokenizer:
    def __init__(self, input_text, punctuation_list, vocabulary):
        self.input = input_text
        self.punct = punctuation_list
        self.vocab = vocabulary

    def vocab_dict(self):
        token_to_id = id_to_token = {}
        for id in range(len(self.vocab)):
            token_to_id[vocab[id]] = id
            id_to_token[id] = vocab[id]  
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
    
    def encode(self):
        tokenized_text = self.tokenize()
        token_to_id, _ = self.vocab_dict()

        out = []
        out.append(token_to_id["[CLS]"])
        for seq in tokenized_text:
            for token in seq:
                out.append(token_to_id[token])
            out.append(token_to_id["[SEP]"])

        return out
    
    def decode(self):
        encoded_text = self.encode()
        _, id_to_token = self.vocab_dict()

        out = []
        for id in encoded_text:
            out.append(id_to_token[id])
        return out
    
    def return_tensor(self):
        encoded_text = self.encode()
        out = torch.tensor(encoded_text)
        return out
        
tokenization = MyTokenizer(input_text, punctuation_list, vocab)

# print(tokenization.text_preproc(), sep="\n")
# print(tokenization.tokenize(), sep="\n")
# print(tokenization.encode(), sep="\n")
# print(tokenization.decode(), sep="\n")
print(tokenization.return_tensor(), tokenization.return_tensor().shape, tokenization.return_tensor().dim,sep="\n")