import torch


class BPETokenizer:
    def __init__(self, vocab_size=276, return_tensors='pt'):
        self.vocab_size = vocab_size
        self.return_tensors = return_tensors
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        ids = list(text.encode("utf-8"))
        num_merges = self.vocab_size - 256

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        tokens = list(text.encode("utf-8"))

        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])

        if return_tensors == 'pt':
            return torch.tensor(tokens)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        return token_bytes.decode("utf-8", errors="replace")
