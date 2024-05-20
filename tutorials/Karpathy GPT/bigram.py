import torch
import torch.nn as nn
from torch.nn import functional as F


from data import read_data


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Grab the last prediction from all the batches with all the channels
            probs = F.softmax(logits, dim = 1)
            pred = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, pred), dim = 1)
        
        return idx

if __name__ == '__main__':
    get_batch, encode, decode = read_data('input.txt')
    STEPS = 10_000
    m = BigramLanguageModel(65)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for _ in range(STEPS):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    valx, valy = get_batch('test')
    logits, loss = m(valx, valy)
    print(f"Validation Loss (Bigram): {loss:2f}")
    print(f"Example Generation:")
    print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))