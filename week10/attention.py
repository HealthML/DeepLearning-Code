import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)


# This is a toy dataset
# in the exercise: our q: A, or Apple and the target will be Apple (A-->Apple) and so on
KEYS = ["A", "B", "C", "D", "E"]
VALS = ["apple", "banana", "cherry", "date", "elderberry"]

# Dummy Tokenization
# not required for you to understand this part right now, but worth looking at
itos = ["<pad>"] + KEYS + VALS
stoi = {s: i for i, s in enumerate(itos)}
vocab_size = len(itos)


def make_kv_batch(batch_size, n_pairs):
    T = 2 * n_pairs
    x = torch.empty(batch_size, T, dtype=torch.long)

    # pick random keys; value is determined by the key (A->apple, B->banana, ...)
    key_idx = torch.randint(0, len(KEYS), (batch_size, n_pairs))
    for b in range(batch_size):
        for i in range(n_pairs):
            k = key_idx[b, i].item()
            x[b, 2*i]   = stoi[KEYS[k]]
            x[b, 2*i+1] = stoi[VALS[k]]

    # query a random key position (even indices)
    pair_choice = torch.randint(0, n_pairs, (batch_size,))
    q = 2 * pair_choice

    # target is the value right after that key
    y = x[torch.arange(batch_size), q + 1]
    return x, q, y

class Attention(nn.Module):
    # Attention: that you can't get enough of... just kidding :)
    def __init__(self, vocab_size, d_model=64, d_k=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_k)

    def forward(self, x, q):
        """
        x: (B, T) token ids
        q: (B,) query positions (which token in each sequence will act as the query)
        returns:
          logits: (B, vocab_size)
          attn: (B, T)
        """
        B, T = x.shape

        # TODO: embed tokens into vectors h of shape (B, T, d_model)
        # Hint: use self.emb
        h = None

        # TODO: build Q, K, V
        # - Q should be taken ONLY from the query position q for each batch element
        # Hint: x[batch, q] -- think if you should use x or h here
        # Another Hint: you can use unsqueeze(dim) method to add another dim to a torch tensor
        # - shapes should end up as:
            #   Q: (B, 1, d_k)
            #   K: (B, T, d_k)
            #   V: (B, T, d_model)
        Q = None
        K = None
        V = None

        # TODO: compute attention scores = QK^T / scale
        # expected shape: (B, 1, T)
        scores = None

        # TODO: softmax over the token dimension to get attention weights
        # expected shape: (B, 1, T)
        # Hint: use F.softmax
        attn = None

        # TODO: compute context vector as weighted sum of V
        # Here we use the attention weights to get the output (based on V)
        # Hint: You can remove a dim from torch tensor using squeeze(dim) method
        # ctx should be (B, d_model)
        ctx = None

        # TODO: project to vocab logits (B, vocab_size)
        # Hint: see the self.out layer to move back to vocab space (opposite of self.emb)
        logits = None

        return logits, attn.squeeze(1)


def main():
    n_pairs = 5
    batch_size = 2
    model = Attention(vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(800):
        x, q, y = make_kv_batch(batch_size, n_pairs)
        logits, attn = model(x, q)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ----------------------------
    # Inspect one example + plot attention
    # ----------------------------
    x, q, y = make_kv_batch(1, n_pairs)
    logits, attn = model(x, q)
    pred = logits.argmax(dim=-1).item()

    tokens = [itos[i] for i in x[0].tolist()]
    qpos = q.item()

    print("Sequence:", tokens)
    print(f"Query position {qpos} token='{tokens[qpos]}'")
    print("Correct value:", itos[y.item()])
    print("Model predicted:", itos[pred])
    print("Attention:", [round(a, 3) for a in attn[0].tolist()])

    plt.figure(figsize=(10, 2.5))
    plt.title("Attention over positions (higher = model is 'looking' there)")
    plt.imshow(attn.detach().numpy(), aspect="auto")
    plt.yticks([0], ["query"])
    plt.xticks(range(len(tokens)), [f"{i}\n{tok}" for i, tok in enumerate(tokens)], rotation=0)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Task 1:
    #   - Complete the Attention class and make sure that you are getting the correct answers in the above given code.
    # Task 2:
    #   - Change the value of self.scale (sqrt(d)) in Attention class and analyze how that affects the performance (wrt number of steps etc)

    main()