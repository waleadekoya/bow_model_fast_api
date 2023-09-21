import torch


class BoWClassifier(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super(BoWClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(nwords, ntags)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        _type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.bias = torch.zeros(ntags, requires_grad=True).type(_type)

    def forward(self, x):
        emb = self.embedding(x)  # seq_len x ntags (for each seq)
        out = torch.sum(emb, dim=0) + self.bias  # ntags
        out = out.view(1, -1)  # reshape to (1, ntags)
        return out
