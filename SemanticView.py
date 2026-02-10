import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticView(nn.Module):
    def __init__(self, pretrained_embeddings, out_dim=6):
        super(SemanticView, self).__init__()

        self.m_i = nn.Parameter(pretrained_embeddings)

        self.fc = nn.Linear(pretrained_embeddings.size(1), out_dim)
        self.activation = nn.ReLU()

    def forward(self):
        m_hat = self.activation(self.fc(self.m_i))
        return m_hat