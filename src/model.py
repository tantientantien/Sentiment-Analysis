import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, pretrained_embeddings=None):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)
        
        self.bilstm = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, texts, weights, selected_text=None):

        embedded = self.embedding(texts)
        if weights.size(1) != embedded.size(1):
            weights = F.pad(weights, (0, embedded.size(1) - weights.size(1)), "constant", 0)
        
        weights = weights.unsqueeze(2).expand(-1, embedded.size(1), -1)
        embedded = torch.cat((embedded, weights), dim=2) 
        
        lstm_out, _ = self.bilstm(embedded)
        
        # Apply attention
        attn_output = self.attention(lstm_out, weights)
        
        if selected_text is not None:
            pass
        
        # fully connected layer
        output = self.fc(attn_output)
        return output


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output, weights):
        attn_weights = self.attn(lstm_output).squeeze(2)
        attn_weights = attn_weights + weights.squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)
        weighted = lstm_output * attn_weights
        representations = weighted.sum(dim=1)
        return representations 