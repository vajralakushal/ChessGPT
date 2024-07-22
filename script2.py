import os
import sys
import torch
import numpy as np
from gensim.models import Word2Vec
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Custom Vocabulary class
class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def __len__(self):
        return len(self.token2idx)

# Custom dataset for chess moves
class ChessDataset(Dataset):
    def __init__(self, data, vocab, max_len=100):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        moves = self.data[idx].strip().split()
        moves = ['<SOS>'] + moves + ['<EOS>']
        moves = moves[:self.max_len]
        moves = [self.vocab.token2idx.get(move, self.vocab.token2idx['<UNK>']) for move in moves]
        moves = moves + [self.vocab.token2idx['<PAD>']] * (self.max_len - len(moves))
        return torch.tensor(moves)

# Function to create embeddings
def create_embeddings(corpus_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_path = os.path.join(output_dir, 'vocab.pth')
    embeddings_path = os.path.join(output_dir, 'embeddings.pth')

    if os.path.exists(vocab_path) and os.path.exists(embeddings_path):
        print("Loading existing embeddings...")
        vocab = torch.load(vocab_path)
        embedding_matrix = torch.load(embeddings_path)
        return vocab, embedding_matrix

    print("Creating new embeddings...")
    with open(corpus_file, 'r') as f:
        sentences = [line.strip().split() for line in f]

    model = Word2Vec(sentences, vector_size=128, window=3, min_count=1, workers=4, epochs=5)

    vocab = Vocab()
    embedding_dict = {}

    for token in model.wv.key_to_index:
        vocab.add_token(token)
        embedding_dict[token] = model.wv[token]

    for token in vocab.special_tokens:
        if token not in embedding_dict:
            embedding_dict[token] = np.random.randn(128)

    embedding_matrix = []
    for token in vocab.token2idx.keys():
        if token in embedding_dict:
            embedding_matrix.append(embedding_dict[token])
        else:
            embedding_matrix.append(np.random.randn(128))

    embedding_matrix = torch.tensor(embedding_matrix)

    torch.save(vocab, vocab_path)
    torch.save(embedding_matrix, embeddings_path)

    return vocab, embedding_matrix

# Transformer model
class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        output = self.transformer(src, tgt)
        return self.fc_out(output.permute(1, 0, 2))

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            src = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Main function
def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py corpus_file train_file embeddings_dir model_output_dir")
        sys.exit(1)

    corpus_file = sys.argv[1]
    train_file = sys.argv[2]
    embeddings_dir = sys.argv[3]
    model_output_dir = sys.argv[4]

    # Create embeddings
    vocab, embedding_matrix = create_embeddings(corpus_file, embeddings_dir)

    # Prepare training data
    with open(train_file, 'r') as f:
        train_data = f.readlines()

    dataset = ChessDataset(train_data, vocab)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessTransformer(
        vocab_size=len(vocab),
        embed_size=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    ).to(device)

    # Train model
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx['<PAD>'])
    optimizer = Adam(model.parameters())
    train_model(model, train_loader, criterion, optimizer, device)

    # Save model
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model.state_dict(), os.path.join(model_output_dir, 'chess_model.pth'))
    print(f"Model saved to {os.path.join(model_output_dir, 'chess_model.pth')}")

    # Simple chess game interaction
    model.eval()
    print("Let's play chess! Type 'exit' to quit.")
    player_color = input("Choose your color (white/black): ").lower()
    
    if player_color == "white":
        while True:
            move = input("Your move (e.g., 'e4'): ")
            if move.lower() == 'exit':
                break
            
            # Prepare input for the model
            input_seq = torch.tensor([vocab.token2idx.get(token, vocab.token2idx['<UNK>']) for token in ['<SOS>', move]])
            input_seq = input_seq.unsqueeze(0).to(device)
            
            # Generate model's move
            with torch.no_grad():
                output = model(input_seq, input_seq)
                predicted_move = vocab.idx2token[output[0, -1].argmax().item()]
            
            print(f"Model's move: {predicted_move}")
    else:
        print("Model plays as white:")
        while True:
            # Generate model's move
            input_seq = torch.tensor([vocab.token2idx['<SOS>']])
            input_seq = input_seq.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_seq, input_seq)
                predicted_move = vocab.idx2token[output[0, -1].argmax().item()]
            
            print(f"Model's move: {predicted_move}")
            
            move = input("Your move (e.g., 'e5'): ")
            if move.lower() == 'exit':
                break

if __name__ == "__main__":
    main()