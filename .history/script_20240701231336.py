import argparse
import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

# Constants
EMBED_SIZE = 128
NHEAD = 8
NUM_LAYERS = 6
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
PATIENCE = 3

class Vocab:
    def __init__(self):
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token]
        self.token2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
            self.idx2token[len(self.idx2token)] = token
    
    def __len__(self):
        return len(self.token2idx)

class AbstractDataHandler(ABC):
    @abstractmethod
    def create_embeddings(self, corpus_file, output_dir):
        pass
    
    @abstractmethod
    def encode_state(self, state):
        pass
    
    @abstractmethod
    def is_valid_token(self, token):
        pass

class AbstractDataset(Dataset):
    def __init__(self, file_path, vocab, data_handler):
        self.vocab = vocab
        self.data_handler = data_handler
        self.data = []
        self.skipped_items = 0
        
        self.load_data(file_path)
    
    @abstractmethod
    def load_data(self, file_path):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        states, tokens = zip(*item)
        return torch.stack(states), torch.tensor(tokens)

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers, state_size):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.state_encoder = nn.Linear(state_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, src, states, src_mask):
        src = self.embedding(src)
        states = self.state_encoder(states)
        src = src + states
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for states, tokens in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        states = states.to(device)
        tokens = tokens.to(device)
        
        src = tokens[:-1]  # Input tokens
        tgt = tokens[1:]   # Target tokens
        
        src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
        
        output = model(src, states[:-1], src_mask)
        loss = criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for states, tokens in tqdm(data_loader, desc="Validating"):
            states = states.to(device)
            tokens = tokens.to(device)
            
            src = tokens[:-1]  # Input tokens
            tgt = tokens[1:]   # Target tokens
            
            src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
            
            output = model(src, states[:-1], src_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def main(embed_dir, data_file, output_dir, dataset_class, data_handler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create embeddings if they don't exist
    if not os.path.exists(os.path.join(embed_dir, 'embeddings.pth')):
        print("Creating embeddings...")
        vocab, embedding_matrix = data_handler.create_embeddings('corpus.txt', embed_dir)
    else:
        # Load vocabulary and embeddings
        vocab = torch.load(os.path.join(embed_dir, 'vocab.pth'))
        embedding_matrix = torch.load(os.path.join(embed_dir, 'embeddings.pth'))
    
    # Create dataset and split into train and validation
    dataset = dataset_class(data_file, vocab, data_handler)
    train_data, val_data = train_test_split(dataset, test_size=VALIDATION_SPLIT, random_state=42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # Initialize model
    state_size = data_handler.get_state_size()
    model = GPTModel(len(vocab), EMBED_SIZE, NHEAD, NUM_LAYERS, state_size).to(device)
    model.embedding.weight.data.copy_(embedding_matrix)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx[vocab.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'gpt_model_best.pth'))
            print(f"New best model saved to {os.path.join(output_dir, 'gpt_model_best.pth')}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("embed_dir", help="Path to embeddings directory")
    parser.add_argument("data_file", help="Path to data file")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("domain", help="Domain of the data (e.g., chess, sanskrit, hebrew)")
    args = parser.parse_args()
    
    # Import domain-specific modules
    if args.domain == "chess":
        from chess_modules import ChessDataHandler, ChessDataset
        data_handler = ChessDataHandler()
        dataset_class = ChessDataset
    elif args.domain == "sanskrit":
        from sanskrit_modules import SanskritDataHandler, SanskritDataset
        data_handler = SanskritDataHandler()
        dataset_class = SanskritDataset
    elif args.domain == "hebrew":
        from hebrew_modules import HebrewDataHandler, HebrewDataset
        data_handler = HebrewDataHandler()
        dataset_class = HebrewDataset
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")
    
    main(args.embed_dir, args.data_file, args.output_dir, dataset_class, data_handler)