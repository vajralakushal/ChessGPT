import argparse
import os
from abstract_classes import AbstractDataHandler, AbstractDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
#from chess_modules import EMBED_SIZE


# Constants
EMBED_SIZE = 128
NHEAD = 8
NUM_LAYERS = 6
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
PATIENCE = 3

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_layers, state_size):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.state_encoder = nn.Linear(state_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, src, states, src_mask):
        src = self.embedding(src)
        states = self.state_encoder(states)
        # Ensure dimensions match
        src = src[:, :states.size(1), :]
        src = src + states
        src = self.pos_encoder(src)
        batch_size, seq_len, embed_size = src.size()
        nhead = self.transformer_encoder.layers[0].self_attn.num_heads
        src = src.view(batch_size * nhead, seq_len, embed_size // nhead)
        output = self.transformer_encoder(src, src_mask)
        output = output.view(batch_size, seq_len, embed_size)
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
        x = x + self.pe[:x.size(0), :].to(x.device)
        return x

def create_attention_mask(moves, nhead, device):
    pad_mask = (moves != 0).unsqueeze(1).repeat(1, moves.size(1), 1)
    seq_length = moves.size(1)
    subsequent_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(device)
    subsequent_mask = subsequent_mask.unsqueeze(0)
    attn_mask = pad_mask & ~subsequent_mask
    attn_mask = attn_mask.unsqueeze(1).repeat(1, nhead, 1, 1)
    attn_mask = attn_mask.view(-1, seq_length, seq_length)
    return attn_mask

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for states, moves in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        states = states.to(device)
        moves = moves.to(device)
        
        attn_mask = create_attention_mask(moves, model.transformer_encoder.layers[0].self_attn.num_heads, device)
        
        seq_len = min(moves.size(1) - 1, states.size(1))
        print(f"seq_len is {seq_len}")
        output = model(moves[:, :seq_len], states[:, :seq_len], attn_mask[:, :seq_len, :seq_len])
        
        loss = criterion(output.contiguous().view(-1, output.size(-1)), moves[:, 1:seq_len+1].contiguous().view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for states, moves in tqdm(data_loader, desc="Validating"):
            states = states.to(device)
            moves = moves.to(device)
            
            attn_mask = create_attention_mask(moves, model.transformer_encoder.layers[0].self_attn.num_heads, device)
            
            seq_len = min(moves.size(1) - 1, states.size(1))
            output = model(moves[:, :seq_len], states[:, :seq_len], attn_mask[:, :seq_len, :seq_len])
            
            loss = criterion(output.contiguous().view(-1, output.size(-1)), moves[:, 1:seq_len+1].contiguous().view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def custom_collate(batch):
    # Separate states and moves
    states, moves = zip(*batch)
    
    # Get the maximum sequence length in this batch
    max_len = max(len(m) for m in moves)
    
    # Pad the sequences
    state_size = states[0].size(1)  # Assuming all states have the same feature size
    states_padded = [torch.cat([s, torch.zeros(max_len - len(s), state_size)]) if len(s) < max_len else s for s in states]
    moves_padded = [torch.cat([m, torch.zeros(max_len - len(m))]) if len(m) < max_len else m for m in moves]
    
    # Stack the padded sequences
    states_padded = torch.stack(states_padded)
    moves_padded = torch.stack(moves_padded)
    
    return states_padded, moves_padded

# Main function and argument parser (assuming you have this part in the script)
def main(embed_file, train_file, embed_dir, output_dir, dataset_class, data_handler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create embeddings if they don't exist
    if not os.path.exists(os.path.join(embed_dir, 'embeddings.pth')):
        print("Creating embeddings...")
        vocab, embedding_matrix = data_handler.create_embeddings(embed_file, embed_dir)
    else:
        print("Loading existing embeddings...")
        vocab = torch.load(os.path.join(embed_dir, 'vocab.pth'))
        embedding_matrix = torch.load(os.path.join(embed_dir, 'embeddings.pth'))
        assert embedding_matrix.shape[1] == EMBED_SIZE, f"Loaded embedding size {embedding_matrix.shape[1]} does not match expected size {EMBED_SIZE}"

    
    # Create dataset and split into train and validation
    dataset = dataset_class(train_file, vocab, data_handler, max_len=200)  # Adjust max_len as needed
    train_data, val_data = train_test_split(dataset, test_size=VALIDATION_SPLIT, random_state=42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=custom_collate)
    
    # Initialize model
    state_size = data_handler.get_state_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(len(vocab), EMBED_SIZE, NHEAD, NUM_LAYERS, state_size).to(device)
    model.embedding.weight.data.copy_(embedding_matrix.to(device))
    
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
    parser.add_argument("embed_file", help="Path to file for creating embeddings")
    parser.add_argument("train_file", help="Path to file for training the model")
    parser.add_argument("embed_dir", help="Path to embeddings directory")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("domain", help="Domain of the data (e.g., chess, sanskrit, hebrew)")
    args = parser.parse_args()
    
    # Import domain-specific modules
    if args.domain == "chess":
        from chess_modules import ChessDataHandler, ChessDataset, Vocab
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
    
    main(args.embed_file, args.train_file, args.embed_dir, args.output_dir, dataset_class, data_handler)