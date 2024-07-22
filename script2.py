import os
import sys
import torch
import numpy as np
from gensim.models import Word2Vec
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import chess
import random

# Custom Vocabulary class (unchanged)
class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']  # Removed spaces in '<SOS>'
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def __len__(self):
        return len(self.token2idx)

# New ChessState class to represent the game state
class ChessState:
    def __init__(self):
        self.board = chess.Board()
        
    def apply_move(self, move):
        self.board.push_san(move)
        
    def get_legal_moves(self):
        return [self.board.san(move) for move in self.board.legal_moves]
    
    def is_game_over(self):
        return self.board.is_game_over()
    
    def get_result(self):
        return self.board.result()
    
    def to_fen(self):
        return self.board.fen()

# Modified ChessDataset class
# Modified ChessDataset class
class ChessDataset(Dataset):
    def __init__(self, data, vocab, max_len=200):
        self.vocab = vocab
        self.max_len = max_len
        self.valid_games = []
        self.process_data(data)

    def process_data(self, data):
        for idx, game in enumerate(data):
            moves = game.strip().split()
            state = ChessState()
            valid_game = []
            try:
                for move in moves:
                    if move not in state.get_legal_moves():
                        print(f"Skipping game {idx + 1} due to illegal move: {move}")
                        raise ValueError("Illegal move")
                    valid_game.append((encode_board_state(state.board), move))
                    state.apply_move(move)
                self.valid_games.append(valid_game)
            except ValueError:
                continue

    def __len__(self):
        return len(self.valid_games)

    def __getitem__(self, idx):
        game = self.valid_games[idx]
        input_seq = torch.tensor([self.vocab.token2idx['<SOS>']])  # Changed from '< SOS >' to '<SOS>'
        board_states = []
        output_seq = torch.tensor([self.vocab.token2idx['<SOS>']])  # Changed from '< SOS >' to '<SOS>'
        
        for board_state, move in game[:self.max_len-1]:
            board_states.append(torch.tensor(board_state))
            move_idx = self.vocab.token2idx.get(move, self.vocab.token2idx['<UNK>'])
            output_seq = torch.cat([output_seq, torch.tensor([move_idx])])
        
        output_seq = torch.cat([output_seq, torch.tensor([self.vocab.token2idx['<EOS>']])])
        board_states = torch.stack(board_states)
        
        # Pad sequences
        input_seq = torch.cat([input_seq, torch.zeros(self.max_len - len(input_seq), dtype=torch.long)])
        output_seq = torch.cat([output_seq, torch.full((self.max_len - len(output_seq),), self.vocab.token2idx['<PAD>'])])
        if len(board_states) < self.max_len - 1:
            board_states = torch.cat([board_states, torch.zeros(self.max_len - 1 - len(board_states), 64)])
        
        return input_seq, board_states, output_seq
# Function to create embeddings (unchanged)
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

    model = Word2Vec(sentences, vector_size=256, window=5, min_count=1, workers=4, epochs=10)

    vocab = Vocab()
    embedding_dict = {}

    for token in model.wv.key_to_index:
        vocab.add_token(token)
        embedding_dict[token] = model.wv[token]

    for token in vocab.special_tokens:
        if token not in embedding_dict:
            embedding_dict[token] = np.random.randn(256)

    embedding_matrix = []
    for token in vocab.token2idx.keys():
        if token in embedding_dict:
            embedding_matrix.append(embedding_dict[token])
        else:
            embedding_matrix.append(np.random.randn(256))

    embedding_matrix = torch.tensor(embedding_matrix)

    torch.save(vocab, vocab_path)
    torch.save(embedding_matrix, embeddings_path)

    return vocab, embedding_matrix

def encode_board_state(board):
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    state = np.zeros(64, dtype=int)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            state[i] = piece_to_index[piece.symbol()]
    return state

# Modified Transformer model
class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.board_encoder = nn.Linear(64, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=0.1, max_len=1000)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, board_state, tgt):
        # src shape: [batch_size, seq_len]
        # board_state shape: [batch_size, seq_len-1, 64]
        # tgt shape: [batch_size, seq_len]
        
        src = self.embedding(src) * np.sqrt(256)  # [batch_size, seq_len, embed_size]
        
        # Process board states
        board_encoding = self.board_encoder(board_state.float())  # [batch_size, seq_len-1, embed_size]
        
        # Add a dummy encoding for the SOS token
        dummy_encoding = torch.zeros(board_encoding.shape[0], 1, board_encoding.shape[2], device=board_encoding.device)
        board_encoding = torch.cat([dummy_encoding, board_encoding], dim=1)  # [batch_size, seq_len, embed_size]
        
        # Combine move embeddings and board state encodings
        src = src + board_encoding  # [batch_size, seq_len, embed_size]
        
        src = self.pos_encoder(src.transpose(0, 1))  # [seq_len, batch_size, embed_size]
        tgt = self.embedding(tgt) * np.sqrt(256)  # [batch_size, seq_len, embed_size]
        tgt = self.pos_encoder(tgt.transpose(0, 1))  # [seq_len, batch_size, embed_size]
        
        output = self.transformer(src, tgt)  # [seq_len, batch_size, embed_size]
        return self.fc_out(output.transpose(0, 1))  # [batch_size, seq_len, vocab_size]

# New PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Modified training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            src, board_states, tgt = [b.to(device) for b in batch]
            
            # Ensure all tensors have batch dimension
            if src.dim() == 1:
                src = src.unsqueeze(0)
            if board_states.dim() == 2:
                board_states = board_states.unsqueeze(0)
            if tgt.dim() == 1:
                tgt = tgt.unsqueeze(0)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, board_states, tgt_input)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")


# New function for beam search
def beam_search(model, initial_state, vocab, device, beam_width=3, max_length=10):
    model.eval()
    initial_input = torch.tensor([vocab.token2idx['<SOS>']]).unsqueeze(0).to(device)  # Changed from '< SOS >' to '<SOS>'
    board_state = torch.tensor(encode_board_state(initial_state.board)).unsqueeze(0).unsqueeze(0).to(device)
    initial_score = 0
    beam = [(initial_input, initial_score, initial_state)]
    
    for _ in range(max_length):
        candidates = []
        for seq, score, state in beam:
            if seq[0, -1].item() == vocab.token2idx['<EOS>']:
                candidates.append((seq, score, state))
                continue
            
            with torch.no_grad():
                # Ensure board_state has the correct shape [batch_size, seq_len-1, 64]
                current_board_state = board_state.repeat(1, seq.size(1) - 1, 1)
                output = model(seq, current_board_state, seq)
                probs = torch.softmax(output[0, -1], dim=-1)
                top_probs, top_indices = probs.topk(beam_width)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score - torch.log(prob).item()
                    new_state = ChessState()
                    new_state.board = state.board.copy()
                    move = vocab.idx2token[idx.item()]
                    if move in new_state.get_legal_moves():
                        new_state.apply_move(move)
                        candidates.append((new_seq, new_score, new_state))
                    else:
                        print(f"Beam search: Illegal move {move} suggested by model")
        
        if not candidates:
            print("Beam search: No legal moves found")
            return None, initial_state

        beam = sorted(candidates, key=lambda x: x[1])[:beam_width]
        if all(seq[0, -1].item() == vocab.token2idx['<EOS>'] for seq, _, _ in beam):
            break
    
    if not beam:
        print("Beam search: No valid sequences found")
        return None, initial_state

    return beam[0][0].squeeze(), beam[0][2]

# Modified main function
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
    print(f"Total games in dataset: {len(train_data)}")
    print(f"Valid games after filtering: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No valid games found in the dataset. Please check your data.")
        sys.exit(1)

    # Modified DataLoader
    def collate_fn(batch):
        # Unzip the batch into separate lists
        input_seqs, board_states, output_seqs = zip(*batch)
        
        # Pad sequences to the same length
        input_seqs = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=vocab.token2idx['<PAD>'])
        board_states = torch.nn.utils.rnn.pad_sequence(board_states, batch_first=True, padding_value=0)
        output_seqs = torch.nn.utils.rnn.pad_sequence(output_seqs, batch_first=True, padding_value=vocab.token2idx['<PAD>'])
        
        return input_seqs, board_states, output_seqs

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessTransformer(
        vocab_size=len(vocab),
        embed_size=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024
    ).to(device)

    # Train model
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx['<PAD>'])
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
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
    
    state = ChessState()
    if player_color == "white":
        while not state.is_game_over():
            print(f"\nCurrent board:\n{state.board}")
            move = input("Your move (e.g., 'e4'): ")
            if move.lower() == 'exit':
                break
            
            if move not in state.get_legal_moves():
                print("Illegal move. Try again.")
                continue
            
            state.apply_move(move)
            
            if state.is_game_over():
                break
            
            # Generate model's move using beam search
            input_seq = torch.tensor([vocab.token2idx['<SOS>']]).unsqueeze(0).to(device)
            board_state = torch.tensor(encode_board_state(state.board)).unsqueeze(0).to(device)
            
            output_seq, new_state = beam_search(model, state, vocab, device)
            if output_seq is None:
                print("Model couldn't find a legal move. Game over.")
                break
            predicted_move = vocab.idx2token[output_seq[-2].item()]  # -2 to get the move before <EOS>
            
            if predicted_move not in state.get_legal_moves():
                print(f"Model suggested illegal move: {predicted_move}. Game over.")
                break
            
            print(f"Model's move: {predicted_move}")
            state.apply_move(predicted_move)
    else:
        print("Model plays as white:")
        while not state.is_game_over():
            print(f"\nCurrent board:\n{state.board}")
            
            # Generate model's move using beam search
            input_seq = torch.tensor([vocab.token2idx['<SOS>']]).unsqueeze(0).to(device)
            board_state = torch.tensor(encode_board_state(state.board)).unsqueeze(0).to(device)
            
            output_seq, new_state = beam_search(model, state, vocab, device)
            if output_seq is None:
                print("Model couldn't find a legal move. Game over.")
                break
            predicted_move = vocab.idx2token[output_seq[-2].item()]  # -2 to get the move before <EOS>
            
            if predicted_move not in state.get_legal_moves():
                print(f"Model suggested illegal move: {predicted_move}. Game over.")
                break
            
            print(f"Model's move: {predicted_move}")
            state.apply_move(predicted_move)
            
            if state.is_game_over():
                break
            
            print(f"\nCurrent board:\n{state.board}")
            move = input("Your move (e.g., 'e5'): ")
            if move.lower() == 'exit':
                break
            
            if move not in state.get_legal_moves():
                print("Illegal move. Try again.")
                continue
            
            state.apply_move(move)
    
    print(f"\nGame over. Result: {state.get_result()}")

if __name__ == "__main__":
    main()