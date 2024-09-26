import chess_modules
from gensim.models import Word2Vec
import torch
import os
from abstract_classes import AbstractDataHandler, AbstractDataset
import gc
import numpy as np
import chess
from script import EMBED_SIZE

#EMBED_SIZE = 128

class ChessDataHandler(AbstractDataHandler):
    def create_embeddings(self, corpus_file, output_dir):
        from gensim.models import Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec
        import torch
        import os
        import numpy as np

        class callback(CallbackAny2Vec):
            def __init__(self):
                self.epoch = 0
                self.previous_loss = None
            
            def on_epoch_end(self, model):
                current_loss = model.get_latest_training_loss()
                if self.previous_loss is None:
                    print(f'Loss after epoch {self.epoch}: {current_loss}')
                else:
                    loss_diff = current_loss - self.previous_loss
                    print(f'Loss after epoch {self.epoch}: {loss_diff}')
                self.previous_loss = current_loss
                self.epoch += 1

        model = Word2Vec(corpus_file=corpus_file, vector_size=EMBED_SIZE, window=3, min_count=1, 
                        workers=4, epochs=5, callbacks=[callback()], compute_loss=True)

        vocab = Vocab()
        embedding_dict = {}

        print(f"Number of tokens in Word2Vec vocabulary: {len(model.wv.key_to_index)}")

        for token in model.wv.key_to_index:
            vocab.add_token(token)
            embedding_dict[token] = model.wv[token]

        # Add special tokens to vocab and create random embeddings for them
        for token in vocab.special_tokens:
            if token not in vocab.token2idx:
                vocab.add_token(token)
            if token not in embedding_dict:
                embedding_dict[token] = np.random.randn(128)

        print(f"Number of tokens in custom vocabulary: {len(vocab)}")

        # Create embedding matrix ensuring all vocab tokens have an embedding
        embedding_matrix = []
        for token in vocab.token2idx.keys():
            if token in embedding_dict:
                embedding_matrix.append(embedding_dict[token])
            else:
                # If a token is in vocab but not in embedding_dict, create a random embedding
                embedding_matrix.append(np.random.randn(128))

        embedding_matrix = np.array(embedding_matrix)
        embedding_matrix = torch.tensor(embedding_matrix)

        print(f"Final embedding matrix shape: {embedding_matrix.shape}")

        torch.save(vocab, os.path.join(output_dir, 'vocab.pth'))
        torch.save(embedding_matrix, os.path.join(output_dir, 'embeddings.pth'))

        return vocab, embedding_matrix

    def encode_state(self, board):
        state = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                state.extend([0] * 12)
            else:
                piece_vector = [0] * 12
                piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color == chess.WHITE else 1)
                piece_vector[piece_idx] = 1
                state.extend(piece_vector)
        
        state.append(int(board.turn))
        state.extend([int(board.has_kingside_castling_rights(chess.WHITE)),
                      int(board.has_queenside_castling_rights(chess.WHITE)),
                      int(board.has_kingside_castling_rights(chess.BLACK)),
                      int(board.has_queenside_castling_rights(chess.BLACK))])
        
        return torch.tensor(state, dtype=torch.float)

    def get_state_size(self):
        return 64 * 12 + 5  # 64 squares * 12 piece types + 5 additional state info

    def is_valid_token(self, token, board):
        try:
            move = chess_modules.Move.from_uci(token)
            return move in board.legal_moves
        except ValueError:
            return False

    def get_state_size(self):
        return 64 * 12 + 5  # 64 squares * 12 piece types + 5 additional state info

class ChessDataset(AbstractDataset):
    def __init__(self, file_path, vocab, data_handler, max_len=200):
        self.max_len = max_len
        super().__init__(file_path, vocab, data_handler)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                game = [self.vocab.sos_token] + line.strip().split() + [self.vocab.eos_token]
                if len(game) > self.max_len:
                    game = game[:self.max_len-1] + [self.vocab.eos_token]
                board = chess.Board()
                game_data = []
                try:
                    for move in game[1:-1]:  # Exclude SOS and EOS for actual moves
                        state = self.data_handler.encode_state(board)
                        if move not in self.vocab.token2idx:
                            raise ValueError(f"Unknown move: {move}")
                        try:
                            chess_move = board.parse_san(move)
                            if chess_move not in board.legal_moves:
                                raise ValueError(f"Illegal move: {move}")
                        except ValueError:
                            raise ValueError(f"Invalid move: {move}")
                        move_idx = self.vocab.token2idx[move]
                        game_data.append((state, move_idx))
                        board.push_san(move)
                    
                    # Pad or truncate the game data to max_len
                    if len(game_data) < self.max_len:
                        pad_length = self.max_len - len(game_data)
                        pad_state = torch.zeros_like(game_data[0][0])
                        pad_move = self.vocab.token2idx[self.vocab.pad_token]
                        game_data.extend([(pad_state, pad_move)] * pad_length)
                    else:
                        game_data = game_data[:self.max_len]
                    
                    self.data.append(game_data)
                except ValueError as e:
                    print(f"Skipping game at line {line_num} due to error: {str(e)}")
                    self.skipped_items += 1
        
        if self.skipped_items > 0:
            print(f"Skipped {self.skipped_items} games due to errors.")

    def __getitem__(self, idx):
        game = self.data[idx]
        states, moves = zip(*game)
        return torch.stack(states), torch.tensor(moves)

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