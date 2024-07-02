import chess_modules
from gensim.models import Word2Vec
import torch
import os
from abstract_classes import AbstractDataHandler, AbstractDataset

class ChessDataHandler(AbstractDataHandler):
    def create_embeddings(self, corpus_file, output_dir):
        with open(corpus_file, 'r') as f:
            sentences = [line.strip().split() for line in f]
        
        model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=4)
        
        vocab = Vocab()
        embedding_matrix = []
        
        for token in model.wv.key_to_index:
            vocab.add_token(token)
            embedding_matrix.append(model.wv[token])
        
        for token in vocab.special_tokens:
            embedding_matrix.append(torch.randn(128))
        
        embedding_matrix = torch.tensor(embedding_matrix)
        
        torch.save(vocab, os.path.join(output_dir, 'vocab.pth'))
        torch.save(embedding_matrix, os.path.join(output_dir, 'embeddings.pth'))
        
        return vocab, embedding_matrix

    def encode_state(self, board):
        state = []
        for square in chess_modules.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                state.extend([0] * 12)
            else:
                piece_vector = [0] * 12
                piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color == chess_modules.WHITE else 1)
                piece_vector[piece_idx] = 1
                state.extend(piece_vector)
        
        state.append(int(board.turn))
        state.extend([int(board.has_kingside_castling_rights(chess_modules.WHITE)),
                      int(board.has_queenside_castling_rights(chess_modules.WHITE)),
                      int(board.has_kingside_castling_rights(chess_modules.BLACK)),
                      int(board.has_queenside_castling_rights(chess_modules.BLACK))])
        
        return torch.tensor(state, dtype=torch.float)

    def is_valid_token(self, token, board):
        try:
            move = chess_modules.Move.from_uci(token)
            return move in board.legal_moves
        except ValueError:
            return False

    def get_state_size(self):
        return 64 * 12 + 5  # 64 squares * 12 piece types + 5 additional state info

class ChessDataset(AbstractDataset):
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                game = [self.vocab.sos_token] + line.strip().split() + [self.vocab.eos_token]
                board = chess_modules.Board()
                game_data = []
                try:
                    for move in game[1:-1]:  # Exclude SOS and EOS for actual moves
                        state = self.data_handler.encode_state(board)
                        if move not in self.vocab.token2idx:
                            raise ValueError(f"Unknown move: {move}")
                        if not self.data_handler.is_valid_token(move, board):
                            raise ValueError(f"Illegal move: {move}")
                        move_idx = self.vocab.token2idx[move]
                        game_data.append((state, move_idx))
                        board.push_san(move)
                    self.data.append(game_data)
                except ValueError as e:
                    print(f"Skipping game at line {line_num} due to error: {str(e)}")
                    self.skipped_items += 1
        
        if self.skipped_items > 0:
            print(f"Skipped {self.skipped_items} games due to errors.")

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