import math
import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.gate import *
from config import *

class Seq2Seq(nn.Module):
    def __init__(self, 
                inp_dim,
                out_dim,
                max_length,
                pad_idx,
                device):
        super().__init__()
        
        
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.device = device
        self.encoder = Encoder(
            inp_dim, HID_DIM, ENC_LAYERS, ENC_HEADS, 
            ENC_PF_DIM, ENC_DROPOUT, PRUNING, max_length, device)
        
        self.decoder = Decoder(
            out_dim, HID_DIM, DEC_LAYERS, DEC_HEADS, 
            DEC_PF_DIM, DEC_DROPOUT, PRUNING, max_length, device)
        
        # Initialize parameters with Glorot
        """
        reference
        Understanding the difficulty of training deep feedforward neural networks
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]

        # [batch_size, 1, trg_len, trg_len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        assert src.shape[1] <= self.max_length
        assert trg.shape[1] <= self.max_length
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    
    def get_penalty(self):
        assert len([name for name, param in self.named_parameters() if param.requires_grad and 'gate' in name]) == (ENC_LAYERS + 2 * DEC_LAYERS), 'gate.penalty() requires_grad=True'
        gates = [layer.self_attention.gate for layer in self.encoder.layers]
        gates += [layer.self_attention.gate for layer in self.decoder.layers]
        gates += [layer.self_attention.gate for layer in self.decoder.layers]
        return torch.tensor([GATE_L0_PENALTY * torch.sum(gate.penalty(), dim=(1)).mean() for gate in gates]).mean().item()
    
    def get_gates(self):
        enc_self_attn_gates = [layer.self_attention.gate.get_gates(values=None, is_train=False).flatten().tolist() for layer in self.encoder.layers]
        dec_self_attn_gates = [layer.self_attention.gate.get_gates(values=None, is_train=False).flatten().tolist() for layer in self.decoder.layers]
        dec_enc_attn_gates = [layer.encoder_attention.gate.get_gates(values=None, is_train=False).flatten().tolist() for layer in self.decoder.layers]
        return enc_self_attn_gates, dec_self_attn_gates, dec_enc_attn_gates
  