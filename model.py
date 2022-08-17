import timm
import torch
from torch import nn
from timm.models.layers import trunc_normal_


from config import CFG
from utils import create_mask


class Encoder(nn.Module):
    def __init__(self, model_name='deit3_small_patch16_384_in21ft1k', pretrained=False, out_dim=256):
        super().__init__()
        self.model = timm.create_model(
            model_name, num_classes=0, global_pool='', pretrained=pretrained)
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):
        features = self.model(x)
        return self.bottleneck(features[:, 1:])


class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_length, dim, num_heads, num_layers):
        super().__init__()
        self.dim = dim
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, CFG.max_len-1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.05)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)
        
        
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_length, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)
        
        self.init_weights()
        
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name: 
                print("skipping pos_embed...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
    
    def forward(self, encoder_out, tgt):
        """
        encoder_out: shape(N, L, D)
        tgt: shape(N, L)
        """
        
        tgt_mask, tgt_padding_mask = create_mask(tgt)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )
        
        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )
        
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        
        preds = self.decoder(memory=encoder_out, 
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask, 
                             tgt_key_padding_mask=tgt_padding_mask)
        
        preds = preds.transpose(0, 1)
        return self.output(preds)
    
    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), CFG.max_len-length-1).fill_(CFG.pad_idx).long().to(tgt.device)
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt)
        # is it necessary to multiply it by math.sqrt(d) ?
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )
        
        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )
        
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        
        preds = self.decoder(memory=encoder_out, 
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask, 
                             tgt_key_padding_mask=tgt_padding_mask)
        
        preds = preds.transpose(0, 1)
        return self.output(preds)[:, length-1, :]

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, image, tgt):
        encoder_out = self.encoder(image)
        preds = self.decoder(encoder_out, tgt)
        return preds
    def predict(self, image, tgt):
        encoder_out = self.encoder(image)
        preds = self.decoder.predict(encoder_out, tgt)
        return preds