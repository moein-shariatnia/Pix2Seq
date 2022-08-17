
import torch
from torch import nn
import numpy as np
from glob import glob
from transformers import get_linear_schedule_with_warmup

from utils import seed_everything
from preprocess import build_df
from dataset import split_df, get_loaders
from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from engine import train_eval
from config import CFG

if __name__ == '__main__':
    seed_everything(42)

    IMG_FILES = glob(CFG.img_path + "/*.jpg")
    XML_FILES = glob(CFG.xml_path + "/*.xml")
    assert len(IMG_FILES) == len(
        XML_FILES) != 0, "images or xml files not found"
    print("Number of found images: ", len(IMG_FILES))

    df, classes = build_df(XML_FILES)
    # build id to class name and vice verca mappings
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    train_df, valid_df = split_df(df)
    print("Train size: ", train_df['id'].nunique())
    print("Valid size: ", valid_df['id'].nunique())

    tokenizer = Tokenizer(num_classes=len(classes), num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code
    train_loader, valid_loader = get_loaders(
        train_df, valid_df, tokenizer, CFG.img_size, CFG.batch_size, CFG.max_len, tokenizer.PAD_code)

    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * \
        (len(train_loader.dataset) // CFG.batch_size)
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=CFG.pad_idx)

    train_eval(model,
               train_loader,
               valid_loader,
               criterion,
               optimizer,
               lr_scheduler=lr_scheduler,
               step='batch',
               logger=None)
