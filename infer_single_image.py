import os
import cv2
import argparse
import torch
from tqdm import tqdm
import numpy as np

from dataset import VOCDatasetTest
from model import Encoder, Decoder, EncoderDecoder
from test import generate, postprocess
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize



parser = argparse.ArgumentParser("Infer single image")
parser.add_argument("--image", type=str, help="Path to image", default="./VOCdevkit/VOC2012/JPEGImages/2012_000947.jpg")

if __name__ == '__main__':
    with open("classes.txt", 'r') as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    tokenizer = Tokenizer(num_classes=len(classes), num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    img_paths = [parser.parse_args().image]
    test_dataset = VOCDatasetTest(img_paths, size=CFG.img_size)

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        './pix2seq_weights.pth', map_location=CFG.device))
    print(msg)
    model.eval()


    x = test_dataset[0].unsqueeze(0)

    with torch.no_grad():
        batch_preds, batch_confs = generate(
            model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
        bboxes, labels, confs = postprocess(
            batch_preds, batch_confs, tokenizer)

    img_path = img_paths[0]
    img = cv2.imread(img_path)[..., ::-1]
    img = cv2.resize(img, (CFG.img_size, CFG.img_size))
    img = visualize(img, bboxes[0], labels[0], id2cls, show=True)
