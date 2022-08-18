import os
import cv2
import torch
from tqdm import tqdm
import numpy as np

from dataset import VOCDatasetTest
from model import Encoder, Decoder, EncoderDecoder
from test import generate, postprocess
from tokenizer import Tokenizer
from config import CFG
from visualize import visualize


with open("valid_paths.txt", "r") as f:
    img_paths = f.readlines()
    img_paths = [path.strip() for path in img_paths]
    
idxs = np.random.randint(0, len(img_paths)-1, size=10)
img_paths = [img_paths[idx] for idx in idxs]

if __name__ == '__main__':
    with open("classes.txt", 'r') as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    tokenizer = Tokenizer(num_classes=len(classes), num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
    CFG.pad_idx = tokenizer.PAD_code

    test_dataset = VOCDatasetTest(img_paths, size=CFG.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(img_paths), shuffle=False, num_workers=0)

    encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
    model = EncoderDecoder(encoder, decoder)
    model.to(CFG.device)

    msg = model.load_state_dict(torch.load(
        './pix2seq_weights.pth', map_location=CFG.device))
    print(msg)
    model.eval()

    all_bboxes = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for x in tqdm(test_loader):
            batch_preds, batch_confs = generate(
                model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            bboxes, labels, confs = postprocess(
                batch_preds, batch_confs, tokenizer)
            all_bboxes.extend(bboxes)
            all_labels.extend(labels)
            all_confs.extend(confs)

    os.mkdir("results")
    for i, (bboxes, labels, confs) in enumerate(zip(all_bboxes, all_labels, all_confs)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)[..., ::-1]
        img = visualize(img, bboxes, labels, id2cls, show=False)

        cv2.imwrite("results/" + img_path.split("/")[-1], img[..., ::-1])
