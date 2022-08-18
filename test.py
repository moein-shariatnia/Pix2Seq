import cv2
import argparse
import torch
from torch import nn
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import top_k_top_p_filtering
from map_boxes import mean_average_precision_for_boxes


from utils import seed_everything
from preprocess import build_df
from dataset import split_df, get_loaders
from tokenizer import Tokenizer
from model import Encoder, Decoder, EncoderDecoder
from config import CFG

def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1):
    x = x.to(CFG.device)
    batch_preds = torch.ones(x.size(0), 1).fill_(tokenizer.BOS_code).long().to(CFG.device)
    confs = []
    
    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)
        
    with torch.no_grad():
        for i in range(max_len):
            preds = model.predict(x, batch_preds)
            ## If top_k and top_p are set to default, the following line does nothing!
            preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
            if i % 4 == 0:
                confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                confs.append(confs_)
            preds = sample(preds)
            batch_preds = torch.cat([batch_preds, preds], dim=1)
    
    return batch_preds.cpu(), confs

def postprocess(batch_preds, batch_confs, tokenizer):
    EOS_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    ## sanity check
    invalid_idxs = ((EOS_idxs - 1) % 5 != 0).nonzero().view(-1)
    EOS_idxs[invalid_idxs] = 0
    
    all_bboxes = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(EOS_idxs.tolist()):
        if EOS_idx == 0:
            all_bboxes.append(None)
            all_labels.append(None)
            all_confs.append(None)
            continue
        labels, bboxes = tokenizer.decode(batch_preds[i, :EOS_idx+1])
        confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]
        
        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_confs.append(confs)
        
    return all_bboxes, all_labels, all_confs


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

 
    
    if CFG.run_eval:
        encoder = Encoder(model_name=CFG.model_name, pretrained=False, out_dim=256)
        decoder = Decoder(vocab_size=tokenizer.vocab_size,
                        encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)
        model = EncoderDecoder(encoder, decoder)
        model.to(CFG.device)
        
        msg = model.load_state_dict(torch.load('./pix2seq_weights.pth', map_location=CFG.device))
        print(msg)
        model.eval()
        
        all_bboxes = []
        all_labels = []
        all_confs = []

        with torch.no_grad():
            for x, _ in tqdm(valid_loader):
                batch_preds, batch_confs = generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
                bboxes, labels, confs = postprocess(batch_preds, batch_confs, tokenizer)
                all_bboxes.extend(bboxes)
                all_labels.extend(labels)        
                all_confs.extend(confs)        
        
        preds_df = pd.DataFrame()
        valid_df = valid_df.iloc[:len(all_bboxes)]
        preds_df['id'] = valid_df['id'].copy()
        preds_df['bbox'] = all_bboxes
        preds_df['label'] = all_labels
        preds_df['conf'] = all_confs

        preds_df = preds_df.explode(['bbox', 'label', 'conf']).reset_index(drop=True)
        preds_df = preds_df[preds_df['bbox'].map(lambda x: isinstance(x, list))].reset_index(drop=True)
        bbox = pd.DataFrame(preds_df['bbox'].tolist(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
        bbox /= float(CFG.img_size)
        preds_df = pd.concat([preds_df, bbox], axis=1)
        preds_df = preds_df.drop('bbox', axis=1)
        preds_df.to_csv("voc_preds.csv", index=False)
    
    preds_df = pd.read_csv("voc_preds.csv")
    valid_df = df[df['id'].isin(preds_df['id'].unique())].reset_index(drop=True)
    
    shapes = {img_path: cv2.imread(img_path).shape[:2] for img_path in valid_df['img_path'].unique()}
    shapes = pd.DataFrame(valid_df['img_path'].map(shapes).tolist(), columns=['h', 'w'])
    valid_df = pd.concat([valid_df, shapes], axis=1)
    
    valid_df['xmin'] = valid_df['xmin'] / valid_df['w']
    valid_df['xmax'] = valid_df['xmax'] / valid_df['w']
    valid_df['ymin'] = valid_df['ymin'] / valid_df['h']
    valid_df['ymax'] = valid_df['ymax'] / valid_df['h']
    
    
    preds_df['label'] = preds_df['label'].map(id2cls)
    valid_df['label'] = valid_df['label'].map(id2cls)
    
    
    ann = valid_df[['id', 'label', 'xmin', 'xmax', 'ymin', 'ymax']].values
    det = preds_df[['id', 'label', 'conf', 'xmin', 'xmax', 'ymin', 'ymax']].values

    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det, iou_threshold=0.5)

    
    

