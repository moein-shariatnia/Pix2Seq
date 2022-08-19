import torch

class CFG:
    img_path = './VOCdevkit/VOC2012/JPEGImages/'
    xml_path = './VOCdevkit/VOC2012/Annotations/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_len = 300
    img_size = 384
    num_bins = img_size
    
    batch_size = 16
    epochs = 25
    
    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    
    lr = 1e-4
    weight_decay = 1e-4
    
    
    generation_steps = 101
    run_eval = False