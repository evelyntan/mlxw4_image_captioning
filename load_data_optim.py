# Import necessary libraries
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from torch.utils.data import Dataset
import torch


# load dataset and create train and test sets
raw_dataset = load_dataset("nlphuji/flickr30k", split='test[:5000]')
train_test_split = raw_dataset.train_test_split(test_size=0.3, seed=42)
train = train_test_split['train']
test = train_test_split['test']


class CaptionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['image'], item['caption']
    


def collate_fn(batch):
    # batch is a list of (patch_embeddings, text_embeddings, target_ids, masks) tuples
    # Each item in batch is already processed by CaptionDataset

    # Stack all tensors
    patch_embeddings = torch.stack([item[0] for item in batch])  # [batch_size, num_captions, 49, 768]
    text_embeddings = torch.stack([item[1] for item in batch])   # [batch_size, num_captions, 18, 512]
    target_ids = torch.stack([item[2] for item in batch])        # [batch_size, num_captions, 18]
    masks = torch.stack([item[3] for item in batch])            # [batch_size, num_captions, 18]

    # Reshape to match your current output
    batch_size = patch_embeddings.size(0)
    num_captions = patch_embeddings.size(1)

    patch_embeddings = patch_embeddings.view(-1, 49, 768)  # [batch_size*num_captions, 49, 768]
    text_embeddings = text_embeddings.view(-1, 18, 512)    # [batch_size*num_captions, 18, 512]
    target_ids = target_ids.view(-1, 18)                   # [batch_size*num_captions, 18]
    masks = masks.view(-1, 18)                             # [batch_size*num_captions, 18]

    return patch_embeddings, text_embeddings, target_ids, masks