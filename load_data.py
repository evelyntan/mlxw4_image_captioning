# Import necessary libraries
from datasets import load_dataset, DatasetDict
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from torch.utils.data import Dataset, DataLoader
import torch

# Initialize CLIP processor and tokenizer
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# Load the dataset from Hugging Face
raw_dataset = load_dataset("nlphuji/flickr30k", split="test[:5000]", )

#raw_dataset = load_from_disk('./data')

# Split the dataset into training and testing sets
train_test_split = raw_dataset.train_test_split(test_size=0.3)
train = train_test_split['train']
test = train_test_split['test']

def preprocess(data):
    caption = [caption_list[0] for caption_list in data['caption']]
    image = data['image']

    # process the image
    img_tensor = processor(images=image, return_tensors='pt', padding=True)['pixel_values'].squeeze()
    print('IMG TENSOR SHAPE', img_tensor.shape) # channels, height, width
    
    # tokenize the caption 
    bos_token = tokenizer.bos_token
    caption_with_bos = [bos_token] + caption
    text_inputs = tokenizer(caption_with_bos, padding=True, return_tensors='pt')
    input_ids = text_inputs['input_ids'].squeeze()
    print('CAPTION LENGTH', len(caption))
    print('TEXT INPUT SHAPE', input_ids.shape)

    # tokenize the target caption
    eos_token = tokenizer.eos_token
    target_caption = caption + [eos_token]
    target_input = tokenizer(target_caption, padding=True, return_tensors='pt')
    target_input_ids = target_input['input_ids'].squeeze()
    print('TARGET INPUT IDS SHAPE', target_input_ids.shape)


    return img_tensor, input_ids, target_input_ids


