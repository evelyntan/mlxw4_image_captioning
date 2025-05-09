import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import random
import time
from IPython.display import display, clear_output

def generate_and_visualize(test_dataset, decoder, clip_model, clip_processor, tokenizer, device):
    """
    Generate caption for a random image and visualize the results.
    
    Args:
        test_dataset: Dataset containing images and captions
        decoder: Trained decoder model
        clip_model: CLIP model
        clip_processor: CLIP processor
        tokenizer: CLIP tokenizer
        device: torch device
    """
    # Get a random image from the test set
    idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]
    image = sample['image']
    true_caption = sample['caption'][0]  # Get the first caption
    
    # Process image through CLIP
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    text_projection_layer = torch.nn.Linear(512, 256).to(device)
    
    with torch.no_grad():
        # Get image embeddings from CLIP
        patch_embeddings = clip_model.vision_model(**inputs).last_hidden_state[:, 1:, :].squeeze(0)
        
        # Project image features to the embedding dimension
        image_projection_layer = torch.nn.Linear(patch_embeddings.shape[-1], 256).to(device)
        img_features = image_projection_layer(patch_embeddings)
        img_features = img_features.unsqueeze(0)

        
    
    # Initialize with start token
    start_token = tokenizer.encode("<|startoftext|>")[0]
    end_token = tokenizer.encode("<|endoftext|>")[0]
    
    generated_tokens = [start_token]
    generated_caption = ""
    
    # Generate tokens one by one
    for _ in range(32):  # max length of 32 tokens
        # Convert generated tokens to tensor
        input_ids = torch.tensor([generated_tokens]).to(device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = clip_model.text_model.embeddings(input_ids).squeeze(0)
            text_embeddings = text_embeddings.unsqueeze(0)
            text_inputs = text_projection_layer(text_embeddings)
        
        # Get logits from decoder
        logits = decoder(text_inputs, img_features)

        # Get probabilities
        probabilities = torch.softmax(logits[0, -1], dim=-1)

        
        next_token = torch.multinomial(probabilities, 1).item()

        # Get next token (greedy decoding)
        #next_token = logits[0, -1].argmax(dim=-1).item()
        generated_tokens.append(next_token)
        
        # Decode the current token and add to caption
        current_word = tokenizer.decode([next_token], skip_special_tokens=True)
        generated_caption += current_word + " "
        
        # Print the current state of the caption
        print(f"\rGenerating caption: {generated_caption}", end="", flush=True)
        time.sleep(0.1)  # Add a small delay for visualization
        
        # Stop if we generate the end token
        if next_token == end_token:
            break
    
    print()  # New line after caption generation
    
    # Display the image and captions
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.figtext(0.5, 0.01, f"True Caption: {true_caption}", 
                ha='center', fontsize=12, wrap=True)
    plt.figtext(0.5, 0.05, f"\nGenerated Caption: {generated_caption}", 
                ha='center', fontsize=12, wrap=True)
    plt.show()