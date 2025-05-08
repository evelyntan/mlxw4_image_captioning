import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import random
from decoder import Decoder
from IPython.display import display, clear_output
import time

def generate_caption_with_visualization(image, decoder, clip_model, clip_processor, tokenizer, device, max_length=32):
    """
    Generate a caption for an image with word-by-word visualization.
    
    Args:
        image: PIL Image
        decoder: Trained decoder model
        clip_model: CLIP model
        clip_processor: CLIP processor
        tokenizer: CLIP tokenizer
        device: torch device
        max_length: maximum caption length
    
    Returns:
        str: Generated caption
    """
    # Process image through CLIP
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get image embeddings
        patch_embeddings = clip_model.vision_model(**inputs).last_hidden_state[:, 1:, :].squeeze(0)
        
        # Project image features to match text embedding dimension
        image_projection_layer = torch.nn.Linear(patch_embeddings.shape[-1], 512).to(device)
        img_features = image_projection_layer(patch_embeddings)
        
        # Initialize with start token
        start_token = tokenizer.bos_token_id
        current_tokens = torch.tensor([[start_token]]).to(device)
        
        # Initialize caption
        generated_caption = ""
        
        # Generate caption token by token
        for _ in range(max_length):
            # Get text embeddings for current sequence
            text_embeddings = clip_model.text_model.embeddings(current_tokens).squeeze(0)
            
            # Concatenate image and text features
            decoder_inputs = torch.cat([img_features.unsqueeze(0), text_embeddings.unsqueeze(0)], dim=1)
            
            # Get decoder output
            logits = decoder(decoder_inputs)
            
            # Get next token (take the last token's prediction)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            
            # Append to current sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Decode the current token
            current_word = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # Update the caption
            if current_word:  # Only add non-empty tokens
                generated_caption += current_word + " "
                # Print the current state of the caption
                print(f"\rGenerating caption: {generated_caption}", end="", flush=True)
                time.sleep(0.1)  # Add a small delay for visualization
            
            # Stop if we predict the end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n")  # New line after caption generation
    return generated_caption.strip()

def visualize_caption(image, true_caption, generated_caption):
    """
    Display the image with both true and generated captions.
    
    Args:
        image: PIL Image
        true_caption: str, ground truth caption
        generated_caption: str, model generated caption
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Generated: {generated_caption}\nTrue: {true_caption}')
    plt.show()

def run_inference(decoder_path=None):
    """
    Run inference on a random image from the test set.
    
    Args:
        decoder_path: str, path to the trained decoder weights (optional)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models and processors
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load the test dataset
    test_dataset = load_dataset("nlphuji/flickr30k", split="test[:5000]")
    
    # Initialize decoder
    embedding_dim = 512
    num_heads = 8
    mlp_dimension = 2048
    num_layers = 2
    vocab_size = 49408
    
    decoder = Decoder(embedding_dim, num_heads, mlp_dimension, num_layers, vocab_size).to(device)
    
    # Load trained weights if provided
    if decoder_path:
        decoder.load_state_dict(torch.load(decoder_path))
    decoder.eval()
    
    # Get a random image from the test set
    idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]
    image = sample['image']
    true_caption = sample['caption'][0]  # Get the first caption
    
    # Generate caption with visualization
    generated_caption = generate_caption_with_visualization(
        image, decoder, clip_model, clip_processor, tokenizer, device
    )
    
    # Display results
    visualize_caption(image, true_caption, generated_caption)
    
    return generated_caption, true_caption 