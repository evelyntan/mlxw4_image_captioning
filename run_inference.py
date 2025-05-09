import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from datasets import load_dataset
from inference import generate_and_visualize
from decoder import Decoder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run image captioning inference')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=256,
                      help='Dimension of embeddings')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--mlp_dimension', type=int, default=2048,
                      help='Dimension of MLP layers')
    parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of transformer layers')
    parser.add_argument('--vocab_size', type=int, default=49408,
                      help='Size of vocabulary')
    
    # Model weights
    parser.add_argument('--weights_path', type=str, default='decoder_08052002.pt',
                      help='Path to the trained model weights')
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='nlphuji/coco_captions',
                      help='Name of the dataset to use')
    parser.add_argument('--split', type=str, default='test',
                      help='Dataset split to use')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models and processors
    print("Loading CLIP model and processors...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize decoder with command line arguments
    print("Initializing decoder...")
    decoder = Decoder(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        mlp_dimension=args.mlp_dimension,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size
    ).to(device)

    # Load trained weights
    print(f"Loading trained weights from {args.weights_path}...")
    decoder.load_state_dict(torch.load(args.weights_path))
    decoder.eval()

    # Load test dataset
    print(f"Loading {args.split} dataset from {args.dataset_name}...")
    test_dataset = load_dataset(args.dataset_name, split=args.split)

    # Generate and visualize caption
    print("Generating caption...")
    generate_and_visualize(test_dataset, decoder, clip_model, clip_processor, tokenizer, device)

if __name__ == "__main__":
    main() 