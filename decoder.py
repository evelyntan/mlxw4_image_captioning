import torch.nn as nn
import torch
# Create Masked Self Attention Head
class MaskedAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super(MaskedAttentionHead, self).__init__()
        self.head_dim = head_dim

        # Linear projections for query, key, value
        self.weight_q = nn.Linear(embedding_dim, head_dim)
        self.weight_k = nn.Linear(embedding_dim, head_dim)
        self.weight_v = nn.Linear(embedding_dim, head_dim)

        self.linear_projection = nn.Linear(head_dim, embedding_dim)

    def forward(self, decoder_sequence, input_sequence_length, padding_mask=None, ):
        # embedded decoder sequence shape: [batch_size, seq_length, embedding_dim]

        #print('Decoder sequence shape:', decoder_sequence.shape)

        # Project to head dimension
        Q = self.weight_q(decoder_sequence)
        K = self.weight_k(decoder_sequence)
        V = self.weight_v(decoder_sequence)

        # Make the mask
        decoder_sequence_length = decoder_sequence.shape[1]
        #mask = torch.triu(torch.ones(decoder_sequence_length, decoder_sequence_length, device=decoder_sequence.device), diagonal=1)
        #print("Causal mask shape:", mask.shape)
        #mask = mask.masked_fill(mask==1, float('-inf'))

        mask = torch.zeros(decoder_sequence_length, decoder_sequence_length, device=decoder_sequence.device)
        
        # Only mask the text portion (after image patches)
        num_image_patches = 49  # CLIP ViT-B/32 has 49 patches
        text_start_idx = num_image_patches
        
        # Create causal mask for text portion only
        text_mask = torch.triu(torch.ones(decoder_sequence_length - text_start_idx, decoder_sequence_length - text_start_idx, 
                                        device=decoder_sequence.device), diagonal=1)
        text_mask = text_mask.masked_fill(text_mask==1, float('-inf'))
        
        # Place the text mask in the bottom-right corner of the full mask
        mask[text_start_idx:, text_start_idx:] = text_mask
        

        # Calculate attention scores (scaled dot product)
        A = torch.einsum('bid,bjd->bij', Q, K)
        A = A / (self.head_dim ** 0.5) 

        A = A + mask

        
        # Apply padding mask if provided
        if padding_mask is not None:
            #print('Padding mask shape:', padding_mask.shape)
            # Convert padding mask to attention mask
            padding_mask = padding_mask.expand(-1, A.size(1), -1)
            A = A.masked_fill(padding_mask, float('-inf'))
            
        # Apply softmax
        A = torch.softmax(A, dim=-1)

    
        #  Apply attention weights to values
        H = torch.einsum('bij,bjd->bid', A, V)

        return H

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

     
        self.heads = nn.ModuleList(
            [MaskedAttentionHead(embedding_dim, self.head_dim) for _ in range(num_heads)]
            )
        
        # The output of the CrossAttention Head and MaskedAttentionHead still needs to be projected
        # Back to the embedding dimensions of the head_dim x vocab_size
        
        self.output_projection = nn.Linear(num_heads * self.head_dim, embedding_dim)


    def forward(self, decoder_sequence, padding_mask=None):
        # decoder_sequence: [batch_size, seq_length, embedding_dim]
        # encoder_output: [batch_size, num_patches, embedding_dim] (only used in cross-attention)
        # mask: [batch_size, seq_length, seq_length] (only used in self-attention)

        # Process each head
        head_outputs = []
        for head in self.heads:
                # For masked self-attention, we only need decoder sequence and mask
                head_output = head(decoder_sequence, padding_mask)
                #print("\nmasked attention head output shape: ", head_output.shape)
                
                head_outputs.append(head_output)

        # Concatenate head outputs
        concat_heads = torch.cat(head_outputs, dim=-1)
        
        # Project back to embedding dimension
        output = self.output_projection(concat_heads)
        #print("Multihead attention output shape: ", output.shape)
        
        return output



class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dimension):
        super(DecoderBlock, self).__init__()
        
        # First layer norm
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        # Masked multi-head attention for decoder sequence self-attention
        self.masked_mha = MultiHeadAttention(embedding_dim, num_heads)
        
        
        # Third layer norm
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dimension),
            nn.ReLU(),
            nn.Linear(mlp_dimension, embedding_dim)
        )

    def forward(self, decoder_sequence, padding_mask=None):
        # decoder_sequence: the input sequence to decode (e.g., [START, 1, 2, 3])
        # encoder_output: the encoded image from the encoder
        # mask: causal mask to prevent attending to future tokens

        # First masked self-attention block with residual connection
        # This allows decoder sequence to attend to its own past tokens
        # First masked self-attention
        residual = decoder_sequence
        decoder_sequence = self.ln1(decoder_sequence)
        decoder_sequence = self.masked_mha(decoder_sequence, padding_mask)
        decoder_sequence = residual + decoder_sequence

        
        # # FFN block with residual connection
        residual = decoder_sequence
        decoder_sequence = self.ln2(decoder_sequence)
        decoder_sequence = self.ffn(decoder_sequence)
        decoder_sequence = residual + decoder_sequence
        
        return decoder_sequence


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dimension, num_layers, vocab_size):
        super(Decoder, self).__init__()
        
        # Store embedding dimension
        self.embedding_dim = embedding_dim
        
        # Projection layers for image and text features
        self.image_projection = nn.Linear(768, embedding_dim)  # CLIP image features are 768-dim
        self.text_projection = nn.Linear(512, embedding_dim)   # CLIP text features are 512-dim
        
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
        # Create positional embeddings - single embedding that will be added to each position
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1, embedding_dim),
            requires_grad=True
        )
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_dim, num_heads, mlp_dimension)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(embedding_dim)

        # Output projection to vocabulary size
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, text_embeddings, img_features, padding_mask=None, return_logits=True):
        # Project image and text features to same dimension
        img_features = self.image_projection(img_features)  # [batch, num_captions, num_patches, embedding_dim]
        text_embeddings = self.text_projection(text_embeddings)  # [batch, num_captions, seq_len, embedding_dim]
        #print(f"After projection:")
        #print(f"Image features shape: {img_features.shape}")
        #print(f"Text embeddings shape: {text_embeddings.shape}")

        # Reshape tensors to combine batch and caption dimensions
        batch_size = img_features.size(0)
        num_captions = img_features.size(1)
        #print(f"batch_size: {batch_size}, num_captions: {num_captions}")
        
        # Reshape image features: [batch, num_captions, num_patches, embedding_dim] -> [batch*num_captions, num_patches, embedding_dim]
        img_features = img_features.reshape(batch_size * num_captions, 49, self.embedding_dim)
        text_embeddings = text_embeddings.reshape(batch_size * num_captions, 18, self.embedding_dim)
        #print(f"After reshaping:")
        #print(f"Image features shape: {img_features.shape}")
        #print(f"Text embeddings shape: {text_embeddings.shape}")

        # Reshape padding mask if provided
        if padding_mask is not None:
            # Create a mask for image patches (all False since we want to attend to all patches)
            img_mask = torch.zeros(batch_size * num_captions, 49, dtype=torch.bool, device=padding_mask.device)
            # Reshape text padding mask
            text_mask = padding_mask.reshape(batch_size * num_captions, -1)
            # Concatenate masks
            padding_mask = torch.cat([img_mask, text_mask], dim=1)

        # Concatenate image features and text embeddings
        decoder_inputs = torch.cat([img_features, text_embeddings], dim=1)  # [batch*num_captions, num_patches+seq_len, embedding_dim]
        #print(f"Decoder inputs shape: {decoder_inputs.shape}")
        
        # Pass through decoder blocks
        for block in self.decoder_blocks:
            decoder_inputs = block(decoder_inputs, padding_mask)
        
        # Apply final layer norm
        decoder_features = self.final_ln(decoder_inputs)

        # Get text hidden states only (18 tokens, not 32!)
        text_hidden = decoder_features[:, -18:, :]  # [batch*num_captions, seq_len, embedding_dim]
        #print(f"Text hidden states shape: {text_hidden.shape}")
        
        if return_logits:
            # Convert features to logits for prediction
            # Shape: [batch*num_captions, seq_len, vocab_size]
            logits = self.output_projection(text_hidden)
            #print(f"Logits shape: {logits.shape}")
            return logits
        else:
            # Return decoder features if needed
            return text_hidden
        

if __name__ == "__main__":
    # Test parameters for sanity checking 
    embedding_dim = 256
    num_heads = 8
    mlp_dimension = 2048
    num_layers = 4
    vocab_size = 49408  # CLIP's vocab size
    
    # Create decoder
    decoder = Decoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dimension=mlp_dimension,
        num_layers=num_layers,
        vocab_size=vocab_size
    )
    
    # Create sample inputs
    batch_size = 2
    num_captions = 3
    # Create inputs with correct dimensions for concatenation
    img_features = torch.randn(batch_size, num_captions, 49, 768)  # [batch, num_captions, num_patches, hidden_dim]
    text_embeddings = torch.randn(batch_size, num_captions, 18, 512)  # [batch, num_captions, seq_len, hidden_dim]
    
    # Reshape by flattening for concatenation
    img_features = img_features.view(batch_size * num_captions, 49, 768)  # [batch*num_captions, num_patches, hidden_dim]
    text_embeddings = text_embeddings.view(batch_size * num_captions, 18, 512)  # [batch*num_captions, seq_len, hidden_dim]
    
    # Test forward pass
    logits = decoder(text_embeddings, img_features)
    
    print("\nTest completed successfully!")
    print(f"Input shapes:")
    print(f"Image features: {img_features.shape}")
    print(f"Text embeddings: {text_embeddings.shape}")
    print(f"Output logits: {logits.shape}")
        

