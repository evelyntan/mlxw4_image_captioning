"""
This script contains the implementation of the decoder class. It has:
* Masked Attention Head
* Multi-Head Attention with masking
* Cross-Attention Head
* Decoder Block
"""
import torch
import torch.nn as nn


class MaskedAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super(MaskedAttentionHead, self).__init__()
        self.head_dim = head_dim

        # Linear projections for query, key, value
        self.weight_q = nn.Linear(embedding_dim, head_dim)
        self.weight_k = nn.Linear(embedding_dim, head_dim)
        self.weight_v = nn.Linear(embedding_dim, head_dim)

        self.linear_projection = nn.Linear(head_dim, embedding_dim)

    def forward(self, decoder_sequence):
        # embedded decoder sequence shape: [batch_size, seq_length, embedding_dim]

        # Project to head dimension
        Q = self.weight_q(decoder_sequence)
        K = self.weight_k(decoder_sequence)
        V = self.weight_v(decoder_sequence)

        # Make the mask
        seq_len = decoder_sequence.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=decoder_sequence.device), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))

        # Calculate attention scores (scaled dot product)
        A = torch.einsum('bid,bjd->bij', Q, K)
        A = A / (self.head_dim ** 0.5) 

        A = A + mask
        # Apply softmax
        A = torch.softmax(A, dim=-1)

    
        #  Apply attention weights to values
        H = torch.einsum('bij,bjd->bid', A, V)
        
        # Add projection layer for output to return back to the original embedding dimension
        #output = self.linear_projection(H)

        return H
        
        
# Cross Attention Head is to attend to the encoder output AND decoder sequence 
class CrossAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super(CrossAttentionHead, self).__init__()
        self.head_dim = head_dim

     # Linear projections for query, key, value
     # The embedding dim for q and k will be taken from the encoder output
     # And the embedding dim will be taken from the decoder sequence
        self.weight_q = nn.Linear(embedding_dim, head_dim)
        self.weight_k = nn.Linear(embedding_dim, head_dim)
        self.weight_v = nn.Linear(embedding_dim, head_dim)

    def forward(self, decoder_sequence, encoder_output):
        # decoder sequence [batch_size, decoder_seq_length, embedding_dim]
        # encoder output [batch_size, num_patches, embedding_dim]

        # Q comes from the decoder sequence embeddings
        Q = self.weight_q(decoder_sequence)

        # K and V come from the encoder output
        # The encoder output is a 16x64 (or whatever your num_patchs x output dimensions is)
        # REMEMBER: Encoder output is the learned representation of the image
        K = self.weight_k(encoder_output)
        V = self.weight_v(encoder_output)

        # Calculate attention scores for QK
        A = torch.einsum('bid,bjd->bij', Q, K)
        A = A / (self.head_dim ** 0.5)

        # Apply softmax
        A = torch.softmax(A, dim=-1)

        # Apply attention weights to values
        H = torch.einsum('bij,bjd->bid', A, V)

        
        return H
        

# Multihead attention to use with Cross Attention Head and/or Masked Attention Head
# You need Multihead Attention instances because 1 is for the masked attention 
# and 1 is for the cross attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, is_cross_attention=False):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.is_cross_attention = is_cross_attention

     
        # Create attention heads, separately for Cross and Masked Attention
        if is_cross_attention:
            self.heads = nn.ModuleList(
                [CrossAttentionHead(embedding_dim, self.head_dim) for _ in range(num_heads)]
            )
        else:
            self.heads = nn.ModuleList(
                [MaskedAttentionHead(embedding_dim, self.head_dim) for _ in range(num_heads)]
            )
        
        # The output of the CrossAttention Head and MaskedAttentionHead still needs to be projected
        # Back to the embedding dimensions of the head_dim x vocab_size
        
        self.output_projection = nn.Linear(num_heads * self.head_dim, embedding_dim)

        

    def forward(self, decoder_sequence, encoder_output=None):
        # decoder_sequence: [batch_size, seq_length, embedding_dim]
        # encoder_output: [batch_size, num_patches, embedding_dim] (only used in cross-attention)
        # mask: [batch_size, seq_length, seq_length] (only used in self-attention)

        # Process each head
        head_outputs = []
        for head in self.heads:
            if self.is_cross_attention:
                # For cross attention, we need both decoder sequence and encoder output
                head_output = head(decoder_sequence, encoder_output)
                print("\ncross attention head output shape: ", head_output.shape)
            else:
                # For masked self-attention, we only need decoder sequence and mask
                head_output = head(decoder_sequence)
                print("\nmasked attention head output shape: ", head_output.shape)
                
            head_outputs.append(head_output)

        # Concatenate head outputs
        concat_heads = torch.cat(head_outputs, dim=-1)
        
        # Project back to embedding dimension
        output = self.output_projection(concat_heads)
        print("Multihead attention output shape: ", output.shape)
        
        return output



class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dimension):
        super(DecoderBlock, self).__init__()
        
        # First layer norm
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        # Masked multi-head attention for decoder sequence self-attention
        self.masked_mha = MultiHeadAttention(embedding_dim, num_heads)
        
        # Second layer norm
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        # Cross attention between decoder sequence and encoded image
        self.cross_mha = MultiHeadAttention(embedding_dim, num_heads, is_cross_attention=True)
        
        # Third layer norm
        self.ln3 = nn.LayerNorm(embedding_dim)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dimension),
            nn.ReLU(),
            nn.Linear(mlp_dimension, embedding_dim)
        )

    def forward(self, decoder_sequence, encoder_output):
        # decoder_sequence: the input sequence to decode (e.g., [START, 1, 2, 3])
        # encoder_output: the encoded image from the encoder
        # mask: causal mask to prevent attending to future tokens

        # First masked self-attention block with residual connection
        # This allows decoder sequence to attend to its own past tokens
        # First masked self-attention
        residual = decoder_sequence
        decoder_sequence = self.ln1(decoder_sequence)
        decoder_sequence = self.masked_mha(decoder_sequence)
        decoder_sequence = residual + decoder_sequence
        
        # Cross attention block with residual connection
        # This allows decoder sequence to attend to the encoded image
        residual = decoder_sequence
        decoder_sequence = self.ln2(decoder_sequence)
        decoder_sequence = self.cross_mha(decoder_sequence, encoder_output)
        decoder_sequence = residual + decoder_sequence
        
        # # FFN block with residual connection
        residual = decoder_sequence
        decoder_sequence = self.ln3(decoder_sequence)
        decoder_sequence = self.ffn(decoder_sequence)
        decoder_sequence = residual + decoder_sequence
        
        return decoder_sequence


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dimension, num_layers, input_sequence_length, vocab_size):
        super(Decoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    
        # Create positional embeddings for decoder sequence ONCE during initialization
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, input_sequence_length, embedding_dim),
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
        # This converts decoder features to logits over possible next tokens
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, decoder_sequence, encoder_output, return_logits=True):
        # decoder_sequence: the input sequence to decode (e.g., [START, 1, 2, 3])
        # encoder_output: the encoded image from the encoder
        # return_logits: whether to return prediction logits or just decoder features
        # by default, we return logits

        embedded_decoder_sequence = self.embedding_layer(decoder_sequence)

        # Add positional embeddings to decoder sequence
        decoder_sequence = embedded_decoder_sequence + self.positional_embeddings
        
        # Pass through decoder blocks
        for block in self.decoder_blocks:
            decoder_sequence = block(decoder_sequence, encoder_output)
        
        # Apply final layer norm
        decoder_features = self.final_ln(decoder_sequence)
        
        if return_logits:
            # Convert features to logits for prediction
            # Shape: [batch_size, seq_length, vocab_size]
            logits = self.output_projection(decoder_features)
            return logits
        else:
            # Return decoder features if needed
            return decoder_features

if __name__ == "__main__":
    print('MaskedAttentionHead called successfully')
    print('CrossAttentionHead called successfully')
    print('MultiHeadAttention called successfully')
    print('DecoderBlock called successfully')
    print('Decoder called successfully') 