"""
Parameter-Efficient Fine-tuning for wav2vec 2.0

WHY THIS IMPROVEMENT:
Traditional fine-tuning updates all model parameters, which can lead to 
overfitting when labeled data is extremely limited (10min-1hr). Parameter-efficient
methods reduce the number of trainable parameters while maintaining performance:

1. LoRA (Low-Rank Adaptation) - Add small trainable matrices
2. Adapters - Insert small bottleneck layers
3. Progressive Unfreezing - Gradually unfreeze layers during training
4. Selective Layer Training - Train only the most important layers

EXPECTED BENEFITS:
- Reduced overfitting with limited data
- Faster training and less memory usage
- Better generalization to unseen data
- Easier transfer across different domains/languages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer
    
    WHY: Instead of fine-tuning the full weight matrix W, we learn a low-rank
    decomposition: W + AB where A and B are much smaller matrices.
    This dramatically reduces trainable parameters while maintaining expressivity.
    
    Original paper: "LoRA: Low-Rank Adaptation of Large Language Models"
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        """
        Args:
            original_layer: The original linear layer to adapt
            rank: Rank of the adaptation (lower = fewer parameters)
            alpha: Scaling factor for the adaptation
            dropout: Dropout probability for adaptation layers
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create low-rank adaptation matrices
        # A: (in_features, rank), B: (rank, out_features)
        # Final adaptation: x @ A @ B = (batch, in_features) @ (in_features, rank) @ (rank, out_features)
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming uniform, B with zeros (ensures adaptation starts at 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Calculate scaling factor
        self.scaling = alpha / rank
        
        logger.info(f"LoRA layer: {original_layer.in_features}x{original_layer.out_features} "
                   f"-> rank {rank} ({self.count_parameters()} parameters)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original_output + scaled_adaptation
        
        WHY: We add the low-rank adaptation to the original frozen output.
        This preserves the pre-trained knowledge while allowing task-specific adaptation.
        """
        # Original frozen transformation
        original_output = self.original_layer(x)
        
        # Low-rank adaptation: x -> A -> dropout -> B -> scale
        adaptation = self.lora_A(x)
        adaptation = self.dropout(adaptation)
        adaptation = self.lora_B(adaptation)
        adaptation = adaptation * self.scaling
        
        return original_output + adaptation
    
    def count_parameters(self) -> int:
        """Count trainable parameters in this LoRA layer"""
        return sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight])

class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    
    WHY: Adapters insert small bottleneck layers between transformer blocks.
    They learn task-specific transformations while keeping the backbone frozen.
    This is particularly effective for domain adaptation.
    """
    
    def __init__(self, hidden_size: int, bottleneck_size: int = 64, dropout: float = 0.1):
        """
        Args:
            hidden_size: Size of the hidden representations
            bottleneck_size: Size of the bottleneck (smaller = more parameter efficient)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        
        # Down-projection: hidden_size -> bottleneck_size
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up-projection: bottleneck_size -> hidden_size  
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize to near-identity (start with minimal impact)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
        logger.info(f"Adapter layer: {hidden_size} -> {bottleneck_size} -> {hidden_size} "
                   f"({self.count_parameters()} parameters)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        WHY: Residual connection ensures that without training, the adapter
        has minimal impact. During training, it learns task-specific adjustments.
        """
        # Residual connection: x + adapter(x)
        residual = x
        
        # Adapter transformation: down -> activate -> dropout -> up
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        return residual + x
    
    def count_parameters(self) -> int:
        """Count trainable parameters in this adapter"""
        return sum(p.numel() for p in self.parameters())

class ParameterEfficientWav2Vec2(nn.Module):
    """
    Parameter-efficient version of wav2vec 2.0
    
    WHY: Combines multiple parameter-efficient techniques to minimize
    trainable parameters while maintaining model capacity for low-resource
    speech recognition tasks.
    """
    
    def __init__(self, base_model, adaptation_method: str = "lora", 
                 lora_rank: int = 16, adapter_bottleneck: int = 64):
        """
        Args:
            base_model: Pre-trained wav2vec 2.0 model
            adaptation_method: Type of adaptation ('lora', 'adapter', 'hybrid')
            lora_rank: Rank for LoRA adaptation
            adapter_bottleneck: Bottleneck size for adapter layers
        """
        super().__init__()
        
        self.base_model = base_model
        self.adaptation_method = adaptation_method
        
        # Freeze the base model initially
        self.freeze_base_model()
        
        # Apply parameter-efficient adaptations
        if adaptation_method == "lora":
            self._apply_lora_adaptation(lora_rank)
        elif adaptation_method == "adapter":
            self._apply_adapter_adaptation(adapter_bottleneck)
        elif adaptation_method == "hybrid":
            self._apply_hybrid_adaptation(lora_rank, adapter_bottleneck)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Parameter-efficient model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    def freeze_base_model(self):
        """
        Freeze all parameters in the base model
        
        WHY: We want to preserve the pre-trained representations and only
        learn task-specific adaptations with minimal parameters.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Frozen base model parameters")
    
    def _apply_lora_adaptation(self, rank: int):
        """Apply LoRA to attention layers in the transformer"""
        logger.info(f"Applying LoRA adaptation with rank {rank}")
        
        # Apply LoRA to transformer attention layers
        for layer_idx, layer in enumerate(self.base_model.wav2vec2.encoder.layers):
            # Self-attention projections
            if hasattr(layer.attention.self, 'query'):
                layer.attention.self.query = LoRALayer(layer.attention.self.query, rank)
            if hasattr(layer.attention.self, 'key'):
                layer.attention.self.key = LoRALayer(layer.attention.self.key, rank)
            if hasattr(layer.attention.self, 'value'):
                layer.attention.self.value = LoRALayer(layer.attention.self.value, rank)
            
            # Output projection
            if hasattr(layer.attention.output, 'dense'):
                layer.attention.output.dense = LoRALayer(layer.attention.output.dense, rank)
    
    def _apply_adapter_adaptation(self, bottleneck_size: int):
        """Apply adapter layers to transformer blocks"""
        logger.info(f"Applying adapter adaptation with bottleneck size {bottleneck_size}")
        
        # Add adapters after each transformer layer
        for layer_idx, layer in enumerate(self.base_model.wav2vec2.encoder.layers):
            hidden_size = layer.attention.self.query.in_features
            layer.adapter = AdapterLayer(hidden_size, bottleneck_size)
    
    def _apply_hybrid_adaptation(self, lora_rank: int, adapter_bottleneck: int):
        """Apply both LoRA and adapter adaptations"""
        logger.info(f"Applying hybrid adaptation: LoRA rank {lora_rank}, adapter bottleneck {adapter_bottleneck}")
        
        self._apply_lora_adaptation(lora_rank)
        self._apply_adapter_adaptation(adapter_bottleneck)
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        """Forward pass through the parameter-efficient model"""
        
        # If using adapters, we need to modify the forward pass
        if self.adaptation_method in ["adapter", "hybrid"]:
            return self._forward_with_adapters(input_values, attention_mask, **kwargs)
        else:
            # LoRA is integrated into the layers, so normal forward pass
            return self.base_model(input_values, attention_mask=attention_mask, **kwargs)
    
    def _forward_with_adapters(self, input_values, attention_mask=None, **kwargs):
        """Modified forward pass that includes adapter layers"""
        
        # Extract features using the feature extractor
        extract_features = self.base_model.wav2vec2.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        
        # Project features if needed
        if self.base_model.wav2vec2.feature_projection is not None:
            extract_features = self.base_model.wav2vec2.feature_projection(extract_features)
        
        # Pass through encoder layers with adapters
        hidden_states = extract_features
        
        for layer in self.base_model.wav2vec2.encoder.layers:
            # Original layer computation
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            # Apply adapter if present
            if hasattr(layer, 'adapter'):
                hidden_states = layer.adapter(hidden_states)
        
        # Final projection to vocabulary
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(hidden_states)
        else:
            logits = hidden_states
        
        return {"logits": logits, "hidden_states": hidden_states}

class ProgressiveUnfreezing:
    """
    Progressive unfreezing strategy for low-resource fine-tuning
    
    WHY: Instead of unfreezing all layers at once, we gradually unfreeze
    layers starting from the top. This prevents early layers (which capture
    general speech features) from being disrupted by limited task-specific data.
    """
    
    def __init__(self, model, unfreeze_schedule: List[int]):
        """
        Args:
            model: The model to apply progressive unfreezing to
            unfreeze_schedule: List of training steps at which to unfreeze layers
        """
        self.model = model
        self.unfreeze_schedule = sorted(unfreeze_schedule)
        self.current_step = 0
        self.layers_unfrozen = 0
        
        # Get transformer layers
        if hasattr(model, 'wav2vec2'):
            self.transformer_layers = model.wav2vec2.encoder.layers
        else:
            self.transformer_layers = model.encoder.layers
        
        self.total_layers = len(self.transformer_layers)
        
        logger.info(f"Progressive unfreezing initialized: {self.total_layers} layers, "
                   f"schedule: {unfreeze_schedule}")
    
    def step(self, training_step: int):
        """
        Update unfreezing based on training step
        
        WHY: Gradual unfreezing allows lower layers to stabilize before
        upper layers start adapting. This is crucial with limited data.
        """
        self.current_step = training_step
        
        # Check if we should unfreeze more layers
        while (self.layers_unfrozen < len(self.unfreeze_schedule) and 
               training_step >= self.unfreeze_schedule[self.layers_unfrozen]):
            
            self._unfreeze_next_layer()
            self.layers_unfrozen += 1
    
    def _unfreeze_next_layer(self):
        """Unfreeze the next layer (from top to bottom)"""
        if self.layers_unfrozen < self.total_layers:
            # Unfreeze from the top (closest to output)
            layer_idx = self.total_layers - 1 - self.layers_unfrozen
            layer = self.transformer_layers[layer_idx]
            
            for param in layer.parameters():
                param.requires_grad = True
            
            logger.info(f"Unfroze layer {layer_idx} at step {self.current_step}")

def apply_parameter_efficient_adaptation(model, method: str = "lora", **kwargs):
    """
    Factory function to apply parameter-efficient adaptations
    
    Args:
        model: Pre-trained wav2vec 2.0 model
        method: Adaptation method ('lora', 'adapter', 'hybrid', 'progressive')
        **kwargs: Method-specific parameters
        
    Returns:
        Adapted model or unfreezing scheduler
    """
    if method in ["lora", "adapter", "hybrid"]:
        return ParameterEfficientWav2Vec2(model, method, **kwargs)
    elif method == "progressive":
        unfreeze_schedule = kwargs.get("unfreeze_schedule", [100, 300, 600, 1000])
        return ProgressiveUnfreezing(model, unfreeze_schedule)
    else:
        raise ValueError(f"Unknown adaptation method: {method}")

def count_trainable_parameters(model) -> Dict[str, int]:
    """
    Count trainable parameters in the model
    
    Returns:
        Dictionary with parameter counts and ratios
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
    } 