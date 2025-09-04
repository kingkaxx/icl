import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ModelConfig:
    input_dim = 5          # d=5
    max_context = 20       
    noise_level = 0.1      
    hidden_dim = 128       
    attention_dim = 128    
    learning_rate = 0.001  
    weight_decay = 1e-4    
    batch_size = 64        
       

config = ModelConfig()


#data generation

class TaskGenerator:
    def __init__(self, dim=5, noise_level=0.1, variance_scales=None):

        self.dim = dim  
        self.noise_level = noise_level  #noise added to outputs
        self.variance_scales = None if variance_scales is None else np.asarray(variance_scales, dtype=np.float32)

    def generate_weights(self):
        return np.random.randn(self.dim).astype(np.float32)  #shape- (dim,)

    def generate_inputs(self, num_samples):
        inputs = np.random.randn(num_samples, self.dim).astype(np.float32) 
        if self.variance_scales is not None:
            inputs = inputs * self.variance_scales  #scaling

        return inputs  

    def generate_task(self, num_context, add_noise=True):
        weights = self.generate_weights()  
        inputs = self.generate_inputs(num_context + 1)  #shape- (K+1, dim)
        noise = np.random.randn(num_context + 1).astype(np.float32) * self.noise_level if add_noise else 0.0
        outputs = inputs @ weights + noise  #shape- (K+1,)

        return inputs, outputs, weights

def prepare_tokens_and_targets(inputs, outputs):
    num_steps, dim = inputs.shape  
    previous_output = 0.0  #initial output is 0
    tokens = []
    targets = []


    for t in range(num_steps):
        #token creation by combining current input (x_t) and previous output (y_[t-1])
        token = np.concatenate([inputs[t], np.array([previous_output], dtype=np.float32)])
        tokens.append(token)
        targets.append(outputs[t])
        previous_output = outputs[t]
    
    return np.stack(tokens, axis=0).astype(np.float32), np.asarray(targets, dtype=np.float32)


class LinearAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        #linear layers for Q,K,V,O
        self.query_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.output_layer = nn.Linear(attention_dim, hidden_dim, bias=False)

    @staticmethod
    def activation(x):
        return F.elu(x) + 1.0

    def forward(self, hidden):
        queries = self.activation(self.query_layer(hidden)) 
        keys = self.activation(self.key_layer(hidden))
        values = self.value_layer(hidden)



        #cumulative sums for past information 
        keys_cumsum = torch.cumsum(keys, dim=1)  #sum keys upto each step
        key_value_product = keys * values 
        kv_cumsum = torch.cumsum(key_value_product, dim=1)  


        #attention scores: (Q * KV_cumsum) / (Q * K_cumsum)
        numerator = (queries * kv_cumsum).sum(dim=-1, keepdim=True)  
        denominator = (queries * keys_cumsum).sum(dim=-1, keepdim=True) + 1e-6 
        attention_scores = numerator / denominator 
        output = self.output_layer(attention_scores.expand(-1, -1, values.size(-1)))  #shape- (batch, sequence_length, hidden_dim)
        
        return output

class ICLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, attention_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim) 
        self.attention = LinearAttentionLayer(hidden_dim, attention_dim)  

        self.norm = nn.LayerNorm(hidden_dim)  
        self.output_head = nn.Linear(hidden_dim, 1)  #scalar output

    def forward(self, tokens):
        hidden = self.embedding(tokens)  
        hidden = self.attention(hidden) + hidden 

        hidden = self.norm(hidden)  
        predictions = self.output_head(hidden).squeeze(-1) 
        
        return predictions


#train

def compute_masked_mse(predictions, targets, mask):
 
    squared_error = (predictions - targets) ** 2 * mask
    
    return squared_error.sum() / mask.sum()


def train_on_isotropic_tasks(model, config):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    task_generator = TaskGenerator(dim=config.input_dim, noise_level=config.noise_level, variance_scales=None)
    

        



if __name__ == "__main__":
    input_dim = config.input_dim + 1
    model = ICLModel(input_dim=input_dim, hidden_dim=config.hidden_dim, attention_dim=config.attention_dim)

    