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
    training_steps = 6000 
    eval_tasks = 500 

config = ModelConfig()


#data generation

class TaskGenerator:
    def __init__(self, dim=5, noise_level=0.1, variance_scales=None):

        self.dim = dim  
        self.noise_level = noise_level  #noise added to outputs
        self.variance_scales = None if variance_scales is None else np.asarray(variance_scales, dtype=np.float32)

    def generate_inputs(self, num_samples):
        inputs = np.random.randn(num_samples, self.dim).astype(np.float32) 
        if self.variance_scales is not None:
            inputs = inputs * self.variance_scales  #scaling

        return inputs  
    
    fixed_weights = np.random.randn(config.input_dim).astype(np.float32) 
    def generate_task(self, num_context, add_noise=True):
        weights = TaskGenerator.fixed_weights
        inputs = self.generate_inputs(num_context + 1)  #shape- (K+1, dim)
        noise = np.random.randn(num_context + 1).astype(np.float32) * self.noise_level if add_noise else 0.0
        outputs = inputs @ weights + noise  #shape- (K+1,)

        return inputs, outputs, weights

def prepare_tokens_and_targets(inputs, outputs):
    num_steps, dim = inputs.shape  #(K+1, dim) (21,5)
  
    #token creation by combining  input (x_t) and  output (y_t)
    z = np.zeros((num_steps, dim+1), dtype=np.float32)

    z[:num_steps-1, :dim] = inputs[:num_steps - 1]
    z[:num_steps-1, dim] = outputs[:num_steps - 1]

    z[-1, :dim] = inputs[-1]
    z[-1, dim] = 0.0
    
    return z
    
  

class LinearAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        #linear layers for Q,K,V,O
        self.query_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.output_layer = nn.Linear(attention_dim, hidden_dim, bias=False)

    
    def forward(self, hidden):
        queries = self.query_layer(hidden)
        keys = self.key_layer(hidden)
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

def compute_mse(predictions, targets):
 
    query_pred = (predictions[:, -1])
    Loss = F.mse_loss(query_pred, targets[:, -1])
    return Loss

def generate_task_batch(batch_size, max_context, task_generator):
    max_length = 0
    tasks = []

    #tasks with random context lengths
    for _ in range(batch_size):
        context_size = np.random.randint(1, max_context + 1)
        inputs, outputs, _ = task_generator.generate_task(context_size)
        Z = prepare_tokens_and_targets(inputs, outputs)
        tasks.append((Z, outputs[-1]))
        max_length = max(max_length, Z.shape[0])
    
    #padding
    input_dim = tasks[0][0].shape[1]
    tokens_batch = np.zeros((batch_size, max_length, input_dim), np.float32)
    targets_batch = np.zeros((batch_size, max_length), np.float32)
   
    

    for i, (Z, targets) in enumerate(tasks):
        length = Z.shape[0]
        tokens_batch[i, :length] = Z
        targets_batch[i] = targets ##

    return (torch.from_numpy(tokens_batch),
            torch.from_numpy(targets_batch))


def train_on_isotropic_tasks(model, config):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    task_generator = TaskGenerator(dim=config.input_dim, noise_level=config.noise_level, variance_scales=None)
    
    for step in range(1, config.training_steps + 1):
        tokens, targets = generate_task_batch(config.batch_size, config.max_context, task_generator)
       
        predictions = model(tokens)
        loss = compute_mse(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Training step {step}/{config.training_steps}, Loss: {loss.item():.4e}")

def evaluate_model_performance(model, context_sizes, num_tasks, dim, variance_scales):
    model.eval()
    task_generator = TaskGenerator(dim=dim, noise_level=0.0, variance_scales=variance_scales)
    results = {}
    with torch.no_grad():
        for context_size in context_sizes:
            errors = []
            for _ in range(num_tasks):
                inputs, outputs, _ = task_generator.generate_task(context_size)
                Z = prepare_tokens_and_targets(inputs, outputs)
                targets = np.array([outputs[-1]], dtype=np.float32)

                tokens_tensor = torch.from_numpy(Z)[None]
                targets_tensor = torch.from_numpy(targets)
                
                predictions = model(tokens_tensor)[0, -1].item()
             
                error = (predictions - targets_tensor) ** 2
                errors.append(error)
            results[context_size] = float(np.mean(errors))
    return results       


def plot_line_graph(x_values, y_values, title, x_label, y_label):
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar_graph(labels, values, title, y_label):
    
    x_positions = np.arange(len(labels))
    plt.figure()
    plt.bar(x_positions, values)
    plt.xticks(x_positions, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_dim = config.input_dim + 1
    model = ICLModel(input_dim=input_dim, hidden_dim=config.hidden_dim, attention_dim=config.attention_dim)
    
    train_on_isotropic_tasks(model, config)
    context_sizes = list(range(1, config.max_context + 1))
    isotropic_results = evaluate_model_performance(model, context_sizes, config.eval_tasks, config.input_dim, variance_scales=None)
    
    
    plot_line_graph(context_sizes, [isotropic_results[k] for k in context_sizes],
                    "Linear Attention on Isotropic Tasks (d=5)",
                    "Context Length (K)", "Test MSE")
    
    def compute_std(eigenvalues):
        
        return np.sqrt(np.array(eigenvalues, dtype=np.float32))

    anisotropic_configs = {
        "Isotropic [1,1,1,1,1]": compute_std([1, 1, 1, 1, 1]),
        "Mild [2,1,1,1,1]": compute_std([2, 1, 1, 1, 1]),
        "Strong [4,1,1,1,1]": compute_std([4, 1, 1, 1, 1]),
        "Very Strong [8,1,1,1,1]": compute_std([8, 1, 1, 1, 1]),
        "Skewed [4,2,1,0.5,0.25]": compute_std([4, 2, 1, 0.5, 0.25]),
    }
    task_labels = []
    mse_values = []
    fixed_context = 20
    for task_name, variance_scales in anisotropic_configs.items():
        results = evaluate_model_performance(model, [fixed_context], config.eval_tasks, config.input_dim, variance_scales)
        
        task_labels.append(task_name)
        mse_values.append(results[fixed_context])
    
    
    plot_bar_graph(task_labels, mse_values,
                   f"Anisotropic Generalization at K={fixed_context} (d=5)",
                   "Test MSE")