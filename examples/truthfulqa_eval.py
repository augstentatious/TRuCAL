"""
TruthfulQA Evaluation Script for CAL-TRM
Uses DistilBERT embeddings + vulnerability detection for deception proxy
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

from cal import UnifiedCAL_TRM

# Configuration
d_model = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = UnifiedCAL_TRM(d_model=d_model)
model.to(device)
model.eval()

# Initialize tokenizer and embedding model
print("Loading tokenizer and embedding model...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
embed_model = AutoModel.from_pretrained('distilbert-base-uncased')
embed_model.to(device)
embed_model.eval()

# Projection layer to match DistilBERT's hidden size (768) to d_model
proj = nn.Linear(768, d_model)
proj.to(device)

# Load TruthfulQA dataset
print("Loading TruthfulQA dataset...")
try:
    dataset = load_dataset("truthful_qa", "generation")
    split = "validation"
    data = dataset[split]
    print(f"Loaded {len(data)} examples from '{split}' split")
    
    # Limit to 100 examples for evaluation
    data = data.select(range(min(100, len(data))))
    print(f"Evaluating on {len(data)} examples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Evaluation metrics
total_examples = 0
v_t_scores = []
triggered_count = 0
cycles_run_list = []
coherence_scores = []

print("\nStarting evaluation...")
print("-" * 50)

for i, example in enumerate(data):
    question = example['question']
    correct_answers = example.get('correct_answers', [])
    incorrect_answers = example.get('incorrect_answers', [])
    
    if not question:
        continue
        
    total_examples += 1
    
    # Prepare input
    input_text = f"Question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', 
                                   padding=True, truncation=True, max_length=128)
    input_ids = input_ids.to(device)
    
    if input_ids.size(1) == 0:
        print(f"Skipping example {i} due to empty input")
        continue
    
    # Get embeddings
    with torch.no_grad():
        bert_output = embed_model(input_ids)
        embeddings = bert_output.last_hidden_state
    
    projected_embeddings = proj(embeddings)
    
    # Run through CAL-TRM model
    with torch.no_grad():
        output, metadata = model(projected_embeddings, return_metadata=True)
    
    # Collect metrics
    if 'v_t_score' in metadata and metadata['v_t_score'] is not None:
        v_t_batch = metadata['v_t_score']
        if torch.is_tensor(v_t_batch):
            v_t_mean = v_t_batch.mean().item()
            v_t_scores.append(v_t_mean)
        else:
            v_t_scores.append(float(v_t_batch))
    
    if metadata.get('confessional_triggered', False):
        triggered_count += 1
        
        if 'cycles_run' in metadata:
            cycles_run_list.append(metadata['cycles_run'])
        
        if 'coherence_score' in metadata:
            coherence_scores.append(metadata['coherence_score'])
    
    # Progress update
    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(data)} examples...")

# Final results
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Total examples evaluated: {total_examples}")
print(f"Confessional triggered: {triggered_count} ({100*triggered_count/total_examples:.1f}%)")

if v_t_scores:
    print(f"\nVulnerability Scores (v_t):")
    print(f"  Mean: {np.mean(v_t_scores):.4f}")
    print(f"  Std:  {np.std(v_t_scores):.4f}")
    print(f"  Min:  {np.min(v_t_scores):.4f}")
    print(f"  Max:  {np.max(v_t_scores):.4f}")

if cycles_run_list:
    print(f"\nConfessional Cycles (when triggered):")
    print(f"  Mean: {np.mean(cycles_run_list):.2f}")
    print(f"  Max:  {max(cycles_run_list)}")

if coherence_scores:
    print(f"\nCoherence Scores (when triggered):")
    print(f"  Mean: {np.mean(coherence_scores):.4f}")

print("\n" + "=" * 50)
print("Evaluation complete!")
