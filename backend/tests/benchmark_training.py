import os
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path to import model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spread.ml_analyzer import SpreadSequenceLSTM

def benchmark_training():
    print("=" * 50)
    print("ArbML PyTorch Deep Learning Benchmark")
    print("=" * 50)
    
    # 1. GPU Check
    has_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if has_gpu else 'cpu')
    print(f"\n[DIAGNOSTICO DE HARDWARE]")
    print(f"Device detectado pelo PyTorch: {device.type.upper()}")
    if not has_gpu:
        print("-> MOTIVO: O PyTorch instalado no seu ambiente virtual (.venv) é a versão CPU-only,")
        print("   ou você não possui uma GPU da NVIDIA com drivers CUDA instalados/configurados.")
        print("   Para instalar com suporte a GPU, use: pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
    else:
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")

    # 2. Simulate Massive Dataset
    # A multi-GB SQLite-backed tracker typically materializes down to a few million valid sequences.
    num_sequences = 1_000_000  # 1 Million sequences for the stress test
    seq_length = 15
    batch_size = 1024  # High batch size for stress testing
    
    print(f"\n[DIAGNOSTICO DE CARGA MAXIMA]")
    print(f"Gerando dataset sintético gigante para simular semanas de histórico pesado...")
    
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    # Generate X: (1,000,000, 15, 4) - 4 features: entry, exit, delta_entry, delta_exit
    num_features = 4
    X_tensor = torch.randn(num_sequences, seq_length, num_features, dtype=torch.float32)
    # Generate Y: (1,000,000, 2) - [Prob, ETA]
    Y_tensor = torch.randn(num_sequences, 2, dtype=torch.float32)
    Y_tensor[:, 0] = torch.sigmoid(Y_tensor[:, 0]) # Probabilities between 0 and 1
    
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    tensor_ram = end_mem - start_mem
    
    print(f"-> Sequências geradas: {num_sequences:,}")
    print(f"-> Consumo puro de RAM dos Tensors (Machine Learning): {tensor_ram:.2f} MB")
    print("   (Nota: o gargalo real continua sendo materializar o histórico bruto; o treino em si vira tensores compactos.)")

    # 3. Training Loop Benchmark
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SpreadSequenceLSTM(input_sz=num_features, hidden_sz=32, num_layers=2, dropout=0.2).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    criterion_prob = nn.BCEWithLogitsLoss()
    criterion_eta = nn.SmoothL1Loss()
    
    epochs = 2 # Just 2 epochs to measure speed per epoch
    print(f"\n[TREINAMENTO DE STRESS - {epochs} EPOCHS]")
    print(f"Processando {num_sequences:,} sequências por época (Batch Size: {batch_size})...")
    
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for batch_i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            pred_prob, pred_eta = model(batch_x)
            
            target_prob = batch_y[:, 0].unsqueeze(1)
            target_eta = batch_y[:, 1].unsqueeze(1)
            
            loss_prob = criterion_prob(pred_prob, target_prob)
            loss_eta = criterion_eta(pred_eta, target_eta)
            
            loss = loss_prob + loss_eta * 0.25
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Print progress every 200 batches
            if batch_i > 0 and batch_i % 200 == 0:
                pass # Suppress to keep output clean
                
        epoch_end = time.time()
        print(f"-> Epoch {epoch+1}/{epochs} completada em: {epoch_end - epoch_start:.2f} segundos")

    total_time = time.time() - total_start_time
    
    print(f"\n[CONCLUSÃO DO BENCHMARK]")
    print(f"Tempo Total de Treino: {total_time:.2f} segundos")
    print(f"Desempenho: {(num_sequences * epochs) / total_time:,.0f} amostras processadas por segundo.")
    print("\n[ANALISE DE GARGALOS]")
    if not has_gpu:
        print("1. GPU: O uso da GPU (CUDA) aceleraria este processo em cerca de 10x a 30x durante o Backpropagation.")
    print("2. Materialização do histórico: ler milhões de records de um tracker SQLite ainda pode ser caro se o pipeline não fizer streaming/chunking.")
    print("   Solução ideal para Big Data: snapshots intermediários do dataset (.pt/.npy/parquet) ou leitura incremental do SQLite apenas para treino.")

if __name__ == "__main__":
    benchmark_training()
