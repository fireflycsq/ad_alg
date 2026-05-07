"""Training script for InterFormer."""

import torch
from torch.utils.data import TensorDataset, DataLoader
from dataset import CTRDataset, make_synthetic_batch, create_dataloaders
from model import InterFormer
from trainer import CTRTrainer
from utils import set_seed, get_device, print_model_summary


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Device: {device}\n")

    # ---- Config ----
    DENSE_DIM = 16
    SPARSE_VOCAB_SIZES = [100, 200, 150, 300]   # 4 sparse features
    SEQ_LEN = 50
    EMBED_DIM = 64
    N_LAYERS = 3
    BATCH_SIZE = 32
    EPOCHS = 10

    # ---- Build model ----
    model = InterFormer(
        dense_dim=DENSE_DIM,
        sparse_vocab_sizes=SPARSE_VOCAB_SIZES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        interaction="dcnv2",    # try "fm" or "dhen" too
        n_heads=4,
        k_seeds=1,
        dropout=0.1,
        mlp_hidden_dims=[128, 64],
    )

    # Print model summary
    print_model_summary(model)
    print()

    # ---- Generate synthetic data ----
    N_TRAIN, N_VAL = 2000, 500
    
    def gen_dataset(n):
        d, s, sq, _, y = make_synthetic_batch(n, DENSE_DIM, len(SPARSE_VOCAB_SIZES),
                                               300, SEQ_LEN, "cpu")
        return TensorDataset(d, s, sq, y)

    train_ds = gen_dataset(N_TRAIN)
    val_ds   = gen_dataset(N_VAL)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)

    # ---- Train model ----
    print("=== Training InterFormer ===")
    trainer = CTRTrainer(model, lr=1e-3, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()