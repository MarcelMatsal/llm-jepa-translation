"""
Minimal training example - quick test with small dataset.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import MultilingualJEPA
from src.data import TranslationDataset, get_dataloader
from src.training import Trainer


def main():
    print("=" * 60)
    print("Minimal Training Example")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create synthetic dataset (or load real one)
    print("\nCreating synthetic dataset...")
    texts_en = [
        "Hello world",
        "How are you?",
        "The weather is nice",
        "I love programming",
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Natural language processing",
        "Computer vision applications",
    ] * 10  # 80 samples
    
    texts_fr = [
        "Bonjour le monde",
        "Comment allez-vous?",
        "Le temps est agréable",
        "J'adore la programmation",
        "L'apprentissage automatique est fascinant",
        "L'apprentissage profond utilise des réseaux neuronaux",
        "Traitement du langage naturel",
        "Applications de vision par ordinateur",
    ] * 10
    
    lang_en = [0] * len(texts_en)
    lang_fr = [1] * len(texts_fr)
    
    dataset = TranslationDataset(texts_en, texts_fr, lang_en, lang_fr)
    train_loader = get_dataloader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = MultilingualJEPA(
        encoder_name='bert-base-multilingual-cased',
        pooling='cls',
        num_languages=2,
        tau=0.999
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        max_grad_norm=1.0,
        log_interval=5
    )
    
    # Train for a few epochs
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train(num_epochs=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_texts = ["Hello world", "Machine learning"]
        embeddings = model.get_embeddings(test_texts, encoder='x')
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Sample embedding norm: {embeddings[0].norm().item():.4f}")


if __name__ == '__main__':
    main()

