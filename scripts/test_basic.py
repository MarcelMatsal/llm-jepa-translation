"""
Simple test script to verify Multilingual JEPA implementation works.
Tests with a small synthetic dataset.
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import MultilingualJEPA
from src.data import TranslationDataset, get_dataloader
from src.training import Trainer


def create_synthetic_dataset(num_samples=100):
    """Create a small synthetic dataset for testing."""
    # Simple English-French pairs
    texts_en = [
        "Hello world",
        "How are you?",
        "The weather is nice",
        "I love programming",
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Natural language processing",
        "Computer vision applications",
        "Reinforcement learning algorithms",
        "Artificial intelligence systems"
    ] * (num_samples // 10)
    
    texts_fr = [
        "Bonjour le monde",
        "Comment allez-vous?",
        "Le temps est agréable",
        "J'adore la programmation",
        "L'apprentissage automatique est fascinant",
        "L'apprentissage profond utilise des réseaux neuronaux",
        "Traitement du langage naturel",
        "Applications de vision par ordinateur",
        "Algorithmes d'apprentissage par renforcement",
        "Systèmes d'intelligence artificielle"
    ] * (num_samples // 10)
    
    # Language IDs: en=0, fr=1
    lang_en = [0] * len(texts_en)
    lang_fr = [1] * len(texts_fr)
    
    return TranslationDataset(texts_en, texts_fr, lang_en, lang_fr)


def test_model_creation():
    """Test that model can be created."""
    print("Testing model creation...")
    model = MultilingualJEPA(
        encoder_name='bert-base-multilingual-cased',
        pooling='cls',
        num_languages=2,
        tau=0.999
    )
    print(f"✅ Model created successfully")
    print(f"   - X-encoder: {type(model.x_encoder).__name__}")
    print(f"   - Y-encoder: {type(model.y_encoder).__name__}")
    print(f"   - Predictor: {type(model.predictor).__name__}")
    print(f"   - Embedding dim: {model.d_model}")
    return model


def test_forward_pass(model, device='cpu'):
    """Test forward pass."""
    print("\nTesting forward pass...")
    model = model.to(device)
    
    texts_src = ["Hello world", "How are you?"]
    texts_tgt = ["Bonjour le monde", "Comment allez-vous?"]
    lang_src = torch.tensor([0, 0])  # English
    lang_tgt = torch.tensor([1, 1])  # French
    
    model.eval()
    with torch.no_grad():
        s_x, s_y, s_y_pred = model.forward(
            texts_src, texts_tgt, lang_src, lang_tgt, normalize=True
        )
    
    print(f"✅ Forward pass successful")
    print(f"   - s_x shape: {s_x.shape}")
    print(f"   - s_y shape: {s_y.shape}")
    print(f"   - s_y_pred shape: {s_y_pred.shape}")
    print(f"   - Embeddings normalized: {torch.allclose(s_x.norm(dim=-1), torch.ones_like(s_x.norm(dim=-1)), atol=1e-5)}")
    
    return s_x, s_y, s_y_pred


def test_loss_computation(model, device='cpu'):
    """Test loss computation."""
    print("\nTesting loss computation...")
    model = model.to(device)
    
    texts_src = ["Hello world", "How are you?"]
    texts_tgt = ["Bonjour le monde", "Comment allez-vous?"]
    lang_src = torch.tensor([0, 0])
    lang_tgt = torch.tensor([1, 1])
    
    model.train()
    loss, metrics = model.compute_loss(
        texts_src, texts_tgt, lang_src, lang_tgt, loss_type='mse'
    )
    
    print(f"✅ Loss computation successful")
    print(f"   - Total loss: {loss.item():.4f}")
    print(f"   - Forward loss: {metrics['loss_forward']:.4f}")
    print(f"   - Backward loss: {metrics['loss_backward']:.4f}")
    print(f"   - Cosine similarity (forward): {metrics['cosine_sim_forward']:.4f}")
    print(f"   - Cosine similarity (backward): {metrics['cosine_sim_backward']:.4f}")
    
    return loss, metrics


def test_training_step(model, device='cpu'):
    """Test a single training step."""
    print("\nTesting training step...")
    model = model.to(device)
    
    # Create small dataset
    dataset = create_synthetic_dataset(num_samples=20)
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        device=device,
        max_grad_norm=1.0,
        log_interval=10
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Manual training step
    texts_src = batch['texts_src']
    texts_tgt = batch['texts_tgt']
    lang_src = torch.tensor(batch['lang_src']).to(device)
    lang_tgt = torch.tensor(batch['lang_tgt']).to(device)
    
    model.train()
    loss, metrics = model.compute_loss(
        texts_src, texts_tgt, lang_src, lang_tgt, loss_type='mse'
    )
    
    # Backward
    trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.max_grad_norm)
    trainer.optimizer.step()
    
    # EMA update
    model.update_ema()
    
    print(f"✅ Training step successful")
    print(f"   - Loss before step: {loss.item():.4f}")
    print(f"   - Gradients computed: {any(p.grad is not None for p in model.parameters() if p.requires_grad)}")
    
    return loss.item()


def test_ema_update(model, device='cpu'):
    """Test EMA update."""
    print("\nTesting EMA update...")
    model = model.to(device)
    
    # Get initial weights
    x_param = next(model.x_encoder.parameters())
    y_param = next(model.y_encoder.parameters())
    initial_y = y_param.data.clone()
    
    # Modify x_encoder weights slightly
    with torch.no_grad():
        x_param.data += 0.1
    
    # Update EMA
    model.update_ema()
    
    # Check that y_encoder moved slightly towards x_encoder
    final_y = y_param.data.clone()
    diff = (final_y - initial_y).abs().mean()
    
    print(f"✅ EMA update successful")
    print(f"   - Y-encoder weights changed: {diff.item():.6f} (should be small)")
    print(f"   - Expected change magnitude: {(1 - model.tau) * 0.1:.6f}")
    
    return diff.item()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multilingual JEPA Basic Tests")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model, device=device)
        
        # Test 3: Loss computation
        test_loss_computation(model, device=device)
        
        # Test 4: EMA update
        test_ema_update(model, device=device)
        
        # Test 5: Training step
        test_training_step(model, device=device)
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try training on a real dataset:")
        print("   python scripts/train.py --dataset opus_books --lang_pair en-fr --epochs 1 --batch_size 8")
        print("\n2. Or use a smaller HuggingFace dataset for testing")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

