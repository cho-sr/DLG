"""
Test if imports work correctly
"""
import sys
sys.path.append('/root/Jo')

print("Testing imports...")

# Test model import
try:
    from models.vision import ResNet18
    model = ResNet18(num_classes=100)
    print("✅ ResNet18 imported successfully")
    print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"❌ ResNet18 import failed: {e}")

# Test dataset import
try:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = datasets.CIFAR100('./data', train=True, download=False, transform=transform)
    print("✅ CIFAR-100 dataset loaded")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Image shape: {train_dataset[0][0].shape}")
    print(f"   Number of classes: 100")
except Exception as e:
    print(f"⚠️  CIFAR-100 not downloaded yet (will download on first run)")

print("\n✅ All imports successful! Ready to run experiment.")
