"""
HYPERPARAMETER ANALYSIS
Tests influence of individual hyperparameters on model performance

Based on:
- PyTorch TorchVision Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- DebuggerCafe Tutorial: https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import json
import os
from tqdm import tqdm
from datetime import datetime
import random
import csv #EXCEL


# BASEMODELL CONFIG (STANDARTMODELL)

BASELINE = {
    'BATCH_SIZE': 4,
    'MAX_EPOCHS': 10,  # Reduziert für schnellere Analyse
    'LEARNING_RATE': 0.003,
    'WEIGHT_DECAY': 0.0003,
    'DROPOUT': 0.15,
    'AUGMENTATION': True
}


# EXPERIMENTS: JEDES ÄNDERT NUR EINEN PARAMETER

EXPERIMENTS = [
    # Baseline
    {'name': 'baseline', **BASELINE},
    
    # Learning Rate Varianten
    {'name': 'lr_0.001', **BASELINE, 'LEARNING_RATE': 0.001},
    {'name': 'lr_0.005', **BASELINE, 'LEARNING_RATE': 0.005},
    {'name': 'lr_0.01', **BASELINE, 'LEARNING_RATE': 0.01},
    
    # Dropout Varianten
    {'name': 'dropout_0.0', **BASELINE, 'DROPOUT': 0.0},
    {'name': 'dropout_0.25', **BASELINE, 'DROPOUT': 0.25},
    {'name': 'dropout_0.4', **BASELINE, 'DROPOUT': 0.4},
    
    # Weight Decay Varianten
    {'name': 'wd_0.0', **BASELINE, 'WEIGHT_DECAY': 0.0},
    {'name': 'wd_0.001', **BASELINE, 'WEIGHT_DECAY': 0.001},
    {'name': 'wd_0.01', **BASELINE, 'WEIGHT_DECAY': 0.01},
    
    # Batch Size Varianten
    {'name': 'batch_2', **BASELINE, 'BATCH_SIZE': 2},
    
    # Augmentation
    {'name': 'no_augmentation', **BASELINE, 'AUGMENTATION': False},
]
#PFADE
JSON_PATH = 'stanford_cars_yolo/stanford_train_24brands_yolo.json'
IMAGE_ROOT = 'stanford_cars_train'


class CarDataset(Dataset):
    def __init__(self, json_path, image_root, augment=False):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.image_root = image_root
        self.augment = augment
        
        self.img_to_ann = {}
        for ann in data['annotations']:
            self.img_to_ann[ann['image_id']] = ann
        
        self.images = [img for img in data['images'] if img['id'] in self.img_to_ann]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        
        img = Image.open(img_path).convert('RGB')
        
        if self.augment and random.random() > 0.5:
            img = torchvision.transforms.functional.hflip(img)
        
        if self.augment and random.random() > 0.7:
            img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        ann = self.img_to_ann[img_info['id']]
        
        x, y, w, h = ann['bbox']
        boxes = torch.tensor([[x, y, x+w, y+h]], dtype=torch.float32)
        labels = torch.tensor([ann['category_id']], dtype=torch.int64)
        
        return img_tensor, {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_info['id']])}


def collate_fn(batch):
    return tuple(zip(*batch))


def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                if len(pred['boxes']) > 0:
                    pred_label = int(pred['labels'][pred['scores'].argmax()])
                    if pred_label == int(target['labels'][0]):
                        correct += 1
                total += 1
    
    return (correct / total * 100) if total > 0 else 0


def create_model(dropout, device):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 25
    
    if dropout > 0:
        model.roi_heads.box_predictor.cls_score = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(in_features, num_classes))
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(in_features, num_classes * 4))
    else:
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)
    
    return model.to(device)


def run_experiment(config, train_loader, val_loader, device, output_dir):
    """Run single experiment with given config"""
    
    name = config['name']
    print(f"\n{'='*60}")
    print(f" EXPERIMENT: {name}")
    print(f"{'='*60}")
    print(f"   LR: {config['LEARNING_RATE']}, Dropout: {config['DROPOUT']}, "
          f"WD: {config['WEIGHT_DECAY']}, Batch: {config['BATCH_SIZE']}, Aug: {config['AUGMENTATION']}")
    
    # MODELL ERSTELLEN
    model = create_model(config['DROPOUT'], device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['LEARNING_RATE'], momentum=0.9, weight_decay=config['WEIGHT_DECAY'])
    
    # EXCEL FÜR DIESES EXPERIMENT
    exp_csv = f'{output_dir}/{name}_log.csv'
    with open(exp_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'gap', 'train_accuracy', 'val_accuracy'])
    
    # Training loop
    for epoch in range(config['MAX_EPOCHS']):
        # TRAIN
        model.train()
        train_loss = 0
        
        for images, targets in tqdm(train_loader, desc=f'{name} Epoch {epoch+1}', leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            train_loss += losses.item()
        
        avg_train = train_loss / len(train_loader)
        
        # VAL
        model.train()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        
        avg_val = val_loss / len(val_loader)
        gap = avg_val - avg_train
        
        # Accuracy
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)
        
        # Log
        with open(exp_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f'{avg_train:.4f}', f'{avg_val:.4f}', f'{gap:+.4f}', f'{train_acc:.2f}', f'{val_acc:.2f}'])
        
        print(f"   Epoch {epoch+1}: Loss {avg_val:.4f}, Acc {val_acc:.2f}%")
    
    # RÜCKGABE FINAL RESULTS
    return {
        'name': name,
        'final_train_loss': avg_train,
        'final_val_loss': avg_val,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'final_gap': gap
    }


def main():
    print("="*80)
    print(" HYPERPARAMETER ANALYSIS")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'hyperparameter_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data once
    print("\n Loading dataset...")
    
    indices = list(range(len(CarDataset(JSON_PATH, IMAGE_ROOT))))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Summary CSV
    summary_csv = f'{output_dir}/summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'parameter_changed', 'value', 'final_train_loss', 
                        'final_val_loss', 'final_gap', 'final_train_acc', 'final_val_acc'])
    
    all_results = []
    
    # LAUFE ALLE EXPERIMENTE
    for i, config in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        
        # Create loaders with current batch size
        augment = config['AUGMENTATION']
        train_dataset = Subset(CarDataset(JSON_PATH, IMAGE_ROOT, augment=augment), train_indices)
        val_dataset = Subset(CarDataset(JSON_PATH, IMAGE_ROOT, augment=False), val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        # Run
        result = run_experiment(config, train_loader, val_loader, device, output_dir)
        all_results.append(result)
        
        # Determine which parameter was changed
        param_changed = 'baseline'
        value = '-'
        if 'lr_' in config['name']:
            param_changed, value = 'learning_rate', config['LEARNING_RATE']
        elif 'dropout_' in config['name']:
            param_changed, value = 'dropout', config['DROPOUT']
        elif 'wd_' in config['name']:
            param_changed, value = 'weight_decay', config['WEIGHT_DECAY']
        elif 'batch_' in config['name']:
            param_changed, value = 'batch_size', config['BATCH_SIZE']
        elif 'no_aug' in config['name']:
            param_changed, value = 'augmentation', False
        
        # SAVE TO ZUSAMMENFASSUNG
        with open(summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([config['name'], param_changed, value, f"{result['final_train_loss']:.4f}",
                           f"{result['final_val_loss']:.4f}", f"{result['final_gap']:.4f}",
                           f"{result['final_train_acc']:.2f}", f"{result['final_val_acc']:.2f}"])
    
    # FINALE ZUSAMMENFASSUNG
    print("\n" + "="*80)
    print(" FINAL RESULTS")
    print("="*80)
    
    for r in sorted(all_results, key=lambda x: x['final_val_acc'], reverse=True):
        print(f"   {r['name']:20s}: Val Acc {r['final_val_acc']:.2f}%, Val Loss {r['final_val_loss']:.4f}")
    
    print(f"\n Results saved to: {output_dir}/")
    print(f"   - summary.csv (Übersicht aller Experimente)")
    print(f"   - [experiment]_log.csv (Details pro Experiment)")


if __name__ == '__main__':
    main()