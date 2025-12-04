"""
STANDART TRAINING - MIT EXCEL TABELLEN SPEICHERUNG

Basiert auf:
- PyTorch TorchVision Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- DebuggerCafe Tutorial: https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
- GitHub Repo: https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline
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
import time
import platform
import subprocess
import random
import csv #Excel Datei

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
        print(f" Loaded {len(self.images)} images (augment={augment})")
    
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
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_info['id']])}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))

#Shutdown nach Training + Option shutdown innerhalb von 60sek zu stoppen
def shutdown_system():
    print("\n Preparing shutdown in 60 seconds...")
    print("   Press Ctrl+C to cancel!")
    time.sleep(5)
    try:
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "60"])
        print(" Shutdown scheduled!")
    except:
        print("⚠️  Manual shutdown needed")


def calculate_accuracy(model, data_loader, device):
    """Calculate accuracy on dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                if len(pred['boxes']) > 0:
                    best_idx = pred['scores'].argmax()
                    pred_label = int(pred['labels'][best_idx])
                    true_label = int(target['labels'][0])
                    if pred_label == true_label:
                        correct += 1
                total += 1
    
    return (correct / total * 100) if total > 0 else 0


def train():
    print("="*80)
    print("Standart TRAINING")
    print("="*80)
    
    # HYPERPARAMETER
    BATCH_SIZE = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.0003
    DROPOUT = 0.15

    #DATEIEN PFAD
    JSON_PATH = 'stanford_train_yolo.json'
    IMAGE_ROOT = 'stanford_cars_train'
    #Benutzung von Grafikkarte, falls nicht wird der Prozessor benutzt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Load dataset
    print(f"\n Loading dataset...")
    full_dataset = CarDataset(JSON_PATH, IMAGE_ROOT, augment=True)
    
    # Split PARAMETER -> 80% und 20%
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset_full = CarDataset(JSON_PATH, IMAGE_ROOT, augment=False)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f" Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # MODELL
    print(f"\n  Creating model...")
    num_classes = 25
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT), torch.nn.Linear(in_features, num_classes))
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT), torch.nn.Linear(in_features, num_classes * 4))
    
    model = model.to(device)
    print(" Model ready!")
   
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Setup output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'balanced_model_{timestamp}'
    os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
    
    # Excel - Epoch Log
    epoch_csv = f'{output_dir}/epoch_log.csv'
    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'gap', 'train_accuracy', 'val_accuracy', 'learning_rate', 'best_model'])
    
    # Excel - Batch Log (für Trainingsschritte) -> wurden eigentlich doch nicht gebraucht
    batch_csv = f'{output_dir}/batch_log.csv'
    with open(batch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'epoch', 'batch', 'train_loss'])
    
    print(f" Logging to: {epoch_csv}")
    print(f" Logging to: {batch_csv}")
    print(f"\n Training...\n{'='*80}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP = 7
    global_step = 0
    
    #MAX EPOCHS HYPERPARAMETER
    for epoch in range(MAX_EPOCHS):
        # TRAIN
        model.train()
        train_loss = 0
        batch_num = 0
        
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1} [TRAIN]'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            batch_loss = losses.item()
            train_loss += batch_loss
            global_step += 1
            batch_num += 1
            
            # Log batch loss
            with open(batch_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch+1, batch_num, f'{batch_loss:.4f}'])
        
        avg_train = train_loss / len(train_loader)
        
        # VAL - Loss
        model.train()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f'Epoch {epoch+1} [VAL]  '):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val = val_loss / len(val_loader)
        gap = avg_val - avg_train
        current_lr = optimizer.param_groups[0]['lr']
        
        # CALCULIERT Accuracy
        print(f"Calculating accuracy...")
        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)
        
        # Check if best
        is_best = avg_val < best_val_loss
        
        # Write epoch to CSV
        with open(epoch_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f'{avg_train:.4f}', f'{avg_val:.4f}', f'{gap:+.4f}', 
                           f'{train_acc:.2f}', f'{val_acc:.2f}', f'{current_lr:.6f}', 
                           'yes' if is_best else 'no'])
        
        # ZUSAMMENFASSUNG
        print(f"\n Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val:.4f} | Accuracy: {val_acc:.2f}%")
       
        
        # Save best_model
        if is_best:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'train_loss': avg_train, 'val_loss': avg_val},
                      f'{output_dir}/checkpoints/best_model.pth')
            print(f"Saved (Best: {avg_val:.4f})")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{EARLY_STOP})")
        
        if (epoch+1) % 5 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      f'{output_dir}/checkpoints/epoch_{epoch+1}.pth')
        
        if patience_counter >= EARLY_STOP:
            print(f"\n Early stop!")
            break
        
        old_lr = current_lr
        scheduler.step(avg_val)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f" LR: {old_lr:.6f} → {new_lr:.6f}")
        print()
    
    print("="*80)
    print("DONE!")
    print(f"   Best val: {best_val_loss:.4f}")
    print(f"   Model: {output_dir}/checkpoints/best_model.pth")
    print(f"   Epoch CSV: {epoch_csv}")
    print(f"   Batch CSV: {batch_csv}")
    print("="*80)
    
    shutdown_system()


if __name__ == '__main__':
    print("\n  AUTO SHUTDOWN in 60s after training!")
    input("Press ENTER...")
    train()