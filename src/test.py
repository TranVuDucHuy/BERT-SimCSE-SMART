from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SST5_Dataset
from model import BertClassifier

from utils import load_config

def test_model(model_path, testset, device, batch_size=32):

    model = BertClassifier(num_labels=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    
    test_loader = DataLoader(testset, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    total_loss = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            
            # Store predictions and labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct_predictions / len(testset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted'
    )
    avg_loss = total_loss / len(test_loader)
    
    # Print metrics
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=np.unique(all_labels),
        yticklabels=np.unique(all_labels)
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    model_path = f"{config['save_dir']}best_model_cls.pth"
    
    metrics = test_model(
    model_path=model_path,
    testset=SST5_Dataset(config['test_path']),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )