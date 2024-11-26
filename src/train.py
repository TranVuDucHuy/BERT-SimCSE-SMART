from tqdm import tqdm
import torch
from sklearn.metrics import  accuracy_score
from torch.utils.data import DataLoader

from utils import load_config
from adversarial import AdversarialReg
from mbpp import MBPP
from dataset import SST5_Dataset
from model import BertClassifier

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    pgd = AdversarialReg(model)
    mbpp = MBPP(model)

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward(retain_graph=True)

        adv_loss = pgd.max_loss_reg((input_ids, attention_mask), logits)
        adv_loss.backward(retain_graph=True)

        breg_div = mbpp.bregman_divergence((input_ids, attention_mask), logits)
        breg_div.backward()

        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(dataloader)

    print(f"Train Loss: {train_loss:.4f}")
    return train_loss

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
          
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            eval_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    eval_loss /= len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

    return eval_loss, accuracy

def train(model, config):
    trainset = SST5_Dataset(config['train_path'])
    valset = SST5_Dataset(config['val_path'])
    
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_model_path = f"{config['save_dir']}best_model_cls.pth"

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss, accuracy= eval_epoch(model, val_loader, criterion, config['device'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Save model at epoch {epoch + 1}")

    return model

if __name__ == "__main__":
    config = load_config("config.yaml")
    model = BertClassifier(num_labels=5).to(config['device'])
    model = train(model, config)
    print("Training finished!")