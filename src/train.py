from tqdm import tqdm
import torch
from sklearn.metrics import  accuracy_score
from torch.utils.data import DataLoader

from utils import load_config
from adversarial import AdversarialReg
from mbpp import MBPP
from dataset import SST5_Dataset
from dataset import Amazon_Dataset
from model import BertClassifier

from transformers import BertModel

from torch.cuda.amp import GradScaler, autocast

def train_epoch(model, dataloader, criterion, optimizer, device, pdg_config):
    model.train()
    train_loss = 0.0
    
    epsilon = pdg_config['epsilon']
    eta = pdg_config['eta']
    lambda_ = pdg_config['lambda_']    
    sigma = pdg_config['sigma']
    K = pdg_config['K']
    
    scaler = GradScaler()

    pgd = AdversarialReg(model, epsilon, lambda_, eta, sigma, K)
    mbpp = MBPP(model)

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward(retain_graph=True)

        # Backpropagation cho adversarial loss
        with autocast():
            adv_loss = pgd.max_loss_reg((input_ids, attention_mask), outputs)
        scaler.scale(adv_loss).backward(retain_graph=True)

        # Backpropagation cho Bregman divergence
        with autocast():
            breg_div = mbpp.bregman_divergence((input_ids, attention_mask), outputs)
        scaler.scale(breg_div).backward()

        scaler.step(optimizer)  # Thay thế optimizer.step()
        scaler.update() 
        # optimizer.step()
        mbpp.apply_momentum(model.named_parameters())
        total_loss += loss.item()

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

def train(model, trainset, valset, config, pdg_config):
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_model_path = config['best_cls_model']

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'], pdg_config)
        val_loss, accuracy= eval_epoch(model, val_loader, criterion, config['device'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Save model at epoch {epoch + 1}")

    return model

def load_simcse_model(model_path, num_labels):
    model = BertClassifier(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == "__main__":
    import argparse

    # Load configurations
    config = load_config("config.yaml")
    pdg_config = load_config("pgd_cf.yaml")

    # Argument parser for dataset and model selection
    parser = argparse.ArgumentParser(description="Choose dataset and model type")
    parser.add_argument('--dataset', type=str, choices=['sst5', 'amazon'], required=True, 
                        help="Dataset to use: 'sst5' or 'amazon'")
    parser.add_argument('--model_type', type=str, choices=['supcl', 'unsupcl'], required=True, 
                        help="Model type to use: 'supcl' (supervised) or 'unsupcl' (unsupervised)")
    args = parser.parse_args()

    # Load BERT model and weights
    bert = BertModel.from_pretrained('bert-base-uncased')
    if args.model_type == 'supcl':
        bert.load_state_dict(torch.load(config['best_supcl_model']))
    elif args.model_type == 'unsupcl':
        bert.load_state_dict(torch.load(config['best_unsupcl_model']))
    else:
        raise ValueError("Invalid model type choice. Please choose 'supcl' or 'unsupcl'.")

    # Wrap BERT in classifier and move to device
    model = BertClassifier(bert, num_labels=5).to(config['device'])

    # Load datasets based on selection
    if args.dataset == 'sst5':
        trainset = SST5_Dataset(config['sst_train_path'])
        valset = SST5_Dataset(config['sst_val_path'])
    elif args.dataset == 'amazon':
        trainset = Amazon_Dataset(config['amazon_train_path'])
        valset = Amazon_Dataset(config['amazon_val_path'])
    else:
        raise ValueError("Invalid dataset choice. Please choose 'sst5' or 'amazon'.")

    # Train the model
    model = train(model, trainset, valset, config, pdg_config)
    print("Training finished!")
