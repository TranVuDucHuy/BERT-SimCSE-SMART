from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import NLI_Dataset
from loss import supCL_loss
from utils import create_triples
from utils import create_large_data_pairs
from utils import group_data_by_level
from utils import read_file

from utils import load_config

# Train with SimCSE
def train_cl(model, config, criterion, cl_loss, trainset, epochs):
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True, collate_fn=trainset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    path = config['best_cl_model']

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training SimCSE ...:"):
            optimizer.zero_grad()
            input_ids_anchor = batch[0]['token_ids_anchor'].to(config['device'])
            attention_mask_anchor = batch[0]['attention_mask_anchor'].to(config['device'])
            input_ids_pos = batch[0]['token_ids_pos'].to(config['device'])
            attention_mask_pos = batch[0]['attention_mask_pos'].to(config['device'])
            input_ids_neg = batch[0]['token_ids_neg'].to(config['device'])
            attention_mask_neg = batch[0]['attention_mask_neg'].to(config['device'])

            anchor_output = model(input_ids_anchor, attention_mask_anchor)['pooler_output']
            pos_output = model(input_ids_pos, attention_mask_pos)['pooler_output']
            neg_output = model(input_ids_neg, attention_mask_neg)['pooler_output']

            loss = cl_loss(criterion, anchor_output, pos_output, neg_output)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        
    torch.save(model.state_dict(), path)   
        
    return model

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(config['device'])
    
    train_data = read_file(config['train_path'])
    data_by_level = group_data_by_level(train_data)

    target_size = 300000
    pairs = create_large_data_pairs(data_by_level, target_size)
    
    nli_data = create_triples(pairs)
    NLI_dataset = NLI_Dataset(nli_data)

    cl_model = train_cl(model, config , nn.CrossEntropyLoss(), supCL_loss, NLI_dataset, epochs=3)
