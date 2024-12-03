import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        # self.bert = BertModel.from_pretrained(saved_model_path)
        
        # Frozen bert
        #self.bert.requires_grad_(False)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
class UnSup_BERT(nn.Module):
    def __init__(self, bert, is_unsup_train=True):
        super(UnSup_BERT, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.is_unsup_train = is_unsup_train

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        pooled = outputs['pooler_output']

        if not self.is_unsup_train:
            return pooled

        return self.dropout(pooled)
