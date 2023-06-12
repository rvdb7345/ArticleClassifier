import torch
from transformers import BertModel

class BertClassifier(torch.nn.Module):
    """
        Bert Model for classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 60, 7

        #         self.bert = RobertaModel.from_pretrained('roberta-base')
        self.bert = BertModel.from_pretrained('scibert_scivocab_uncased')
        #         self.bert = torch.load(cc_path(f'models/baselines/paula_finetuned_bert_56k_10e_tka.pt')).base_model
        #         self.bert = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out))
        self.sigmoid = torch.nn.Sigmoid()

        # Freeze the Bert Model
        # Freeze all layers except the last two
        for name, param in self.bert.named_parameters():
            if 'layer.9' in name or 'layer.10' in name or 'layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        #         last_hidden_state_cls = outputs[0]

        # Feed input to classifier to compute logits
        logit = self.classifier(last_hidden_state_cls)

        #         logits = self.sigmoid(logit)

        return logit