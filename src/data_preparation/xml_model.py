import torch

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

def get_embedding_layer(embedding_weights):
    word_embeddings = nn.Embedding(num_embeddings=embedding_weights.size(0), embedding_dim=embedding_weights.size(1))
    word_embeddings.weight.data.copy_(embedding_weights)
    word_embeddings.weight.requires_grad = False  # not train
    return word_embeddings

class Hybrid_XML(BasicModule):
    def __init__(self, BERTmodel, num_labels=3714, vocab_size=30001, embedding_size=300, embedding_weights=None,
                 max_seq=300, hidden_size=256, d_a=256, label_emb=None):
        super(Hybrid_XML, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.max_seq = max_seq
        self.hidden_size = hidden_size

        self.bert = BERTmodel
        for name, param in self.bert.named_parameters():
            param.required_grad = False
            if 'layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # interaction-attention layer
        self.key_layer = torch.nn.Linear(self.embedding_size, self.hidden_size)
        self.query_layer = torch.nn.Linear(7, self.hidden_size)

        # self-attn layer
        self.linear_first = torch.nn.Linear(self.embedding_size, d_a)
        self.linear_second = torch.nn.Linear(d_a, self.num_labels)

        # weight adaptive layer
        self.linear_weight1 = torch.nn.Linear(self.embedding_size, 1)
        self.linear_weight2 = torch.nn.Linear(self.embedding_size, 1)

        # shared for all attention component
        self.linear_final = torch.nn.Linear(768, self.hidden_size)
        self.decrease_emb_size = torch.nn.Linear(self.embedding_size, 768)
        self.output_layer = torch.nn.Linear(self.hidden_size, 1)

        label_embedding = torch.FloatTensor(self.num_labels, 7)

        #         label_emb = torch.nn.functional.pad(label_emb, pad=(0, 384-52), mode='constant', value=0)
        if label_emb is None:
            nn.init.xavier_normal_(label_embedding)
        else:
            label_embedding.copy_(label_emb)
        self.label_embedding = nn.Parameter(label_embedding, requires_grad=False)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (
                torch.zeros(2, batch_size, self.hidden_size).cuda(),
                torch.zeros(2, batch_size, self.hidden_size).cuda())
        else:
            return (torch.zeros(2, batch_size, self.hidden_size), torch.zeros(2, batch_size, self.hidden_size))

    def forward(self, input_ids, attention_mask, embedding_generation=False):

        #         emb = self.word_embeddings(x)

        #         hidden_state = self.init_hidden(emb.size(0))
        #         output, hidden_state = self.lstm(emb, hidden_state)  # [batch,seq,2*hidden]

        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)[0]

        # get attn_key
        attn_key = self.key_layer(output)  # [batch,seq,hidden]
        attn_key = attn_key.transpose(1, 2)  # [batch,hidden,seq]

        # get attn_query
        label_emb = self.label_embedding.expand(
            (attn_key.size(0), self.label_embedding.size(0), self.label_embedding.size(1)))  # [batch,L,label_emb]
        label_emb = self.query_layer(label_emb)  # [batch,L,label_emb]

        # attention
        similarity = torch.bmm(label_emb, attn_key)  # [batch,L,seq]
        similarity = F.softmax(similarity, dim=2)
        out1 = torch.bmm(similarity, output)  # [batch,L,label_emb]

        # self-attn output
        self_attn = torch.tanh(self.linear_first(output))  # [batch,seq,d_a]
        self_attn = self.linear_second(self_attn)  # [batch,seq,L]
        self_attn = F.softmax(self_attn, dim=1)
        self_attn = self_attn.transpose(1, 2)  # [batch,L,seq]
        out2 = torch.bmm(self_attn, output)  # [batch,L,hidden]

        factor1 = torch.sigmoid(self.linear_weight1(out1))
        factor2 = torch.sigmoid(self.linear_weight2(out2))
        factor1 = factor1 / (factor1 + factor2)
        factor2 = 1 - factor1

        out = factor1 * out1 + factor2 * out2

        out = self.decrease_emb_size(out)

        if embedding_generation:
            return out

        out = F.relu(self.linear_final(out))
        out = torch.sigmoid(self.output_layer(out).squeeze(-1))  # [batch,L]

        return out

