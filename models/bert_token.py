import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BertNet(nn.Module):
    def __init__(self, tag_size, top_rnns=False, device='cpu', finetuning=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, tag_size)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x,training = True ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        output_all_encoded_layers=True   则输出堆叠的每个transformer 的隐状态（bert-base有12层，bert-large有24层）
        output_all_encoded_layers= False 则输出最高层一个transformer 的隐状态
        enc: (N, T, 768)
        '''
        x = x.to(self.device)
        # y = y.to(self.device)
        self.training = training
        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x,output_all_encoded_layers=False)
            enc = encoded_layers
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x,output_all_encoded_layers=False)
                enc = encoded_layers

        if self.top_rnns:
            enc, _ = self.rnn(enc)

        logits = self.fc(enc)
        # print(logits.shape)
        y_hat = logits.argmax(-1)
        # print(y_hat.shape)
        return logits, y_hat

