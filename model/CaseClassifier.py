import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np



class CaseClassifier(nn.Module):

    def __init__(self, config): 
        super(CaseClassifier, self).__init__()
        #Instantiating BERT model object 
        # print(config["model_type"])
        # print("hello") 
        # print(config)
        self.config = config
        if config["model_type"] != "zlucia/custom-legalbert": 
            self.bert_layer = AutoModel.from_pretrained(config["model_type"], attention_probs_dropout_prob=config["attention_probs_dropout_prob"], hidden_dropout_prob=config["attention_probs_dropout_prob"])
        else:
            self.bert_layer = AutoModel.from_pretrained(config["model_type"], from_tf=True, attention_probs_dropout_prob=config["attention_probs_dropout_prob"], hidden_dropout_prob=config["attention_probs_dropout_prob"])

        if self.config["add_year_tag"]:
            all_year_tag = ['[1955]', '[1956]', '[1957]', '[1958]', '[1959]', '[1960]', '[1961]', '[1962]', '[1963]', '[1964]', '[1965]', '[1966]', '[1967]', '[1968]', '[1969]', '[1970]', '[1971]', '[1972]', '[1973]', '[1974]', '[1975]', '[1976]', '[1977]', '[1978]', '[1979]', '[1980]', '[1981]', '[1982]', '[1983]', '[1984]', '[1985]', '[1986]', '[1987]', '[1988]', '[1989]', '[1990]', '[1991]', '[1992]', '[1993]', '[1994]', '[1995]', '[1996]', '[1997]', '[1998]', '[1999]', '[2000]', '[2001]', '[2002]', '[2003]', '[2004]', '[2005]', '[2006]', '[2007]', '[2008]', '[2009]', '[2010]', '[2011]', '[2012]', '[2013]', '[2014]', '[2015]', '[2016]', '[2017]', '[2018]', '[2019]'] 
            # need to resize as we add special tokens
            self.bert_layer.resize_token_embeddings(len(AutoTokenizer.from_pretrained(config["model_type"])) + len(all_year_tag)) 
            

        # self.bert_layer = BertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        #Classification layer
        #input dimension is 768 because [CLS] embedding has a dimension of 768
        #output dimension is 3 cuz there are three types of labels

        # we only have 1 label as the ourput now  
        self.cls_layer = nn.Linear(config["embedding_size"], 1)  

        self.cls_layer_2 = nn.Linear(config["embedding_size"], config["embedding_size"]) 

        # self.softmax = nn.Softmax(dim=1)  
        self.sigmoid = nn.Sigmoid()

        self.drop_layer = nn.Dropout(p=config["dropout"]) 

        # print("hello-------------------------------------------------")
        # for name, param in self.bert_layer.named_parameters():
        #     param.requires_grad = False


    def forward(self, seq, attn_masks): 
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        outputs = self.bert_layer(seq, attention_mask = attn_masks, output_hidden_states = True)

        #Obtaining the representation of [CLS] head (the first token)
        # cls_rep = outputs.last_hidden_state[:, 0]
        if seq.is_cuda:
            dummy_tensor = torch.tensor([0.0]).cuda(seq.device)
        else: 
            dummy_tensor = torch.tensor([0.0])


        hids = outputs.last_hidden_state  # it is (batch_size, sequence_length, hidden_size)
        s = hids.size()
        # print(s)
        # print(attn_masks.size())
        mask = attn_masks.unsqueeze(-1)
        mask = mask.repeat([1, 1, s[2]])
        # print(mask.size())
        # mask = torch.reshape(mask, [s[0],s[1]*s[2]])
        hids = torch.where(mask==1, hids, dummy_tensor)
        
        # print(s)
        # only use cls embedding 
        hid_rep = hids[:, 0] 
        

        # print("cls_rep: ", len(cls_rep))
        # cls_rep = torch.cat((hids[-1][:, 0], hids[-2][:, 0], hids[-3][:, 0]), -1)

        # cls_rep = torch.cat((cls_rep.float(), publication_types_index.float()), 1)
        # concate cls with publication_type index 

        # add the dropout 
        hid_rep = self.drop_layer(hid_rep)

        #Feeding cls_rep to the classifier layer

        # hid_rep = self.cls_layer_2(hid_rep)

        logits = self.cls_layer(hid_rep) 

        logits = self.sigmoid(logits)
        
        return logits




    def get_extracted_embedding(self, seq, attn_masks): 
        outputs = self.bert_layer(seq, attention_mask = attn_masks, output_hidden_states = True)

        if seq.is_cuda:
            dummy_tensor = torch.tensor([0.0]).cuda(seq.device)
        else: 
            dummy_tensor = torch.tensor([0.0])


        hids = outputs.last_hidden_state  # it is (batch_size, sequence_length, hidden_size)
        s = hids.size()
        # print(s)
        # print(attn_masks.size())
        mask = attn_masks.unsqueeze(-1)
        mask = mask.repeat([1, 1, s[2]])
        # print(mask.size())
        # mask = torch.reshape(mask, [s[0],s[1]*s[2]])
        hids = torch.where(mask==1, hids, dummy_tensor)
        
        # print(s)
        # only use cls embedding 
        hid_rep = hids[:, 0]  

        return hid_rep