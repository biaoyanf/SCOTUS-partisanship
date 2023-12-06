import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import numpy as np
import os
import pickle

class CaseDataset(Dataset): 
    
    # def __init__(self, model_type, label_path, evidence_path, maxlen): 
    def __init__(self, config, data_set): 
        
        # self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        
        self.maxlen = config["max_token"]
        self.config = config 
  
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_type"])  


        if self.config["add_year_tag"]:
            all_year_tag = ['[1955]', '[1956]', '[1957]', '[1958]', '[1959]', '[1960]', '[1961]', '[1962]', '[1963]', '[1964]', '[1965]', '[1966]', '[1967]', '[1968]', '[1969]', '[1970]', '[1971]', '[1972]', '[1973]', '[1974]', '[1975]', '[1976]', '[1977]', '[1978]', '[1979]', '[1980]', '[1981]', '[1982]', '[1983]', '[1984]', '[1985]', '[1986]', '[1987]', '[1988]', '[1989]', '[1990]', '[1991]', '[1992]', '[1993]', '[1994]', '[1995]', '[1996]', '[1997]', '[1998]', '[1999]', '[2000]', '[2001]', '[2002]', '[2003]', '[2004]', '[2005]', '[2006]', '[2007]', '[2008]', '[2009]', '[2010]', '[2011]', '[2012]', '[2013]', '[2014]', '[2015]', '[2016]', '[2017]', '[2018]', '[2019]'] 

            self.tokenizer.add_special_tokens({"additional_special_tokens": all_year_tag})

            self.map_year_tag_to_token_id = {}  
            for ele in all_year_tag: 
                self.map_year_tag_to_token_id[ele] = self.tokenizer.encode(ele, 
                                                        add_special_tokens=False, 
                                                        ) 
                assert len(self.map_year_tag_to_token_id[ele]) == 1 
                self.map_year_tag_to_token_id[ele] = self.map_year_tag_to_token_id[ele][0]
            print("-------------------------------------------------------------")
            print(self.map_year_tag_to_token_id)


        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id 
        print("cls_id, sep_id, pad_id: ", cls_id, sep_id, pad_id)



        self.ids = []

        #get the favor side 
        # self.win_side = []  # it will be like yes or no? or two label is better  
        self.labels = []
        self.text = [] 

        map_justice_to_party = {'j__sonia_sotomayor': 'D', 'j__elena_kagan': 'D', 'j__john_g_roberts_jr': 'R', 'j__neil_gorsuch': 'R', 'j__clarence_thomas': 'R', 'j__antonin_scalia': 'R', 'j__brett_m_kavanaugh': 'R', 'j__earl_warren': 'R', 'j__charles_e_whittaker': 'R', 'j__lewis_f_powell_jr': 'R', 'j__harold_burton': 'D', 'j__sherman_minton': 'D', 'j__abe_fortas': 'D', 'j__hugo_l_black': 'D', 'j__potter_stewart': 'R', 'j__warren_e_burger': 'R', 'j__harry_a_blackmun': 'R', 'j__arthur_j_goldberg': 'D', 'j__samuel_a_alito_jr': 'R', 'j__john_m_harlan2': 'R', 'j__anthony_m_kennedy': 'R', 'j__ruth_bader_ginsburg': 'D', 'j__william_j_brennan_jr': 'R', 'j__john_m_harlan': 'R', 'j__david_h_souter': 'R', 'j__william_o_douglas': 'D', 'j__stephen_g_breyer': 'D', 'j__john_paul_stevens': 'R', 'j__thurgood_marshall': 'D', 'j__felix_frankfurter': 'D', 'j__william_h_rehnquist': 'R', 'j__byron_r_white': 'D', 'j__tom_c_clark': 'D', 'j__sandra_day_oconnor': 'R', 'j__stanley_reed': 'D'} 


        count = 0 
        
        label_count = {'R': 0, 'D': 0 }
        party_mapping = {"R": 0, "D": 1}


        for jst in data_set: 
            for utts in data_set[jst]:  
                # print(utts["case_id"]) 

                current_text = " ".join(utts["utt_text"]) 
                current_token_ids = self.tokenizer.encode(current_text, 
                                                        add_special_tokens=False, # Add special tokens for BERT
                                                        )
                tmp_label = party_mapping[map_justice_to_party[jst]] 
                if config["sliding_window"]:  
                    # print( int((self.maxlen-2)/2))

                    for i in range(0, len(current_token_ids), int((self.maxlen-2)/2) - int(self.config["add_year_tag"])): 
                        tmp_text = current_token_ids[i: i+ self.maxlen-2 - int(self.config["add_year_tag"])]  
                        # print(len(tmp_text))
                        if len(tmp_text) < config["token_threshold"]: 
                            continue
                        
                        # print(i, i+ self.maxlen-2)
                        # print(tmp_text)
                        # adding the cls and sep to the tmp token 
                        if self.config["add_year_tag"]:  
                            # print(f'[{utts["case_id"].split("_")[0]}]', [self.map_year_tag_to_token_id[f'[{utts["case_id"].split("_")[0]}]']])
                            tmp_text = [cls_id]+ [self.map_year_tag_to_token_id[f'[{utts["case_id"].split("_")[0]}]']] + tmp_text+[sep_id]  
                            # print(len(tmp_text))
                        else: 
                            tmp_text = [cls_id]+tmp_text+[sep_id] 

                        # print(len(tmp_text))
                        len_tmp_text = len(tmp_text)
                        if len_tmp_text< self.maxlen: 
                            tmp_text = tmp_text+ [pad_id]*(self.maxlen-len_tmp_text) 

                        # print(len(tmp_text))
                        self.labels.append(tmp_label) 
                        self.text.append(tmp_text) 
                        self.ids.append(f"{utts['case_id']}-{utts['utt_id']}-{i}-{jst}") 
                        count+=1 

                        label_count[map_justice_to_party[jst]]+=1 

            # else: use the first 510 tokens 
                else:  
                    tmp_text = current_token_ids[: self.maxlen-2 - int(self.config["add_year_tag"])] 
                    
                    if len(tmp_text) < config["token_threshold"]: 
                            continue
                    
                    if self.config["add_year_tag"]:  
                        # print(f'[{utts["case_id"].split("_")[0]}]', [self.map_year_tag_to_token_id[f'[{utts["case_id"].split("_")[0]}]']])
                        tmp_text = [cls_id]+ [self.map_year_tag_to_token_id[f'[{utts["case_id"].split("_")[0]}]']] + tmp_text+[sep_id]  
                        # print(tmp_text[:2])
                    else: 
                        tmp_text = [cls_id]+tmp_text+[sep_id] 


                    len_tmp_text = len(tmp_text)
                    if len_tmp_text< self.maxlen: 
                        tmp_text = tmp_text+ [pad_id]*(self.maxlen-len_tmp_text) 

                    self.labels.append(tmp_label) 
                    self.text.append(tmp_text) 
                    self.ids.append(f"{utts['case_id']}-{utts['utt_id']}-0-{jst}") 
                    count+=1 
        
                    label_count[map_justice_to_party[jst]]+=1  

                    # print("label_count",  label_count) 
                    # print()

        print(label_count, sum(label_count.values()))

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index): 
        ids = self.ids[index]
        # ids = torch.tensor(ids)
 
        label = self.labels[index]
        label = torch.tensor(label)
        
        
        current_token_ids = self.text[index] 
        
        # print("current_token_id: ", current_token_ids)
       
        # tokens_ids = self.tokenizer.convert_tokens_to_ids(current_token) #Obtaining the indices of the tokens in the BERT Vocabulary

        
        tokens_ids_tensor = torch.tensor(current_token_ids) #Converting the list to a pytorch tensor
        # print("tokens_ids_tensor", tokens_ids_tensor.size())
        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        # print(label)
        # print("label:", label)
        # print("publication_types_index:", publication_types_index)
        # print(tokens_ids_tensor.shape) 
        # print(attn_mask.shape)
        # print(label)
        # # print(ids) 
        # print()
        return tokens_ids_tensor, attn_mask, label, ids
    
    def get_tokenizer(self): 
        return self.tokenizer 
