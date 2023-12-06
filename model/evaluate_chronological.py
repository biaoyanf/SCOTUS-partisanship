import time 

from torch.utils.data import DataLoader
from CaseDataset import CaseDataset
from CaseClassifier import CaseClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pyhocon
import os
import errno

import json
import random
import numpy as np 
import copy 

from sklearn.metrics import f1_score,accuracy_score, classification_report

def initialize_from_env():
#   if "GPU" in os.environ:
#     set_gpus(int(os.environ["GPU"]))
#   else:
#     set_gpus()

  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path



def get_accuracy_from_logits(logits, labels):
    # probs = torch.sigmoid(logits.unsqueeze(-1)) 

    soft_probs = (logits > 0.5).float()
    # print("logits ", logits)
    # print("lables: ", labels)
    # print("soft_probs: ", soft_probs)
    # pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to get the max index
    # pred_arg_max = torch.squeeze(pred_arg_max)
    # ground_arg_max = torch.argmax(labels, dim=-1)
    # print(pred_arg_max)
    # print(pred_arg_max.tolist(), labels.tolist())
    # print((pred_arg_max == labels).float().tolist())
    # acc = (pred_arg_max == labels).float().mean()
    
    flat_soft_probs = torch.reshape(soft_probs, (-1,))
    # flat_pred_arg_max = torch.reshape(pred_arg_max, (-1,))
    flat_labels = torch.reshape(labels, (-1,))
    f1 = f1_score(flat_labels.cpu(), flat_soft_probs.cpu(), average = "micro") 
    macrof1 = classification_report(flat_labels.cpu(), flat_soft_probs.cpu(),output_dict=True)["macro avg"]["f1-score"]


    acc = accuracy_score(flat_labels.cpu(), flat_soft_probs.cpu()) 
    
    # if f1 == 1.0:
    # print("prediction: ", flat_pred_arg_max)
    # print("gold_label: ", flat_labels)
    # print("overlap: ", set(flat_labels)-set(flat_pred_arg_max), set(flat_pred_arg_max)-set(flat_labels))
    # print("\n")

    return macrof1, acc, f1

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_loss = 0
   
    count = 0
    # pred_results = {}
    if gpu >= 0:
        golds = torch.Tensor([]).cuda(gpu)
        preds = torch.Tensor([]).cuda(gpu) 
    else: 
        golds = torch.Tensor([]).cpu()
        preds = torch.Tensor([]).cpu()
   
    
    with torch.no_grad():
        for (seq, attn_masks, labels, ids) in dataloader:
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()
                
            logits = net(seq.long(), attn_masks) 
   
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item() 

            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
         
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    

    

def evaluate_file(net, criterion, dataloader, file_type, gpu):
    net.eval()

    mean_loss = 0
    count = 0
    pred_results = {}
    if gpu >= 0:
        golds = torch.Tensor([]).cuda(gpu)
        preds = torch.Tensor([]).cuda(gpu) 
    else: 
        golds = torch.Tensor([]).cpu()
        preds = torch.Tensor([]).cpu()
   
    
    with torch.no_grad():
        for (seq, attn_masks, labels, ids) in dataloader:
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()
                
            logits = net(seq.long(), attn_masks) 
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item() 
            # tmp_macrof1, tmp_acc, tmp_f1 = get_accuracy_from_logits(logits, labels)
            # mean_acc += tmp_acc
            # mean_f1 += tmp_f1
            # mean_macrof1 += tmp_macrof1
            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
            store_prediction(pred_results, ids, logits, labels) 
    # print(pred_results)
    with open("./{}/pred_{}.txt".format(config["log_dir"], file_type), "w") as fw: 
        fw.write(json.dumps(pred_results))
    # print("preds: ", preds) 
    # print("golds: ", golds)
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    # return mean_macrof1 / count, mean_acc / count, mean_f1 / count,  mean_loss / count


def store_prediction(pred_results, ids, logits, labels):
    # print("logits", logits)

    # probs = torch.sigmoid(logits.unsqueeze(-1))
    # pred_arg_max = torch.argmax(probs, dim=1) # now gfet a list [1, 2, 0, ...]  -> to 
    # pred_arg_max = torch.squeeze(pred_arg_max)
    # flat_pred_arg_max = torch.reshape(pred_arg_max, (-1,))
    
    # soft_probs = (logits> 0.5).float()
    soft_probs = logits.float()
    flat_soft_probs = torch.reshape(soft_probs,  (-1,))

    if flat_soft_probs.size() == torch.Size([]):  # after the squeeze pred_arg_max become an int as the batch size become 1. so we need to make it as a list again 
        # print("pred_arg_max: ", pred_arg_max)
        # print("pred_arg_max.size(): ", pred_arg_max.size())
        flat_soft_probs = [flat_soft_probs.tolist()]  
    else:  
        flat_soft_probs = flat_soft_probs.tolist() 

    labels = labels.tolist() 
    for i, p, l in zip(ids, flat_soft_probs, labels):
        # print(i, pred_results)
        assert i not in pred_results
        pred_results[i] = [p, l] 

def convert_to_jst_dic(cases, all_justices): 
    jst_utt_dic = {jst: [] for jst in all_justices} 
    for case in cases: 
        for utt_id, utts in zip(case["ids"], case["convos"]): 
            tmp_jst = {}
            for utt in utts: 
                spk = utt["speaker_id"]
                if "j__" not in spk: continue 

                if spk not in tmp_jst: tmp_jst[spk] = {"case_id": case["case_id"], "utt_id": utt_id, "utt_text": []} 
                tmp_jst[spk]["utt_text"].append(utt["text"]) 
            
            for jst in tmp_jst: 
                jst_utt_dic[jst].append(tmp_jst[jst]) 
    return jst_utt_dic
                     
                     
def split_data_into_train_dev_test(i, k_folds, jst_layer_utts, seperate_justice_folds): 

    # i is the fold for test, i-1 is the fold for dev, and the rest is for train  
    # i in 0-9  
    # split the justices_use_list into folds first  

    test_fold = i 
    dev_fold = i-1 
    if dev_fold<0: 
        dev_fold = dev_fold+k_folds

    # and the rest is train  
    test_set= {jst: jst_layer_utts[jst] for jst in seperate_justice_folds[test_fold]} 
    dev_set = {jst: jst_layer_utts[jst] for jst in seperate_justice_folds[dev_fold]} 
    train_set = {jst: jst_layer_utts[jst] for jst in jst_layer_utts if jst not in test_set and jst not in dev_set} 

    return train_set, dev_set, test_set   

def split_data_into_train_dev_test_based_on_case(i, k_folds, jst_layer_utts, seperate_instance_folds): 

    # i is the fold for test, i-1 is the fold for dev, and the rest is for train  
    # i in 0-9  
    # split the justices_use_list into folds first  

    test_fold = i 
    dev_fold = i-1 
    if dev_fold<0: 
        dev_fold = dev_fold+k_folds

 
    test_set= {jst: [] for jst in jst_layer_utts} 
    dev_set = {jst: [] for jst in jst_layer_utts} 
    train_set = {jst: [] for jst in jst_layer_utts}  


    count = 0 
    for jst in jst_layer_utts: 
        for ele in jst_layer_utts[jst]: 
            if count in seperate_instance_folds[test_fold]: 
                test_set[jst].append(ele) 
            elif count in seperate_instance_folds[dev_fold]: 
                dev_set[jst].append(ele) 
            else:  # and the rest is train 
                train_set[jst].append(ele) 
               
            count+=1 


    return train_set, dev_set, test_set   


def store_result(data_type, acc, f1, macrof1, loss): 
    with open("./{}/evaluation_result.txt".format(config["log_dir"]), "a") as fw:
        fw.write("{}: Acc: {:.4f}, F1: {:.4f}, Macro F1: {:.4f}, Loss: {} \n".format(data_type, acc, f1, macrof1, loss)) 




def get_chronological_sep(original_cases): 
    # starting from here 
    year_count = {} 

    for ele in original_cases: 
        year = int(ele["case_id"].split("_")[0]) 
        if year not in year_count: year_count[year] = 0 
        year_count[year]+=1 
        
    year_count = dict(sorted(year_count.items()))


    all_count = sum(year_count.values()) 

    tmp_count = 0  
    year_sep = [[], [], []] 
    for k, v in year_count.items(): 
        if tmp_count < all_count*0.8: 
            year_sep[0].append(k) 
        elif tmp_count < all_count*0.9: 
            year_sep[1].append(k)  
        elif tmp_count < all_count: 
            year_sep[2].append(k) 
        tmp_count+=v 
    print(year_sep)  

    tmp = [v for k, v in year_count.items() if k in year_sep[0]] 
    print("train:", sum(tmp), all_count*0.8) 

    tmp = [v for k, v in year_count.items() if k in year_sep[1]] 
    print("dev:", sum(tmp), all_count*0.1) 

    tmp = [v for k, v in year_count.items() if k in year_sep[2]] 
    print("test", sum(tmp), all_count*0.1) 


    return year_sep
    
def get_casedata_structure(selected_cases):  

    jst_utt_dic = {} 
    for case in selected_cases: 
        for utt_id, utts in zip(case["ids"], case["convos"]): 
            tmp_jst = {}
            for utt in utts: 
                spk = utt["speaker_id"]
                if "j__" not in spk: continue 

                if spk not in tmp_jst: tmp_jst[spk] = {"case_id": case["case_id"], "utt_id": utt_id, "utt_text": []} 
                tmp_jst[spk]["utt_text"].append(utt["text"]) 
            
            for jst in tmp_jst: 
                if jst not in jst_utt_dic: jst_utt_dic[jst] = [] 

                jst_utt_dic[jst].append(tmp_jst[jst]) 

    return jst_utt_dic



def split_data_into_train_dev_test_based_on_case_chronological(original_cases, year_sep): 
    # pass

    train_set = [] 
    dev_set = [] 
    test_set = []
    for ele in original_cases: 
        year = int(ele["case_id"].split("_")[0]) 
        if year in year_sep[0]:  # train 
            train_set.append(ele) 
        elif year in year_sep[1]:  # dev 
            dev_set.append(ele) 
        elif year in year_sep[2]:  # test
            test_set.append(ele) 

    return get_casedata_structure(train_set), get_casedata_structure(dev_set), get_casedata_structure(test_set) 



    


def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    set_seed(123) 

    # torch.manual_seed(123)

    
    config = initialize_from_env()
    name = sys.argv[1] 
    
    
    # load all files first 

    # file_names = os.listdir(config["data_path"]) 

    original_cases = []
    with open(config["data_path"], "r") as fr: 
        for line in fr: 
            original_cases.append(json.loads(line)) 
    
    
    year_sep = get_chronological_sep(original_cases) 

    train_set, dev_set, test_set = split_data_into_train_dev_test_based_on_case_chronological(original_cases, year_sep) 

    # original_log_dir = config["log_dir"] 
    
    train_set = CaseDataset(config = config, data_set = train_set)
    dev_set = CaseDataset(config = config, data_set = dev_set)    
    test_set = CaseDataset(config = config, data_set = test_set) 


    #Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size = config["batch_size"], shuffle = True)
    dev_loader = DataLoader(dev_set, batch_size = config["batch_size"], shuffle = True) 
    test_loader = DataLoader(test_set, batch_size = config["batch_size"], shuffle = True) 

    print("Done preprocessing training, development and test data.")


    # gpu = 5  #gpu ID
    gpu = config["gpu"]

    
    print("Creating the CaseClassifier, initialised with pretrained {}".format(config["model_type"]))

    

    net = CaseClassifier(config = config) 

    
    net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))
    print("Done loading the pretrained model from {}".format(config["log_dir"])) 

    
    if gpu > -1:
        net.cuda(gpu) #Enable gpu support for the model
    else:
        net.cpu()


    print("Done loading the CaseClassifier.")
    net.eval() 
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    opti = optim.Adam(net.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])


    # dev_load_file = "answer_merged_dev.txt"
    # with open

    # train_acc, train_f1, train_loss = evaluate_file(net, criterion, train_loader, "train", gpu)
    train_macrof1, train_acc, train_f1, train_loss = evaluate(net, criterion, train_loader, gpu)
    print("Trainset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(train_acc, train_f1, train_macrof1, train_loss))
    store_result("Train", train_acc, train_f1, train_macrof1, train_loss) 

    # dev_acc, dev_f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
    dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
    print("Devset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(dev_acc, dev_f1, dev_macrof1, dev_loss))
    store_result("Dev", dev_acc, dev_f1, dev_macrof1, dev_loss)

    # test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, gpu)
    test_macrof1, test_acc, test_f1, test_loss = evaluate_file(net, criterion, test_loader, "test", gpu)
    print("Testset: Accuracy: {}; f1_score: {}; Macro f1_score: {}; Loss: {}".format(test_acc, test_f1,test_macrof1, test_loss))
    store_result("Test", test_acc, test_f1, test_macrof1, test_loss)


    print("Macro F1 in train, dev, and test:  {:.4f} &  {:.4f} &  {:.4f}".format(train_macrof1, dev_macrof1, test_macrof1))
    with open("./{}/evaluation_result.txt".format(config["log_dir"]), "a") as fw:
        fw.write("Macro F1 in train, dev, and test:  {:.4f} &  {:.4f} &  {:.4f}".format(train_macrof1, dev_macrof1, test_macrof1)) 


        
    #fine-tune the model
    # train(net, config, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


    # net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))

    # print("Best performance in Dev: acc, f1, loss")
    # dev_acc, f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
    # print(dev_acc, f1, dev_loss)

    # print("corresponding result in Train: acc, f1, loss")
    # t_dev_acc, t_f1, t_dev_loss = evaluate(net, criterion, train_loader, gpu)
    # print(t_dev_acc, t_f1, t_dev_loss)

    