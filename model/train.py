import time 

from torch.utils.data import DataLoader
from CaseDataset import CaseDataset
from CaseClassifier import CaseClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pyhocon
import os
import errno

import json 
import copy 
import random  
import numpy as np

import matplotlib.pyplot as plt

def plot_learning_curve(train_loss = None, dev_loss = None, save_path= "./"): 
    if train_loss: 
      
        epoch_index = [i+1 for i in range(len(train_loss))] 
        
        plt.plot(epoch_index, train_loss, label = "Train") 
        
    if dev_loss:  
        epoch_index = [i+1 for i in range(len(dev_loss))] 
        plt.plot(epoch_index, dev_loss, label = "Dev") 
    
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
#     plt.show()
    plt.xticks(epoch_index, epoch_index)
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()

def plot_evaluations(train_eval = None, dev_eval = None, save_path= "./"): 
    if not train_eval: 
        
        epoch_index = [i+1 for i in range(len(train_eval))] 
    
        plt.plot(epoch_index, train_eval, label = "Train") 
    
    if dev_eval:  
        epoch_index = [i+1 for i in range(len(dev_eval))] 

        plt.plot(epoch_index, dev_eval, label = "Dev") 
    
    plt.title("Evaluation")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.tight_layout()
#     plt.show()
    plt.xticks(epoch_index, epoch_index)
    plt.legend()
    plt.savefig(os.path.join(save_path, "f1.png"))
    plt.close()


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


def train(net, config, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0 
    best_f1 = 0
    best_macrof1 = 0
    st = time.time()

    train_loss_list = []
    dev_loss_list = []

    train_eval_list = []
    dev_eval_list = [] 

    temp_train_loss = [] 
    global_steps = 0 
    print("start training...")
    for ep in range(max_eps): 
        net.train() # need to define here again as it will be set to eval() in evaluate function 
        # temp_train_eval = []
        print("training ep: ", ep)
        for it, (seq, attn_masks, labels, ids) in enumerate(train_loader):
            #Clear gradients
            # print(it)
            # print("type(seq):", type(seq))
            # print("type(attn_masks):", type(attn_masks)) 
            # print("type(labels):" , type(labels))
            # print("seq.size(): ", seq.size() )
            # print('\n')

            opti.zero_grad()  
            #Converting these to cuda tensors
            if gpu >= 0:
                seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            else:
                seq, attn_masks, labels = seq.cpu(), attn_masks.cpu(), labels.cpu()

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)
            # print("logits ", logits)
            # print("lables: ", labels)
            #Computing loss 

            # print(logits.squeeze(-1))
            loss = criterion(logits.squeeze(-1), labels.float())

            temp_train_loss.append(loss.item()) 

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()
              
            global_steps +=1  
            
            if global_steps % config["eval_steps"] == 0:  # go into evaluation 
                print("Step {} evaluating...".format(global_steps))
                # dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_random_sample(net, criterion, dev_loader, gpu, global_steps)
                dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
                # train_macrof1, train_acc, train_f1, train_loss = evaluate_random_sample(net, criterion, train_loader, gpu, global_steps)
                # train_macrof1, train_acc, train_f1, train_loss = evaluate(net, criterion, train_loader, gpu)

                #  the ploting part  
                print("the ave and eval train loss: ", sum(temp_train_loss)/len(temp_train_loss))  


                train_loss_list.append(sum(temp_train_loss)/len(temp_train_loss))

                temp_train_loss = []

                dev_loss_list.append(dev_loss) 

                plot_learning_curve(train_loss_list, dev_loss_list, config["log_dir"])
                

                # train_eval_list.append(sum(temp_train_eval)/len(temp_train_eval)) 
                # train_eval_list.append(train_f1)
                dev_eval_list.append(dev_macrof1)

                plot_evaluations([], dev_eval_list, config["log_dir"])
                #  done with ploting part 


                print("Step {} complete! Development Accuracy: {}; f1_score: {}; Macro f1_score: {}; Development Loss: {}".format(global_steps, dev_acc, dev_f1, dev_macrof1, dev_loss))
                
                if dev_macrof1 >= best_macrof1:
                    print("Best development Macro F1 improved from {} to {}, saving model...".format(best_macrof1, dev_macrof1))
                    best_macrof1 = dev_macrof1 

                    torch.save(net.state_dict(), './{}/sstcls_best.dat'.format(config["log_dir"]))
                    # torch.save(net.state_dict(), './{}/saved_weights_LSTM.pt'.format(config["log_dir"]))
        
                net.train() # need to define here again as it will be set to eval() in evaluate function 

            if global_steps > config["max_traning_steps"]: break 
        
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

    mean_acc, mean_loss = 0, 0
    mean_f1 = 0
    mean_macrof1 = 0 
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
            # tmp_macrof1, tmp_acc, tmp_f1 = get_accuracy_from_logits(logits, labels)
            # mean_acc += tmp_acc
            # mean_f1 += tmp_f1
            # mean_macrof1 += tmp_macrof1
            count += 1

            golds = torch.cat([golds, labels], 0) 
            preds = torch.cat([preds, logits], 0) 
            
            # store_prediciton(pred_results, ids, logits) 
    # print(pred_results)
    # with open("./{}/pred_{}.txt".format(config["log_dir"], file_type), "w") as fw: 
    #     fw.write(json.dumps(pred_results))
    # print("preds: ", preds) 
    # print("golds: ", golds)
    macrof1, acc, f1 = get_accuracy_from_logits(preds, golds)  

    return macrof1, acc, f1,  mean_loss / count

    # return mean_macrof1 / count, mean_acc / count, mean_f1 / count,  mean_loss / count

    

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



def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    seed = 123
    set_seed(seed) 

    # torch.manual_seed(123)

    
    config = initialize_from_env()
    name = sys.argv[1] 
    
    # do something for the k_fold:  
    justices_in_chronological_order = ['j__john_m_harlan', 'j__hugo_l_black', 'j__stanley_reed', 'j__felix_frankfurter', 'j__william_o_douglas', 'j__harold_burton', 'j__tom_c_clark', 'j__sherman_minton', 'j__earl_warren', 'j__john_m_harlan2', 'j__william_j_brennan_jr', 'j__charles_e_whittaker', 'j__potter_stewart', 'j__byron_r_white', 'j__arthur_j_goldberg', 'j__abe_fortas', 'j__thurgood_marshall', 'j__warren_e_burger', 'j__harry_a_blackmun', 'j__william_h_rehnquist', 'j__lewis_f_powell_jr', 'j__john_paul_stevens', 'j__sandra_day_oconnor', 'j__antonin_scalia', 'j__anthony_m_kennedy', 'j__david_h_souter', 'j__clarence_thomas', 'j__ruth_bader_ginsburg', 'j__stephen_g_breyer', 'j__samuel_a_alito_jr', 'j__john_g_roberts_jr', 'j__sonia_sotomayor', 'j__elena_kagan', 'j__neil_gorsuch', 'j__brett_m_kavanaugh'] 
    

    justices_use_list = copy.deepcopy(justices_in_chronological_order )

    if config["random_selection"]:  
        # random.seed(seed)
        random.shuffle(justices_use_list) 

    # here we seperate the jsts into folds  
    num_jst = len(justices_use_list)
    seperate_justice_folds = []  
    for k in range(config["k_folds"]): 
        seperate_justice_folds.append(justices_use_list[int(num_jst/config["k_folds"]*k):int(num_jst/config["k_folds"]*(k+1))])   

    assert sum(len(ele) for ele in seperate_justice_folds) == len(justices_use_list)  

    for k in range(config["k_folds"]): 
        print(k, len(seperate_justice_folds[k]), seperate_justice_folds[k])

    # load all files first 

    # file_names = os.listdir(config["data_path"]) 

    original_cases = []
    with open(config["data_path"], "r") as fr: 
        for line in fr: 
            original_cases.append(json.loads(line)) 
    
    jst_layer_utts = convert_to_jst_dic(original_cases, justices_use_list)


    # get all the case year 
    all_years = set()
    for jst in jst_layer_utts: 
        for ele in jst_layer_utts[jst]: 
            current_year = ele["case_id"].split("_")[0] 
            all_years.add(f'[{current_year}]') 

    all_years = list(sorted(all_years) ) 
    print(all_years)
    # ['1955', '1956', '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']  
    # ['[1955]', '[1956]', '[1957]', '[1958]', '[1959]', '[1960]', '[1961]', '[1962]', '[1963]', '[1964]', '[1965]', '[1966]', '[1967]', '[1968]', '[1969]', '[1970]', '[1971]', '[1972]', '[1973]', '[1974]', '[1975]', '[1976]', '[1977]', '[1978]', '[1979]', '[1980]', '[1981]', '[1982]', '[1983]', '[1984]', '[1985]', '[1986]', '[1987]', '[1988]', '[1989]', '[1990]', '[1991]', '[1992]', '[1993]', '[1994]', '[1995]', '[1996]', '[1997]', '[1998]', '[1999]', '[2000]', '[2001]', '[2002]', '[2003]', '[2004]', '[2005]', '[2006]', '[2007]', '[2008]', '[2009]', '[2010]', '[2011]', '[2012]', '[2013]', '[2014]', '[2015]', '[2016]', '[2017]', '[2018]', '[2019]'] 



    # also try to get folds in random setting  

    num_of_instances = sum([len(ele) for ele in jst_layer_utts.values()]) 
    print("num_of_instances:", num_of_instances)
    selected_index = [i for i in range(num_of_instances)]  
    # print(selected_index) 
    # random.seed(seed)
    random.shuffle(selected_index)  
    # print(selected_index)
    seperate_instance_folds = []  
    for k in range(config["k_folds"]): 
        seperate_instance_folds.append(selected_index[int(num_of_instances/config["k_folds"]*k):int(num_of_instances/config["k_folds"]*(k+1))])  




    original_log_dir = config["log_dir"] 
    for i in range(0, config["k_folds"]):   
    # for i in range(5, 10):   
        print("processing {}/{} folds".format(i, config["k_folds"])) 
        
        config["log_dir"] = mkdirs(os.path.join(original_log_dir, str(i)))

        if config["fold_on_jst"]: 
            train_set, dev_set, test_set = split_data_into_train_dev_test(i, config["k_folds"], jst_layer_utts, seperate_justice_folds)
        else:  
            print("random on instances")
            train_set, dev_set, test_set = split_data_into_train_dev_test_based_on_case(i, config["k_folds"], jst_layer_utts, seperate_instance_folds)

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

        print("Done with creating the CaseClassifier.")
        

        # print(" are we here ******************")
        if gpu > -1:
            net.cuda(gpu) #Enable gpu support for the model
        else:
            net.cpu()
        # print(" are we here ===========================")

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()

        opti = optim.Adam(net.parameters(), lr = config["lr"], weight_decay = config["weight_decay"])



        num_epoch = config["num_epoch"]


        # print(" are we here -----------------------------")
        #fine-tune the model
        train(net, config, criterion, opti, train_loader, dev_loader, num_epoch, gpu)


        net.load_state_dict(torch.load('./{}/sstcls_best.dat'.format(config["log_dir"])))

        print("Best performance in Dev: acc, f1, macrof1, loss")
        # dev_acc, dev_f1, dev_loss = evaluate(net, criterion, dev_loader, gpu)
        dev_macrof1, dev_acc, dev_f1, dev_loss = evaluate_file(net, criterion, dev_loader, "dev", gpu)
        print(dev_acc, dev_f1, dev_macrof1, dev_loss)

        print("Corresponding result in Train: acc, f1, macrof1, loss")
        t_macrof1, t_dev_acc, t_f1, t_dev_loss = evaluate(net, criterion, train_loader, gpu)
        print(t_dev_acc, t_f1, t_macrof1, t_dev_loss)
        

        print("Corresponding result in Test: acc, f1, macrof1, loss")
        # test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, gpu)
        test_macrof1, test_acc, test_f1, test_loss = evaluate_file(net, criterion, test_loader, "test", gpu)
        print(test_acc, test_f1, test_macrof1, test_loss)

        