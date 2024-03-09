
from unittest import result
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader
from dgllife.utils import EarlyStopping
from utils import *
from datasets import *
import models
import logging
from datetime import datetime
import time
from argparse import ArgumentParser
from utils import mkdir_p, split_dataset
from collections import defaultdict
import pandas as pd
import json
from sklearn.metrics import f1_score
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from new_datasets import MUTAGDataset, load_SeniGraph
from torch.utils.data import Subset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger(logpath):
    logging.basicConfig(filename=logpath, filemode='w', format='%(asctime)s -%(levelname)s- %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)
    return logger


def run_a_train_epoch(args, epoch, model, rationale, data_loader, optimizer_nn, optimizer_rationale, lambda_generation, lambda_vi, lambda_reg):
    model.train()
    # if epoch < 30:
    #     lambda_generation = 0
    #     lambda_vi = 0
    #train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = batch_data.to(args['device'])
        batch_data.y = batch_data.y.squeeze()

        ###E-step optimize rationale embedding
        
        for param in rationale.parameters():
            param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False
        XR, yR = rationale()
        loss_E = model(XR, yR, batch_data, lambda_generation, lambda_vi, lambda_reg, mode='E')
        
        optimizer_rationale.zero_grad()
        loss_E.backward()
        optimizer_rationale.step()
        #loss_E = torch.FloatTensor([0.0])
        # if epoch < 25:
        ####M-step optimize neural network
        for param in rationale.parameters():
            param.requires_grad = False

        
        if epoch >= 100:
            loss_M = torch.FloatTensor([0.0])
        
        else:
            for param in model.parameters():
                param.requires_grad = True
            
            
            loss_M = model(XR, yR, batch_data, lambda_generation, lambda_vi, lambda_reg, mode='M')
           
            optimizer_nn.zero_grad()
            loss_M.backward()
            optimizer_nn.step()
        
        # else:
        #     loss_M = torch.FloatTensor([0.0])
        
        
        #print(logits.shape, labels.shape)

        #train_meter.update(m(logits), labels)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss_E {:.4f}, loss_M {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss_E.item(), loss_M.item()))
    #train_score = train_meter.compute_metric(args['metric'])
    
    return loss_E.item(), loss_M.item()

def run_an_eval_epoch(args, model, rationale, data_loader):
    model.eval()
    eval_meter = Meter()
    XR, yR = rationale()
    total_num = 0
    num_accurate = 0
    num_accurate_cosine = 0
    num_accurate_topk = 0
    num_accurate_cosine_topk = 0
    
    labels_list = []
    rationale_prediction_list = []
    rationale_prediction_cosine_list = []
    rationale_prediction_topk_list = []
    rationale_prediction_cosine_topk_list = []
    
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(args['device'])
            batch_data.y = batch_data.y.squeeze()
            #logits = model.predict(batch_data, XR, yR)
            probs = model.predict(batch_data, XR, yR)
            labels = batch_data.y
            
            #print(labels)
            if args['training_type'] == 'sigmoid':
                m = nn.Sigmoid()
            elif args['training_type'] == 'softmax':
                m = nn.Softmax(dim=1)
            else:
                raise ValueError('Expect training type to be "sigmoid" or "softmax", got {}'.format(args['training_type']))  
            #eval_meter.update(m(logits), labels)
            eval_meter.update(probs, labels)
            rationale_prediction, rationale_prediction_cosine, rationale_prediction_topk, rationale_prediction_cosine_topk = model.rationale_predict(batch_data, XR, yR)
            #print(rationale_prediction.shape, labels.shape)
            num_accurate += torch.sum(rationale_prediction == labels.squeeze()).item()
            num_accurate_cosine += torch.sum(rationale_prediction_cosine == labels.squeeze()).item()
            num_accurate_topk += torch.sum(rationale_prediction_topk == labels.squeeze()).item()
            num_accurate_cosine_topk += torch.sum(rationale_prediction_cosine_topk == labels.squeeze()).item()
            
            total_num += labels.shape[0]  
            
            
            labels_list.append(labels.cpu().numpy())
            rationale_prediction_list.append(rationale_prediction.cpu().numpy())
            rationale_prediction_cosine_list.append(rationale_prediction_cosine.cpu().numpy())
            rationale_prediction_topk_list.append(rationale_prediction_topk.cpu().numpy())
            rationale_prediction_cosine_topk_list.append(rationale_prediction_cosine_topk.cpu().numpy())
    

        
    labels_list = np.concatenate(labels_list, axis=0)
    rationale_prediction_list = np.concatenate(rationale_prediction_list, axis=0)
    rationale_prediction_cosine_list = np.concatenate(rationale_prediction_cosine_list, axis=0)
    rationale_prediction_topk_list = np.concatenate(rationale_prediction_topk_list, axis=0)
    rationale_prediction_cosine_topk_list = np.concatenate(rationale_prediction_cosine_topk_list, axis=0)
    
    f1_l2 = f1_score(labels_list, rationale_prediction_list, average='binary')
    f1_cosine = f1_score(labels_list, rationale_prediction_cosine_list, average='binary')
    f1_l2_topk = f1_score(labels_list, rationale_prediction_topk_list, average='binary')
    f1_cosine_topk = f1_score(labels_list, rationale_prediction_cosine_topk_list, average='binary')
                
    #print(num_accurate, total_num)          
    rationale_accuracy = num_accurate / total_num        
    rationale_accuracy_cosine = num_accurate_cosine / total_num
    rationale_accuracy_topk = num_accurate_topk / total_num        
    rationale_accuracy_cosine_topk = num_accurate_cosine_topk / total_num
    return eval_meter.compute_metric(args['metric']), eval_meter.compute_metric('ece'), rationale_accuracy, rationale_accuracy_cosine, rationale_accuracy_topk, rationale_accuracy_cosine_topk, f1_l2, f1_cosine, f1_l2_topk, f1_cosine_topk
            


def main():
    parser = ArgumentParser('Molecular Property Prediction')
    parser.add_argument('-d', '--dataset', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'MUTAG2', 'HIV', 'PCBA', 'Tox21', 'IMDB', 'SST'],
                        help='Dataset to use')
    parser.add_argument('-mo', '--model', choices=['RGCN', 'GAT'],
                        help='Model to use')
    parser.add_argument('-tp', '--training_type', choices=['softmax', 'sigmoid'],
                        help='Use softmax or sigmoid at the final classifier layer')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random', 'degree'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 0.8,0.1,0.1)')
    parser.add_argument('-lg', '--lambda_generation', type=float, default=0,
                        help='hyperparameter for generation loss')
    parser.add_argument('-lv', '--lambda_vi', type=float, default= 0.01,
                        help='hyperparameter for variational inference loss')
    parser.add_argument('-lre', '--lambda_reg', type=float, default=0.0,
                        help='hyperparameter for rationale regularization')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'accuracy'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=100,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-pa', '--patience', type=int, default=50,
                        help='Number of epochs for early stopping')
    parser.add_argument('-bz', '--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('-lr_nn', '--lr_nn', type=float, default=1e-4,
                        help='learning rate for training neural network')
    parser.add_argument('-lr_r', '--lr_r', type=float, default=1e-4,
                        help='learning rate for training rationale')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0,
                        help='weight decay for the optimizer')
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-nr', '--num_runs', type=int, default=3,
                        help='Number of runs')
    parser.add_argument('-npr', '--num_pos_rationale', type=int, default=5,
                        help='Number of positive rationales')
    parser.add_argument('-nnr', '--num_neg_rationale', type=int, default=5,
                        help='Number of negative rationales')
    parser.add_argument('-up', '--use_plus', default=True,  action='store_false',
                        help='whether use plus embedding for FNP')
    
    args = parser.parse_args().__dict__

    base_save = args['result_path']
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_folder = os.path.join(base_save, args['dataset'], args['model'], 'graph_fnp', str(experiment_id))
    mkdir_p(save_folder)
    
    logpath = os.path.join(save_folder, 'training_log.log')
    logger = set_logger(logpath=logpath)
    
    with open(os.path.join(save_folder, 'experiment_configuration.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, indent=4)
    
    node_feature_dim = 0
    
    logger.info(args)
    atom_dim = 1
    bond_dim = 1
    #load dataset
    if args['dataset'] == 'BBBP':
        num_relations = 4
        dataset = BBBPDataset()
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'BACE':
        num_relations = 4
        dataset = BACEDataset()
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'HIV':
        num_relations = 4
        dataset = HIVDataset()
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'ClinTox':
        num_relations = 4
        dataset = ClinToxDataset()
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'Tox21':
        dataset = Tox21Dataset()
        num_relations = 4
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'ToxCast':
        dataset = ToxCastDataset()
        num_relations = 4
        atom_dim = dataset.atom_dim
        bond_dim = dataset.bond_dim
    elif args['dataset'] == 'MUTAG':
        dataset = MUTAGDataset()
    elif args['dataset'] == 'IMDB':
        dataset = TUDataset(root='./datasets/IMDB/', name='IMDB-BINARY')
        num_relations = 1
        with open(os.path.join('./datasets/IMDB/IMDB-BINARY', 'split_idx.json')) as f:
            split_idx = json.load(f)
        max_degree = 0
        for data in dataset:
            data.x = degree(data.edge_index[0], data.num_nodes)
            if max_degree < data.x.max():
                max_degree = data.x.max()
        node_feature_dim = max_degree + 1 
    elif args['dataset'] == 'SST':
        dataset = load_SeniGraph()
        num_relations = 1
        node_feature_dim = dataset[0].x.shape[1]
    #print(len(dataset))
    elif args['dataset'] == 'MUTAG2':
        dataset = MUTAGDataset(name='Mutagenicity')

        #print(dataset[0].y.shape)
        num_relations = 1
        with open(os.path.join('./datasets/MUTAG/Mutagenicity', 'split_idx.json')) as f:
            split_idx = json.load(f)
        node_feature_dim = dataset[0].x.shape[1]



    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')          
        
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = split_dataset(args, dataset)
    elif args['split'] == 'random':
        #split_idx = dataset.split_idx
        #print(max(split_idx['train']))
        train_set = Subset(dataset, list(map(int, split_idx['train'])))
        val_set = Subset(dataset, list(map(int, split_idx['valid'])))
        test_set = Subset(dataset, list(map(int, split_idx['test'])))
    elif args['split'] == 'degree':
        train_set, test_set = [], []
        for g in dataset:
            if g.num_edges <= 2: continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.76785714:
                train_set.append(g)
            elif degree <= 1.57142857:
                test_set.append(g)

        val_set = train_set[:int(len(train_set) * 0.1)]
        train_set = train_set[int(len(train_set) * 0.1):]
        
    train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                                num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                                num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                num_workers=args['num_workers'])
    
    train_labels = []
    for i in range(len(train_set)):
        g = train_set[i]
        train_labels.append(g.y.item())
    num_pos = len([x for x in train_labels if x==1])  
    num_neg = len([x for x in train_labels if x==0])   
    num_pos = float(num_pos)
    num_neg = float(num_neg)
    
    #print(max(train_labels))
    pos_weight = (1./num_pos)/((1./num_pos)+(1./num_neg)) *torch.ones([1], dtype=torch.float32).to(args['device'])
    



    df_results = pd.DataFrame(columns=['seed', 'val_score', 'val_ece', 'test_score', 'val_ece', 'best_test_score', 'best_test_ece'])
    final_results = []
    for run in range(args['num_runs']):
        logger.info('##############################################')
        logger.info('Start {}-th run'.format(run+1))
        #create directory for this run
        mkdir_p(os.path.join(save_folder, str(run)))
        #set random seed
        seed = run
        set_seed(seed)
        
        #split dataset

        #print(pos_weight)        
        
        #node_feature_dim = dataset[0].x.shape[1]
        if args['training_type'] == 'sigmoid':
            output_dim = 1 
        elif args['training_type'] == 'softmax':
            output_dim = 2
            weight = torch.cat([1-pos_weight, pos_weight], dim=0).to(args['device'])
        else:
            raise ValueError('Expect training type to be "sigmoid" or "softmax", got {}'.format(args['training_type']))    
        if args['model'] == 'RGCN':
            model = models.FNP_RGCNPredictor(num_relations=num_relations, use_plus=args['use_plus'], node_feature_dim=node_feature_dim, output_dim=output_dim, label_weight=weight, max_num_nodes=30, atom_dim=atom_dim, bond_dim=bond_dim, lambda_generation=args['lambda_generation'], lambda_vi=args['lambda_vi'])
        elif args['model'] == 'GAT':
            model = models.FNP_RGATPredictor(num_relations=num_relations, use_plus=args['use_plus'], node_feature_dim=node_feature_dim, output_dim=output_dim, label_weight=weight, max_num_nodes=30, atom_dim=atom_dim, bond_dim=bond_dim, lambda_generation=args['lambda_generation'], lambda_vi=args['lambda_vi'])
        model = model.to(args['device'])   
        
        rationale = models.Rationle(args['num_pos_rationale'], args['num_neg_rationale'])
        rationale = rationale.to(args['device'])   
           
            
        optimizer_nn = Adam(model.parameters(), lr=args['lr_nn'],
                            weight_decay=args['weight_decay'])
        
        optimizer_rationale = SGD(rationale.parameters(), lr=args['lr_r'])
        
        model_path = os.path.join(save_folder, str(run), 'model.pt')
        rationale_path = os.path.join(save_folder, str(run), 'rationale.pt')


        results = {}
        metric_curve = defaultdict(list)

        for epoch in range(args['num_epochs']):
            # Train
            loss_E, loss_M = run_a_train_epoch(args, epoch, model, rationale, train_loader, optimizer_nn, optimizer_rationale, args['lambda_generation'], args['lambda_vi'], args['lambda_reg'])

            # Validation and early stop
            val_score, val_ece, val_rationale_accuracy, val_rationale_accuracy_cosine, val_rationale_accuracy_topk, val_rationale_accuracy_cosine_topk, val_f1_l2, val_f1_cosine, val_f1_l2_topk, val_f1_cosine_topk = run_an_eval_epoch(args, model, rationale, val_loader)
            test_score, test_ece, test_rationale_accuracy, test_rationale_accuracy_cosine, test_rationale_accuracy_topk, test_rationale_accuracy_cosine_topk, test_f1_l2, test_f1_cosine, test_f1_l2_topk, test_f1_cosine_topk = run_an_eval_epoch(args, model, rationale, test_loader)
            #early_stop = stopper.step(val_score, model)
            logger.info('epoch {:d}/{:d}: training E step loss {:.4f}, M step loss {:.4f}|| validation {} {:.4f}, validation ece {:.4f}, validation rationale: {:.4f}|| test {} {:.4f}, test ece {:.4f}, test rational accuracy: {:.4f}, test rational accuracy cosine: {:.4f}'.format( 
                        epoch + 1, args['num_epochs'], loss_E, loss_M, args['metric'], val_score, val_ece, val_rationale_accuracy,
                        args['metric'], test_score, test_ece, test_rationale_accuracy, test_rationale_accuracy_cosine))

            metric_curve['valid_score_curve'].append(val_score)
            metric_curve['test_score_curve'].append(test_score)   
            metric_curve['valid_ece_curve'].append(val_ece)
            metric_curve['test_ece_curve'].append(test_ece)  
            metric_curve['train_lossE_curve'].append(loss_E)
            metric_curve['train_lossM_curve'].append(loss_M)  
            metric_curve['val_rationale_accuracy_curve'].append(val_rationale_accuracy)      
            metric_curve['test_rationale_accuracy_curve'].append(test_rationale_accuracy) 
            metric_curve['test_rationale_f1_curve'].append(test_f1_l2)
            metric_curve['test_rationale_f1_cosine_curve'].append(test_f1_cosine)
            
            
            metric_curve['val_rationale_accuracy_topk_curve'].append(val_rationale_accuracy_topk)      
            metric_curve['test_rationale_accuracy_topk_curve'].append(test_rationale_accuracy_topk) 
            metric_curve['test_rationale_f1_topk_curve'].append(test_f1_l2_topk) 
            metric_curve['test_rationale_f1_cosine_topk_curve'].append(test_f1_cosine_topk)

            metric_curve['val_rationale_accuracy_cosine_curve'].append(val_rationale_accuracy_cosine)
            metric_curve['test_rationale_accuracy_cosine_curve'].append(test_rationale_accuracy_cosine)
            metric_curve['val_rationale_accuracy_cosine_topk_curve'].append(val_rationale_accuracy_cosine_topk)
            metric_curve['test_rationale_accuracy_cosine_topk_curve'].append(test_rationale_accuracy_cosine_topk)
           
                
        torch.save(rationale.state_dict(), rationale_path)
        torch.save(model.state_dict(), model_path)
        pd.DataFrame(metric_curve).to_csv(os.path.join(save_folder, str(run), 'metric_curve.csv'), index=False)

        #best_val_epoch = np.argmax(np.array(metric_curve['valid_score_curve']))
        best_val_epoch = args['num_epochs'] - 1
        results['random seed'] = seed
        results['val_score'] = metric_curve['valid_score_curve'][best_val_epoch]
        results['test_score'] = metric_curve['test_score_curve'][best_val_epoch]
        results['val_ece'] = metric_curve['valid_ece_curve'][best_val_epoch]
        results['test_ece'] = metric_curve['test_ece_curve'][best_val_epoch]
        results['best_test_score'] = max(metric_curve['test_score_curve'])
        #results['best_test_ece'] = min(metric_curve['test_ece_curve'])
        results['test_rationale_accuracy'] = metric_curve['test_rationale_accuracy_curve'][best_val_epoch]
        #results['best_test_rationale_accuracy'] = max(metric_curve['test_rationale_accuracy_curve'])
        results['test_rationale_accuracy_cosine'] = metric_curve['test_rationale_accuracy_cosine_curve'][best_val_epoch]
        #results['best_test_rationale_accuracy_cosine'] = max(metric_curve['test_rationale_accuracy_cosine_curve'])
        results['test_rationale_accuracy_topk'] = metric_curve['test_rationale_accuracy_topk_curve'][best_val_epoch]
        #results['best_test_rationale_accuracy_topk'] = max(metric_curve['test_rationale_accuracy_topk_curve'])
        results['test_rationale_accuracy_cosine_topk'] = metric_curve['test_rationale_accuracy_cosine_topk_curve'][best_val_epoch]
        #results['best_test_rationale_accuracy_cosine_topk'] = max(metric_curve['test_rationale_accuracy_cosine_topk_curve'])
        
        
        results['test_rationale_f1'] = metric_curve['test_rationale_f1_curve'][best_val_epoch]
        #results['best_test_rationale_f1'] = max(metric_curve['test_rationale_f1_curve'])
        results['test_rationale_f1_cosine'] = metric_curve['test_rationale_f1_cosine_curve'][best_val_epoch]
        #results['best_test_rationale_f1_cosine'] = max(metric_curve['test_rationale_f1_cosine_curve'])

        
        results['test_rationale_f1_topk'] = metric_curve['test_rationale_f1_topk_curve'][best_val_epoch]
        #results['best_test_rationale_f1_topk'] = max(metric_curve['test_rationale_f1_topk_curve'])
        
        results['test_rationale_f1_cosine_topk'] = metric_curve['test_rationale_f1_cosine_topk_curve'][best_val_epoch]
        #results['best_test_rationale_f1_cosine_topk'] = max(metric_curve['test_rationale_f1_cosine_topk_curve'])
        
        final_results.append(results)
    
    df_results = pd.DataFrame(final_results)
    df_results.set_index('random seed', inplace=True)
    df_results.loc['mean'] = df_results.mean()
    df_results.loc['std'] = df_results.std()
    pd.DataFrame(df_results).to_csv(os.path.join(save_folder, 'results.csv'))

        
        
         
        

if __name__ == '__main__':
    main()