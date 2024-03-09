import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import Dataset, InMemoryDataset
from rdkit import Chem
import numpy as np
import pandas as pd
from collections import defaultdict
import dgl
from rdkit.Chem import BondType
from features import *
from torch_geometric.data import Data
import json

def filter_out_invalid_smiles(smiles_list, label_list):
    valid_smiles_list, valid_label_list = [], []
    for smiles, label in zip(smiles_list, label_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid_smiles_list.append(smiles)
        valid_label_list.append(label)
    return valid_smiles_list, valid_label_list

def from_2Dcsv(csv_file, smiles_field, task_list_field):
    if smiles_field is not None:
        columns = [smiles_field] + task_list_field
        df = pd.read_csv(csv_file, usecols=columns)
        smiles_list = df[smiles_field].tolist()
    else:
        columns = task_list_field
        df = pd.read_csv(csv_file, usecols=columns)
        smiles_list = None
    task_label_list = []
    for task in task_list_field:
        task_label_list.append(df[task].tolist())
    task_label_list = np.stack(task_label_list, axis=1)
    return smiles_list, task_label_list

def _get_max_atom_num_from_smiles_list(smiles_list):
    molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    num_list = [mol.GetNumAtoms() for mol in molecule_list]
    return max(num_list)


def preprocessing_classification_task_label_list(task_label_list):
    task_label_list = np.array(task_label_list)
    #task_label_list[task_label_list == 0] = -1
    task_label_list = np.nan_to_num(task_label_list)
    return task_label_list


class ToxCastDataset(InMemoryDataset):
    def __init__(self, root = 'datasets/ToxCast', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        file_name = '{}/toxcast_data.csv'.format(self.root)
        df = pd.read_csv(file_name, nrows=1)
        self.given_targets = list(df.columns[1:])
        print('# of targets: {}'.format(len(self.given_targets)))
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='smiles', task_list_field=[self.given_targets[0]])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []
        with open(os.path.join(self.root, 'split_idx.json')) as f:
            self.split_idx = json.load(f)
        self.n_samples = len(self.molecule_list)
        super(ToxCastDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        
        return 'toxcast_data.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'toxcast_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(Chem.MolToSmiles(mol))
            # if len(edge_index) == 0:
            #     edge_index = np.zeros((2, 1), dtype=np.int64)
            #     edge_attr = np.zeros((1, ), dtype=np.int64)
            if len(edge_index) == 0:
                continue
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
        
        print(len(data_list))
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])






class Tox21Dataset(InMemoryDataset):
    def __init__(self, root = 'datasets/Tox21', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='smiles', task_list_field=['NR-AR'])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []
        with open(os.path.join(self.root, 'split_idx.json')) as f:
            self.split_idx = json.load(f)
        self.n_samples = len(self.molecule_list)
        super(Tox21Dataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        
        return 'tox21.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'tox21_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(Chem.MolToSmiles(mol))
            # if len(edge_index) == 0:
            #     edge_index = np.zeros((2, 1), dtype=np.int64)
            #     edge_attr = np.zeros((1, ), dtype=np.int64)
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])






class ClinToxDataset(InMemoryDataset):
    def __init__(self, root = 'datasets/ClinTox', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='smiles', task_list_field=['CT_TOX'])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []
        with open(os.path.join(self.root, 'split_idx.json')) as f:
            self.split_idx = json.load(f)
        self.n_samples = len(self.molecule_list)
        super(ClinToxDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        
        return 'clintox.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'clintox_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(ih)
            # if len(edge_index) == 0:
            #     edge_index = np.zeros((2, 1), dtype=np.int64)
            #     edge_attr = np.zeros((1, ), dtype=np.int64)
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])






class HIVDataset(InMemoryDataset):
    def __init__(self, root = 'datasets/HIV', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='smiles', task_list_field=['HIV_active'])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []

        self.n_samples = len(self.molecule_list)
        super(HIVDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        
        return 'HIV.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'hiv_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(ih)
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])






class BACEDataset(InMemoryDataset):
    def __init__(self, root = 'datasets/BACE', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='mol', task_list_field=['Class'])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []

        self.n_samples = len(self.molecule_list)
        super(BACEDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        
        return 'bace.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'bace_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(ih)
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])




class BBBPDataset(InMemoryDataset):
    def __init__(self, root = 'datasets/BBBP', transform=None, pre_transform = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        '''

        self.root = root
        
        smiles_list, task_label_list  = from_2Dcsv(csv_file=os.path.join(self.root,self.raw_file_names), smiles_field='smiles', task_list_field=['p_np'])
        smiles_list, task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        self.molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        self.smiles_list = smiles_list
        self.atom_type_list = []
        #self.bond_type_list = []
        atom_num_list = [mol.GetNumAtoms() for mol in self.molecule_list]
        print('0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(atom_num_list, 0.95), np.quantile(atom_num_list, 0.9), np.quantile(atom_num_list, 0.8)))

        for mol in self.molecule_list:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                self.atom_type_list.append(str(atom.GetSymbol()))
                #node_feature[atom_idx] = node_feature_func(atom)
            #for bond in mol.GetBonds():
                #self.bond_type_list.append(str(bond.GetBondType()))
        print(sorted(set(self.atom_type_list)))
        self.atom_type_list = sorted(set(self.atom_type_list))
        #self.bond_type_list = list(set(self.bond_type_list))
        self.atom_dim = len(self.atom_type_list)
        SMILE_to_index = dict((c, i) for i, c in enumerate(self.atom_type_list))
        index_to_SMILE = dict((i, c) for i, c in enumerate(self.atom_type_list))
        self.atom_mapping = dict(SMILE_to_index)
        self.atom_mapping.update(index_to_SMILE)
        self.bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        
        self.bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC})
        self.bond_dim = 4+1        
        self.max_atoms = max(atom_num_list)
        self.graph_lists = []
        self.graph_labels = torch.FloatTensor(preprocessing_classification_task_label_list(task_label_list))
        self.node_types = []
        self.adjacency = []

        self.n_samples = len(self.molecule_list)
        super(BBBPDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def atom_mapping(self):
    #     return self.atom_mapping
    # @property
    # def bond_mapping(self):
    #     return self.bond_mapping
    
    @property
    def raw_file_names(self):
        
        return 'BBBP.csv'
    
    @property 
    def max_num_nodes(self):
        return self.max_atoms

    @property
    def num_node_types(self):
        return self.atom_dim
    
    def num_edge_features(self):
        return self.bond_dim

    @property
    def processed_file_names(self):
        return 'BBBP_processed.pt'

    @property
    def smiles(self):
        return self.smiles_list
    @property
    def mols(self):
        return self.molecule_list

    def process(self):
        
        data_list = []
        ### read pyg graph list
        for mol_id, mol in enumerate(self.molecule_list):
            # Initialize adjacency and feature tensor
            adjacency = np.zeros((self.bond_dim, self.max_atoms, self.max_atoms), "float32")
            node_types  = np.zeros((self.max_atoms, self.atom_dim), "float32") ## last column represents no atom
            #h_u = []
            #b_u = []
            x = []
            k = 0
            for atom in mol.GetAtoms():
                i = atom.GetIdx()
                atom_id = self.atom_mapping[atom.GetSymbol()]
                node_types[i] = np.eye(self.atom_dim)[atom_id]
                # loop over one-hop neighbors
                for neighbor in atom.GetNeighbors():
                    j = neighbor.GetIdx()
                    bond = mol.GetBondBetweenAtoms(i, j)
                    bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                    adjacency[bond_type_idx, [i, j], [j, i]] = 1
                    if bond_type_idx == 3:
                        k += 1
                #h_u = _get_property_prediction_node_feature(atom, self.atom_type_list)
                h_u = torch.LongTensor(atom_to_feature_vector(atom))
                x.append(h_u)
            #print(k)
            x = np.stack(x, axis=0)
            x = torch.LongTensor(x)    

            adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
            #adjacency = torch.LongTensor(adjacency)
            #node_types[np.where(np.sum(node_types, axis=1) == 0)[0], -1] = 1
            #node_types = torch.LongTensor(node_types)
            
            ih = 0
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                
                edge_index.append(np.expand_dims(np.array([u, v]), axis=0))
                edge_index.append(np.expand_dims(np.array([v, u]), axis=0))
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                #b_u.append(bond_type_idx)
                edge_attr.append(bond_type_idx)
                edge_attr.append(bond_type_idx)

                ih += 1
            #print(ih)
            edge_index = np.concatenate(edge_index, axis=0)
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            edge_attr = np.stack(edge_attr, axis=0)
            edge_attr = torch.LongTensor(edge_attr)
            
            label = self.graph_labels[mol_id]
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label, adjacency=adjacency, node_types=node_types)

            data_list.append(data)
            #self.adjacency.append(torch.FloatTensor(adjacency))
            #self.node_types.append(torch.FloatTensor(atom_types))



        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == "__main__":
    BBBP = BBBPDataset()
    from utils import split_dataset
    from torch_geometric.loader import DataLoader
    args= {}
    args['split_ratio'] = '0.8,0.1,0.1'
    args['split'] = 'scaffold'
    train_set, val_set, test_set = split_dataset(args, BBBP)
    print(len(train_set))
    train_num_nodes_list = []
    train_labels = []
    # for i in range(len(train_set)):
    #     g = train_set[i]
    #     print(g.adjacency.shape)
    #     num_nodes = g.number_of_nodes()
    #     train_num_nodes_list.append(num_nodes)
    #     train_labels.append(labels.item())
    
    loader = DataLoader(train_set, batch_size=2, shuffle=True)
    
    for batch in loader:
        print(batch.edge_index.shape)
        print(batch.edge_attr.shape)
        print(batch.x.shape)
        print(batch.adjacency.shape)
        
        break
    
    # print('training set, 0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(train_num_nodes_list, 0.95), \
    #                                                                                     np.quantile(train_num_nodes_list, 0.9), \
    #       np.quantile(train_num_nodes_list, 0.8)))
    
    # print("max number of atoms: {}, mean of atoms: {}, std of atoms: {}".format(max(train_num_nodes_list), np.mean(train_num_nodes_list), np.std(train_num_nodes_list)))
    # print('number of label 0: {}, number of label 1: {}'.format(len([x for x in train_labels if x==0 ]), len([x for x in train_labels if x==1 ])))
    
    # print('########################################')

    # val_num_nodes_list = []
    # val_labels = []
    # for i in range(len(val_set)):
    #     g, labels, adjacencies, node_types = val_set[i]
    #     num_nodes = g.number_of_nodes()
    #     val_num_nodes_list.append(num_nodes)
    #     val_labels.append(labels.item())
        
    # print('val set, 0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(val_num_nodes_list, 0.95), \
    #                                                                             np.quantile(val_num_nodes_list, 0.9), \
    # np.quantile(val_num_nodes_list, 0.8)))  
        
    # print("max number of atoms: {}, mean of atoms: {}, std of atoms: {}".format(max(val_num_nodes_list), np.mean(val_num_nodes_list), np.std(val_num_nodes_list)))
    # print('number of label 0: {}, number of label 1: {}'.format(len([x for x in val_labels if x==0 ]), len([x for x in val_labels if x==1 ])))

    # print('########################################')
    # test_num_nodes_list = []
    # test_labels = []
    # for i in range(len(test_set)):
    #     g, labels, adjacencies, node_types = test_set[i]
    #     num_nodes = g.number_of_nodes()
    #     test_num_nodes_list.append(num_nodes)    
    #     test_labels.append(labels.item())
    # print('test set, 0.95 quantille: {}, 0.9 quantile: {}, 0.8 quantile: {}'.format(np.quantile(test_num_nodes_list, 0.95), \
    #                                                                             np.quantile(test_num_nodes_list, 0.9), \
    # np.quantile(test_num_nodes_list, 0.8)))
    # print("max number of atoms: {}, mean of atoms: {}, std of atoms: {}".format(max(test_num_nodes_list), np.mean(test_num_nodes_list), np.std(test_num_nodes_list)))
    # print('number of label 0: {}, number of label 1: {}'.format(len([x for x in test_labels if x==0 ]), len([x for x in test_labels if x==1])))
    # print(test_labels)