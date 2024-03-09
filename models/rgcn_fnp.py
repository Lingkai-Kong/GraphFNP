from turtle import forward
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_mean_pool 
import torch
from torch import nn
from torch.nn import functional as F
from features import get_atom_feature_dims
#from features import get_atom_feature_dims
from .utils_fnp import *
from torch_geometric.utils import degree

full_atom_feature_dims = get_atom_feature_dims()

def node_flags(adj, eps=1e-5):
    #adj is in shape of B x C x N x N
    flags = torch.abs(adj[:,:-1,:,:]).sum(1).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags

class AtomEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class Rationle(nn.Module):
    def __init__(self, num_pos_rationale, num_neg_rationale, embedding_size=256):
        super(Rationle, self).__init__()
        
        self.register_parameter('rationale_embedding', nn.Parameter(torch.randn(num_pos_rationale+num_neg_rationale, embedding_size),
                                                requires_grad=True))
        neg_rationale_label =  torch.full((num_neg_rationale,), 0, dtype=torch.long)
        pos_rationale_label = torch.full((num_neg_rationale,), 1, dtype=torch.long)
        self.register_buffer('rationale_label', torch.cat((neg_rationale_label, pos_rationale_label)))
        
    def forward(self):
        return self.rationale_embedding, self.rationale_label
    
    def get_rationale(self):
        return self.rationale_embedding, self.rationale_label


class FNP_RGCNPredictor(nn.Module):
    def __init__(self, num_relations, use_plus, node_feature_dim, dim_y=2, fb_z=0, gcn_hidden_dim=[256,256,256,256], linear_hidden_dim=[256], output_dim=1, drop_ratio=0.5, graph_pooling='mean', label_weight=None, max_num_nodes=0, bond_dim=4, atom_dim=5, lambda_generation=0.1, lambda_vi=0.1):
        super(FNP_RGCNPredictor, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.bond_dim = bond_dim
        self.atom_dim = atom_dim
        self.lambda_generation = lambda_generation
        self.lambda_vi = lambda_vi
        if num_relations == 1:
            #self.node_embedding = nn.Embedding(node_feature_dim.long(), gcn_hidden_dim[0])
            self.node_embedding = nn.Linear(node_feature_dim, gcn_hidden_dim[0])
        else:
            self.node_embedding = AtomEncoder(gcn_hidden_dim[0])
        self.gcn_layers_dim = [gcn_hidden_dim[0]] + gcn_hidden_dim
        self.gcn_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gcn_layers_dim[:-1], self.gcn_layers_dim[1:])):
            #self.gcn_layers.append(RGCNConv(in_dim, out_dim, num_relations))
            self.gcn_layers.append(RGCNConv(in_dim, out_dim, 4))
        if graph_pooling == 'mean':
            self.pooling = global_mean_pool
            
        
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(p=drop_ratio)
        # if graph_pooling == 'set2set':
        #     self.linear_layers_dim = [gcn_hidden_dim[-1]*2] + linear_hidden_dim + [output_dim]
        # else:
        #     self.linear_layers_dim = [gcn_hidden_dim[-1]] + linear_hidden_dim + [output_dim]
        
        # self.linear_layers = nn.ModuleList()
        # for layer_idx, (in_dim, out_dim) in enumerate(zip(self.linear_layers_dim[:-1], self.linear_layers_dim[1:])):
        #     self.linear_layers.append(nn.Linear(in_dim, out_dim))
    
        self.dim_u = gcn_hidden_dim[-1]
        self.dim_z = gcn_hidden_dim[-1]
        self.dim_y = dim_y
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.register_buffer('lambda_z', float_tensor(1).fill_(1e-8))  
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)
        self.register_parameter('pairwise_g_logscale', nn.Parameter(float_tensor(1).fill_(math.log(math.sqrt(self.dim_u))), requires_grad=True))
        self.pairwise_g = lambda x: logitexp(-.5 * torch.sum(torch.pow(x[:, self.dim_u:] - x[:, 0:self.dim_u], 2), 1,
                                                             keepdim=True) / self.pairwise_g_logscale.exp()).view(x.size(0), 1)
        self.p_u = nn.Sequential(nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], 2 * self.dim_u))
        #self.p_u = nn.Sequential(nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], gcn_hidden_dim[-1]), nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], 2 * self.dim_u))
        # q(z|x)
        self.q_z = nn.Sequential(nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], 2 * self.dim_z))
        #self.q_z = nn.Sequential(nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], gcn_hidden_dim[-1]), nn.ReLU(), nn.Linear(gcn_hidden_dim[-1], 2 * self.dim_z))
        
        self.trans_cond_y = nn.Linear(self.dim_y, 2 * self.dim_z)
        #self.trans_cond_y = nn.Sequential(nn.Linear(self.dim_y, 2 * self.dim_z), nn.ReLU(), nn.Linear(2 * self.dim_z, 2 * self.dim_z))
        self.linear_layers_dim = [self.dim_z if not self.use_plus else self.dim_z + self.dim_u] + linear_hidden_dim + [self.dim_y]
        num_linear_layers = len(self.linear_layers_dim) - 1
        self.output = []
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.linear_layers_dim[:-1], self.linear_layers_dim[1:])):
            if layer_idx == num_linear_layers - 1:
                self.output.append(nn.Linear(in_dim, out_dim))
            else:
                self.output.append(nn.Linear(in_dim, out_dim))
                self.output.append(nn.ReLU())
                self.output.append(nn.Dropout(p=drop_ratio))
        self.output = nn.Sequential(*self.output)
    
        # self.output = nn.Sequential(nn.ReLU(),
        #                              nn.Linear(self.dim_z if not self.use_plus else self.dim_z + self.dim_u, 256), nn.ReLu(),
        #                              nn.Linear(self.dim_z if not self.use_plus else self.dim_z + self.dim_u, 256))
        
        #self.output = nn.Sequential(nn.ReLU(), nn.Linear(self.dim_u, self.dim_y))
        self.label_weight = label_weight
        self.pred_adj = nn.Sequential(nn.ReLU(), nn.Linear(self.dim_u, self.max_num_nodes*self.max_num_nodes*self.bond_dim))
        self.pred_x = nn.Sequential(nn.ReLU(), nn.Linear(self.dim_u, self.max_num_nodes*self.atom_dim))
        #self.register_parameter('rationle_embedding', nn.Parameter(torch.randn(num_pos_rationale+num_neg_rationale, gcn_hidden_dim[-1]), requires_grad=True))

    def cond_trans(self, x, edge_index, edge_type, batch=None):
        
        features = self.node_embedding(x)
        for layer_idx, layer in enumerate(self.gcn_layers):
            features = layer(features, edge_index, edge_type)
            if layer_idx == len(self.gcn_layers)-1:
                pass
            else:
                features = self.activation(features)
                if self.training:
                    features = self.dropout(features)
        features = self.pooling(features, batch)
        
        return features
    
    def sample_rationale_graphs(self, rationale_embedding):
        
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(rationale_embedding), self.dim_u, dim=1)
        
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()
        
        generated_adj = self.pred_adj(u) 
        generated_adj = generated_adj.view(-1,  self.max_num_nodes, self.max_num_nodes, self.bond_dim)
        generated_adj = 0.5*(generated_adj + generated_adj.permute(0, 2, 1, 3))
        generated_adj = torch.multinomial(generated_adj, 1)
        generated_adj = generated_adj.view(-1, self.max_num_nodes, self.max_num_nodes)
        generated_adj = F.one_hot(generated_adj, self.bond_dim)
        generated_adj = generated_adj.view(-1, self.max_num_nodes, self.max_num_nodes, self.bond_dim)
        generated_adj = torch.tril(generated_adj, diagonal=-1)
        generated_adj = generated_adj + generated_adj.permute(0, 1, 3, 2)
        
        generated_x = self.pred_x(u) # B * （N * A)
        generated_x =  generated_x.view(-1, self.atom_dim)
        generated_x = torch.multinomial(generated_x, 1)
        generated_x = F.one_hot(generated_x, self.atom_dim)
        generated_x = generated_x.view(-1, self.max_num_nodes, self.atom_dim)
        return generated_x, generated_adj 
    
    def forward(self, H_R, yR, batch_data, lambda_generation, lambda_vi, lambda_reg, mode= 'E', freeze=False):
        '''
        #XR, mean and logvar of the rationale embeddings
        XR, rationale embedding
        yR, labels of the rationale embeddings
        batch_data, PyG graphs
        
        '''
        #pu_mean_R, pu_logscale_R = XR['mean'], XR['logvar']
        #x, edge_index, edge_type, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        yM = batch_data.y
        
        
        if batch_data.edge_attr is None:
            edge_type = torch.zeros(batch_data.edge_index.shape[1], dtype=torch.long).to(batch_data.edge_index.device)
        else:
            edge_type = batch_data.edge_attr
        if batch_data.x is None:
            x = degree(batch_data.edge_index[0], batch_data.num_nodes).long()
        else:
            x = batch_data.x
        edge_index, batch = batch_data.edge_index, batch_data.batch
        yM = batch_data.y
        #print(len(batch_data.adjacency))
        #print(batch_data.adjacency[0].shape)

        
        H_M = self.cond_trans(x, edge_index, edge_type, batch) 
        num_M = H_M.size(0)
        num_R = H_R.size(0)
        
        H_all = torch.cat([H_R, H_M], dim=0)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()
        #u= pu_mean_all
        # if mode == 'M':
        # ## one-shot generation
        #     adj = np.stack(batch_data.adjacency, axis=0)
        #     adj = torch.LongTensor(adj).to(x.device)
        #     #print(adj.shape)
        #     node_types = np.stack(batch_data.node_types, axis=0)
        #     node_types = torch.LongTensor(node_types).to(x.device)
        #     node_types = node_types[:, :self.max_num_nodes, :]
 
            
        #     #print(node_types.shape)
        #     #print(adj.shape)
            
        #     node_mask = node_flags(adj) # B * N
        #     num_nodes = torch.sum(node_mask, dim=1) # B
        #     batch_mask = (num_nodes <= self.max_num_nodes)
        #     node_mask = node_mask[batch_mask]
        #     node_mask = node_mask[:, :self.max_num_nodes].contiguous()
        #     #print(node_mask.shape)
        #     bond_dim = adj.shape[1]
        #     atom_dim = node_types.shape[2]
        #     predicted_adj = self.pred_adj(u[num_R:])
        #     ## make the adjacency matrix symmetric
        #     predicted_adj = predicted_adj.view(-1,  self.max_num_nodes, self.max_num_nodes, bond_dim)
        #     predicted_adj = 0.5*(predicted_adj + predicted_adj.permute(0, 2, 1, 3))
        #     predicted_adj = predicted_adj[batch_mask]
        #     #print(predicted_adj.shape)
            

        #     adj_mask = (node_mask.unsqueeze(1).unsqueeze(-1) * node_mask.unsqueeze(1).unsqueeze(-2))
        #     #print(adj_mask.shape)
        #     #compute the loss for the generation
        #     generation_criterion = nn.CrossEntropyLoss(reduction='none')
        #     adj_label = torch.argmax(adj, dim=1) # B * N * N
        #     adj_label = adj_label[:, :self.max_num_nodes, :self.max_num_nodes]
        #     adj_label = adj_label[batch_mask]
        #     #print(adj_label.shape)
        #     adj_label = adj_label.view(-1)
        #     predicted_adj = predicted_adj.view(-1, bond_dim)
        #     generation_loss_adj = generation_criterion(predicted_adj, adj_label)
        #     generation_loss_adj = generation_loss_adj * adj_mask.view(-1)
            
            
            
        #     predicted_node_types = self.pred_x(u[num_R:]) # B * （N * A)
        #     predicted_node_types = predicted_node_types[batch_mask]
        #     predicted_node_types = predicted_node_types.view(-1, atom_dim)
        #     #print(node_types.shape)
        #     node_types_label = torch.argmax(node_types, dim=2) # B * N
        #     node_types_label = node_types_label[batch_mask]
        #     #print(node_types_label.shape)
        #     node_types_label = node_types_label.view(-1)
            
        #     generation_loss_node_types = generation_criterion(predicted_node_types, node_types_label)
        #     generation_loss_node_types = generation_loss_node_types * node_mask.view(-1)
        #     generation_loss = torch.mean(generation_loss_adj) + torch.mean(generation_loss_node_types)
            
        generation_loss = 0



        # get G
        #G = sample_DAG(u[0:XR.size(0)], self.pairwise_g, training=self.training)

        # get A
        A = sample_bipartite(u[num_R:], u[0:num_R], self.pairwise_g, training=self.training)

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1) 
        qz_M = Normal(qz_mean_all[num_R:], qz_logscale_all[num_R:])
        z_M = qz_M.rsample()
        
        cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
        pz_mean_M = torch.mm(self.norm_graph(A), qz_mean_all[0:num_R])
        #pz_mean_M = torch.mm(self.norm_graph(A),  pu_mean_all[:num_R])
        pz_logscale_M = torch.mm(self.norm_graph(A), qz_logscale_all[0:num_R])
        #pz_logscale_M = torch.mm(self.norm_graph(A), pu_logscale_all[:num_R])
        pz_M = Normal(pz_mean_M, pz_logscale_M)
        pqz_M = pz_M.log_prob(z_M) - qz_M.log_prob(z_M)
        
        pz_R = Normal(qz_mean_all[:num_R], qz_logscale_all[:num_R])
        z_R = pz_R.rsample()

        # apply free bits for the latent z
        # if self.fb_z > 0:
        #     log_qpz_M = - torch.sum(pqz_M)

        #     if self.training:
        #         if log_qpz_M.item() > self.fb_z * z_M.size(0) * z_M.size(1) * (1 + 0.05):
        #             self.lambda_z = torch.clamp(self.lambda_z * (1 + 0.1), min=1e-8, max=1.)
        #         elif log_qpz_M.item() < self.fb_z * z_M.size(0) * z_M.size(1):
        #             self.lambda_z = torch.clamp(self.lambda_z * (1 - 0.1), min=1e-8, max=1.)

        #     #log_pqz_R = self.lambda_z * torch.sum(pqz_all[0:num_R.size(0)])
        #     log_pqz_M = self.lambda_z * torch.sum(pqz_M)

        # else:
        #     #log_pqz_R = torch.sum(pqz_all[0:num_R])
        log_pqz_M = torch.sum(pqz_M)

        #final_rep = z_M if not self.use_plus else torch.cat([z_M, u[num_R:]], dim=1)
        final_rep = z_M if not self.use_plus else torch.cat([z_M, u[num_R:]], dim=1)
        logits_M = self.output(final_rep)

        #pyR = Categorical(logits=logits_all[0:num_R])
        #log_pyR = torch.sum(pyR.log_prob(yR))

        # pyM = Categorical(logits=logits_M)
        # if self.label_weight is not None:
        #     log_pyM = torch.sum(2*self.label_weight[yM.long()]*pyM.log_prob(yM))
        # else:
        #     log_pyM = torch.sum(pyM.log_prob(yM))
        # #obj_R = (log_pyR + log_pqz_R) / float(self.num_M)
        # #obj_M = (log_pyM + log_pqz_M) / float(num_M)
        # obj_M = log_pyM/float(num_M)

        loss_criterion = nn.CrossEntropyLoss(weight=self.label_weight, reduction='none')
        #obj_M = -(loss_criterion(logits_M, yM.long().squeeze())).mean()
        log_pyM = -loss_criterion(logits_M, yM.long().squeeze())
        obj_M =  (log_pyM.sum() + lambda_vi*log_pqz_M)/float(num_M)
        #obj = log_pyM.mean()
        obj = obj_M
        prediction_loss = -obj
        

        if mode == 'M':
            #rationale_representation = u[:num_R] if not self.use_plus else torch.cat([u[:num_R], u[:num_R]], dim=1)
            rationale_representation = z_R if not self.use_plus else torch.cat([z_R, u[:num_R]], dim=1)
            logits_R = self.output(rationale_representation)
            rationale_loss = F.cross_entropy(logits_R, yR.long()).mean()

            loss = lambda_generation* generation_loss + prediction_loss + rationale_loss
        elif mode == 'E':
            rationale_representation = z_R if not self.use_plus else torch.cat([z_R, u[:num_R]], dim=1)
            logits_R = self.output(rationale_representation)
            rationale_loss = F.cross_entropy(logits_R, yR.long()).mean()
            
            # rationale_input = torch.cat([pz_mean_M, pz_mean_M], dim=1)
            # rationale_input_representation = self.output(rationale_input)
            # rationale_input_loss = F.cross_entropy(rationale_input_representation, yM.long().squeeze()).mean()
            
            u_R = u[0:num_R] #num_R * dim_u
            u_M = u[num_R:] #num_M * dim_u
            pairwise_distance = torch.cdist(u_R, u_M, p=2.0) #num_R * num_M
            #pairwise_distance = torch.pow(pairwise_distance, 2) #num_R * num_M
            indices = torch.arange(0, num_R/2).unsqueeze(1).expand(-1, num_M).to(yM.device)# num_R/2 * num_M
            #print(yM.shape)
            indices = indices + (1-yM).unsqueeze(0) * num_R/2 # num_R/2 * num_M
            reg = torch.gather(pairwise_distance, 0, indices.long()).mean() # num_R/2 * num_M
            #loss = prediction_loss + rationale_loss + rationale_input_loss
            loss = prediction_loss + rationale_loss - lambda_reg * reg
        return loss
    
    def test_embedding(self, x_new, XR, yR):
        if x_new.edge_attr is None:
            edge_type = torch.zeros(x_new.edge_index.shape[1], dtype=torch.long).to(x_new.edge_index.device)
        else:
            edge_type = x_new.edge_attr
        if x_new.x is None:
            x = degree(x_new.edge_index[0], x_new.num_nodes).long()
        else:
            x = x_new.x
        edge_index, batch = x_new.edge_index, x_new.batch    
        
        num_R = XR.size(0)
        H_R = XR
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 
        num_new = h_new.size(0)
        H_all = torch.cat([H_R, h_new], dim=0)
        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)        
        
        u = pu.rsample()

        A = sample_bipartite(u[num_R:], u[0:num_R], self.pairwise_g, training=False)
        
        return u, A
    
    def get_u(self, x_new):
        if x_new.edge_attr is None:
            edge_type = torch.zeros(x_new.edge_index.shape[1], dtype=torch.long).to(x_new.edge_index.device)
        else:
            edge_type = x_new.edge_attr
        if x_new.x is None:
            x = degree(x_new.edge_index[0], x_new.num_nodes).long()
        else:
            x = x_new.x
        edge_index, batch = x_new.edge_index, x_new.batch    
        
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(h_new), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)        
        
        u = pu.rsample()
        
        return u

    def get_ru(self, XR, yR):
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(XR), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)  
        u = pu.rsample()
        return u
        
    def get_pred_logits(self, x_new, XR, yR, n_samples=100):
        
        if x_new.edge_attr is None:
            edge_type = torch.zeros(x_new.edge_index.shape[1], dtype=torch.long).to(x_new.edge_index.device)
        else:
            edge_type = x_new.edge_attr
        if x_new.x is None:
            x = degree(x_new.edge_index[0], x_new.num_nodes).long()
        else:
            x = x_new.x
        
        
        edge_index, batch = x_new.edge_index, x_new.batch
        
        
        num_R = XR.size(0)
        H_R = XR
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 
        num_new = h_new.size(0)
        #print(h_new.shape)
        H_all = torch.cat([H_R, h_new], dim=0)
        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)

        qz_mean_R, qz_logscale_R = torch.split(self.q_z(H_all[0:num_R]), self.dim_z, 1)

        logits = float_tensor(num_new, self.dim_y, n_samples)
        for i in range(n_samples):
            u = pu.rsample()

            A = sample_bipartite(u[num_R:], u[0:num_R], self.pairwise_g, training=False)

            #cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
            pz_mean_M = torch.mm(self.norm_graph(A),  qz_mean_R)
            #pz_mean_M = torch.mm(self.norm_graph(A), pu_mean_all[:num_R])
            pz_logscale_M = torch.mm(self.norm_graph(A),  qz_logscale_R)
            #pz_logscale_M = torch.mm(self.norm_graph(A), pu_logscale_all[:num_R])
            pz = Normal(pz_mean_M, pz_logscale_M)

            z = pz.rsample()

            #final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)
            final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)

            #logits[:, :, i] = F.log_softmax(self.output(final_rep), 1)
            logits[:, :, i] = F.softmax(self.output(final_rep), dim=1)

        #logits = torch.logsumexp(logits, 2) - math.log(n_samples)
        probs = torch.mean(logits, dim=2)
        return probs

    def predict(self, x_new, XR, yR, n_samples=100):
        logits = self.get_pred_logits(x_new, XR, yR, n_samples=n_samples)
        #probs = F.softmax(logits, )
        #probs = torch.mean(F.softmax(logits, dim=1),dim=2)
        return logits
    
    def tp(self, x_new, XR, yR, n_samples=100):

        x, edge_index, edge_type, batch = x_new.x, x_new.edge_index, x_new.edge_attr, x_new.batch
        num_R = XR.size(0)
        H_R = XR
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 
        num_new = h_new.size(0)
        #print(h_new.shape)
        H_all = torch.cat([H_R, h_new], dim=0)
        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)

        qz_mean_R, qz_logscale_R = torch.split(self.q_z(H_all[0:num_R]), self.dim_z, 1)
        
        logits = float_tensor(n_samples, num_new, self.dim_y)
        for i in range(n_samples):
        
            u = pu.rsample()

            A = sample_bipartite(u[num_R:], u[0:num_R], self.pairwise_g, training=False)

            #cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
            pz_mean_M = torch.mm(self.norm_graph(A),  qz_mean_R)
            #pz_mean_M = torch.mm(self.norm_graph(A), pu_mean_all[:num_R])
            pz_logscale_M = torch.mm(self.norm_graph(A),  qz_logscale_R)
            #pz_logscale_M = torch.mm(self.norm_graph(A), pu_logscale_all[:num_R])
            pz = Normal(pz_mean_M, pz_logscale_M)

            z = pz.rsample()

            #final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)
            final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)

            logits[i, :, :] = self.output(final_rep)
        
        
        return logits


    def tp_train(self, x_new, XR, yR):
    
        x, edge_index, edge_type, batch = x_new.x, x_new.edge_index, x_new.edge_attr, x_new.batch
        num_R = XR.size(0)
        H_R = XR
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 
        num_new = h_new.size(0)
        #print(h_new.shape)
        H_all = torch.cat([H_R, h_new], dim=0)
        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)

        qz_mean_R, qz_logscale_R = torch.split(self.q_z(H_all[0:num_R]), self.dim_z, 1)
        
        
        u = pu.rsample()

        A = sample_bipartite(u[num_R:], u[0:num_R], self.pairwise_g, training=False)

        #cond_y_mean, cond_y_logscale = torch.split(self.trans_cond_y(one_hot(yR, n_classes=self.dim_y)), self.dim_z, 1)
        pz_mean_M = torch.mm(self.norm_graph(A),  qz_mean_R)
        #pz_mean_M = torch.mm(self.norm_graph(A), pu_mean_all[:num_R])
        pz_logscale_M = torch.mm(self.norm_graph(A),  qz_logscale_R)
        #pz_logscale_M = torch.mm(self.norm_graph(A), pu_logscale_all[:num_R])
        pz = Normal(pz_mean_M, pz_logscale_M)

        z = pz.rsample()

        #final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)
        final_rep = z if not self.use_plus else torch.cat([z, u[num_R:]], dim=1)

        logits = self.output(final_rep)
    
        
        return logits


    def rationale_predict(self, x_new, XR, yR, n_samples=1000):

        if x_new.edge_attr is None:
            edge_type = torch.zeros(x_new.edge_index.shape[1], dtype=torch.long).to(x_new.edge_index.device)
        else:
            edge_type = x_new.edge_attr
        if x_new.x is None:
            x = degree(x_new.edge_index[0], x_new.num_nodes).long()
        else:
            x = x_new.x
            
        edge_index, batch = x_new.edge_index, x_new.batch

        num_R = XR.size(0)
        H_R = XR
        h_new = self.cond_trans(x, edge_index, edge_type, batch) 
        num_new = h_new.size(0)
        #print(h_new.shape)
        H_all = torch.cat([H_R, h_new], dim=0)
        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        prediction = torch.zeros(num_new).to(yR.device)
        prediction_cosine = torch.zeros(num_new).to(yR.device)
        prediction_topk = torch.zeros(num_new).to(yR.device)
        prediction_cosine_topk = torch.zeros(num_new).to(yR.device)
        for i in range(n_samples):
            u = pu.rsample()
            u_R = u[0:num_R] #num_R * dim_u
            u_M = u[num_R:] #num_M * dim_u
            pairwise_distance = torch.cdist(u_R, u_M, p=2.0)
            predict_index = torch.argmin(pairwise_distance, dim=0)
            prediction += torch.where(predict_index >= num_R/2, 1, 0)
            predict_index_topk = torch.topk(pairwise_distance, 3, dim=0, largest=False)[1] #3 * num_M
            predict_index_topk = torch.where(predict_index_topk >= num_R/2, 1, 0)#num_M
            #predict_index_topk = torch.mean(predict_index_topk.float(), dim=0)
            #prediction_topk += torch.where(predict_index_topk >= 0.5, 1, 0)
            predict_index_topk = torch.sum(predict_index_topk.float(), dim=0)
            prediction_topk += predict_index_topk 
            
            pairwise_cosine_distance = sim_matrix(u_R, u_M)
            predict_index_cosine = torch.argmax(pairwise_cosine_distance, dim=0)
            prediction_cosine += torch.where(predict_index_cosine >= num_R/2, 1, 0)
            
            predict_index_cosine_topk = torch.topk(pairwise_cosine_distance, 3, dim=0, largest=True)[1] #3 * num_M
            predict_index_cosine_topk = torch.where(predict_index_cosine_topk >= num_R/2, 1, 0)#num_M
            #predict_index_cosine_topk = torch.mean(predict_index_cosine_topk.float(), dim=0)
            #prediction_cosine_topk += torch.where(predict_index_cosine_topk >= 0.5, 1, 0)        
            predict_index_cosine_topk = torch.sum(predict_index_cosine_topk.float(), dim=0)
            prediction_cosine_topk += predict_index_cosine_topk.float()
            
        prediction = prediction.float()/n_samples
        prediction = torch.where(prediction >= 0.5, 1, 0)
        
        prediction_topk = prediction_topk.float()/(3*n_samples)
        prediction_topk = torch.where(prediction_topk >= 0.5, 1, 0)    
        
        prediction_cosine = prediction_cosine.float()/(n_samples)
        prediction_cosine = torch.where(prediction_cosine >= 0.5, 1, 0)
        
        prediction_cosine_topk = prediction_cosine_topk.float()/(3*n_samples)
        prediction_cosine_topk = torch.where(prediction_cosine_topk >= 0.5, 1, 0)
        
        return prediction, prediction_cosine, prediction_topk, prediction_cosine_topk

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    
    

    model = FNP_RGCNPredictor(num_pos_rationale=5, num_neg_rationale=5, use_plus=True, node_feature_dim=10)
    
    #print(model.modules())
    # for child in model.children():
        
    #      print(child)
        
    for name, module in model.named_parameters():
        print(name)

#print(type(list(model.parameters())[0]))
#print(model.state_dict())
