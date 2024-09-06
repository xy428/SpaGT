import os,csv,re
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from . model import EGT_DEC
from torch.utils.data import DataLoader,Dataset
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.cluster import KMeans
import torch.nn as nn
from . data import mclust_R
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.spatial import distance



class DEC(nn.Module):
    def __init__(self, X, adj, node_width, edge_width, num_heads=5,init="kmeans",tol=1e-4, lr=0.001, epochs=20,opt="sgd",
                weight_decay=5e-7,alpha=0.1,n_clusters=14,trajectory=[],nhid=128,
                max_epochs=200,update_interval=50, trajectory_interval=50,  n_neighbors=10):
        super(DEC, self).__init__()
        
        self.init=init
        self.tol=tol
        self.lr=lr
        self.epochs=epochs
        self.opt=opt
        self.weight_decay=weight_decay
        self.alpha=alpha
        self.n_clusters=n_clusters
        self.trajectory=trajectory
        self.nhid=nhid
        self.max_epochs=max_epochs
        self.update_interval=update_interval
        self.trajectory_interval=trajectory_interval
        self.X = X
        self.adj = adj
        self.n_neighbors = n_neighbors
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        
        self.model = EGT_DEC(self.node_width,self.edge_width,self.num_heads)


    def forward(self, x, adj):
        x = self.model(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q
    
    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss
    
    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    
    def priori_cluster(
        self,
        adata,
        n_domains = 11,
        init = 'leiden'
        ):
        for res in sorted(list(np.arange(0.1, 2.0, 0.01)), reverse=True):
            if init == 'leiden':
                sc.tl.leiden(adata, random_state=0, resolution=res)
                count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
                if count_unique_leiden == n_domains:
                    break
            elif init == 'louvain':
                sc.tl.louvain(adata, random_state=0, resolution=res)
                count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
                if count_unique_louvain == n_domains:
                    break
        print("Best resolution: ", res)
        return res
    
    
    def fit(self, X,adj, init="mclust",num_cluster=5, n_neighbors=30,res=0.05, pca_num=25):
        if self.opt=="sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt=="admin":
            self.optimizer = optim.Adam(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        features = self.model(X,adj)
        
        # pca = PCA(n_components=pca_num, random_state=42) 
        # embedding = pca.fit_transform(features.detach().numpy())
        
        # return features
        adata=sc.AnnData(features.detach().numpy())
        # matrix = adata.X 
        # if isinstance(matrix,sparse.spmatrix):
        #     matrix = matrix.toarray()
        adata.obsm['emb'] = adata.X
        
        if init=="louvain":
            # sc.pp.neighbors(adata, n_neighbors)
            sc.pp.neighbors(adata)
            # res = self.priori_cluster(adata,num_cluster,init)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        elif init == "leiden":
            # sc.pp.neighbors(adata, n_neighbors)
            sc.pp.neighbors(adata)
            # res = self.priori_cluster(adata,num_cluster,init)
            sc.tl.leiden(adata, resolution=res)
            y_pred = adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        elif init=="mclust":
            sc.pp.neighbors(adata, n_neighbors)
            # sc.tl.umap(adata)
            y_pred = mclust_R(adata, num_cluster, pca_num)
            # y_pred=adata.obs['mclust'].astype(int).to_numpy()
        elif init=="kmeans":
            sc.pp.neighbors(adata, n_neighbors)
            kmeans = KMeans(num_cluster, n_init=20,random_state=42)
            y_pred = kmeans.fit_predict(embedding)
        y_pred_last = y_pred        
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(self.epochs):
            if epoch%self.update_interval == 0:
                _, q = self.forward(X,adj)
                p = self.target_distribution(q).data
            # if epoch==0:
            #     _, q = self.forward(X,adj)
            #     p = self.target_distribution(q).data
            
            self.optimizer.zero_grad()
            z,q = self.forward(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            self.optimizer.step()
            
            if epoch%self.trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%self.update_interval == 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
            torch.cuda.empty_cache()
            
    def predict(self):
        z,q = self.forward(torch.FloatTensor(self.X),torch.FloatTensor(self.adj))
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        prob=q.detach().numpy()
        return y_pred, prob,z


