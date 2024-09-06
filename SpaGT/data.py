import pandas as pd
import numpy as np
import torch,ot
import random
import scanpy as sc
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import PCA

def spatial_reconstruction(
        adata: sc.AnnData,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = None,
        normalize_total: bool = False,
        copy: bool = False,
) -> Optional[sc.AnnData]:

    adata = adata.copy() if copy else adata

    adata.layers['counts'] = adata.X

    sc.pp.normalize_total(adata) if normalize_total else None
    # sc.pp.log1p(adata)

    adata.layers['log1p'] = adata.X

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)

    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists
    
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

    adata.X = csr_matrix(X_rec)
    
    ReX = pd.DataFrame(X_rec)
    ReX[ReX<0] = 0
    adata.layers['ReX'] = ReX.values

    # del adata.obsm['X_pca']

    adata.uns['spatial_reconstruction'] = {}

    rec_dict = adata.uns['spatial_reconstruction']

    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total

    return adata, conns

def mclust_R(adata, num_cluster,pca_num=25, modelNames='EEE', random_seed=200):
    
    pca = PCA(n_components=pca_num, random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')

    # adata.obs['mclust'] = mclust_res
    # adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        
    return mclust_res

def refine_label(adata, radius=50, key='pred'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs['label_refined'] = np.array(new_type)
    
    return adata,np.array(new_type)


from sklearn.metrics import accuracy_score
import numpy as np

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    nan_indices = np.where(np.isnan(y_true))[0]
    y_true = np.delete(y_true, nan_indices)
    y_pred = np.delete(y_pred, nan_indices)

    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True