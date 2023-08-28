import numpy as np
import dgl
import torch
import networkx as nx
from scipy.sparse import csc_matrix


def multigraph_to_dgl(graph, n_feats=None, rel_feats=None):
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    if rel_feats is not None:
    # Add edges
      for rel, adj in enumerate(graph):
          # Convert adjacency matrix to tuples for nx0
          nx_triplets = []
          for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
              nx_triplets.append((src, dst, {'type': rel, 'emb': rel_feats[rel, :]}))
          g_nx.add_edges_from(nx_triplets)

      # make dgl graph
      #g_dgl = dgl.DGLGraph(multigraph=True)
      g_dgl=dgl.from_networkx(g_nx, edge_attrs=['type', 'emb']) 
    else:
      for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
          nx_triplets.append((src, dst, {'type':rel}))
          g_nx.add_edges_from(nx_triplets)
      g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    if n_feats is not None:
      g_dgl.ndata['feat'] = torch.tensor(n_feats)
    
    return g_dgl


def construct_triplet(kg_file, dataset_name, ent2id, rel2id, node_feature, rel_feature, add_transpose_rel):
  kg_file = f'data/{dataset_name}/kg/'+kg_file
  triplet = []
  with open(kg_file) as f:
    for line in f.readlines():
      sub, rel, obj, _, _ = line.strip().split("\t")
      triplet.append([ent2id[sub], rel2id[rel], ent2id[obj]])
  
  tripts = np.array(triplet)
  adj_list = []
  for i in range(len(rel2id)):
      idx =  np.argwhere(tripts[:, 1]==i)
  adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (tripts[:, 0][idx].squeeze(1), \
                    tripts[:, 2][idx].squeeze(1))), shape=(len(ent2id), len(ent2id))))
  if add_transpose_rel:
      adj_list_t = [adj.T for adj in adj_list]
      adj_list += adj_list_t
  #aug_num_rels = len(adj_list)
  graph = multigraph_to_dgl(adj_list, node_feature, rel_feature)
  return graph