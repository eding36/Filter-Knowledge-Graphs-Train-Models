###NEED PYTHON3.6 CONDA ENV FOR THIS SCRIPT

from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator
from sklearn.decomposition import PCA


from tensorflow import keras

import pandas as pd
import os
import numpy as np

###PARAMS
feature_dim = 50 ##length of graphsage embedding
layer_sizes = [feature_dim, feature_dim] ###output should be same as feature_dim, length of this list should be max # of hops per node
dropout = 0.0 ###graphsage model training dropout rate

batch_size = 50 ###num node per mini batch
num_samples = [10, 5] #number of n hop neighbors, n = len(num_samples)

edges_df = pd.read_csv(os.path.dirname(os.getcwd())+ "/rkg_baseline/baseline_kg/robokop_baseline_kg.csv", usecols = ['node1_id', 'node2_id'])
nodes_df = pd.read_parquet(os.path.dirname(os.getcwd())+'/rkg_baseline/biowordvec_node_embeddings.parquet')
nodes_df['node_id'] = nodes_df['node_id'].str.replace('"', '', regex=False)
nodes_df = nodes_df.drop_duplicates(subset= 'node_id')


col1 = nodes_df.iloc[:, 0]  # First column (node_id)
embedding_features = nodes_df.iloc[:, 1:]  # Remaining columns (float values)

# Perform PCA to reduce to 50 dimensions
pca = PCA(n_components=feature_dim)
reduced_features = pca.fit_transform(embedding_features)

# Combine node_id with the reduced features
nodes_pca_df = pd.DataFrame(reduced_features, columns=[f"feature_{i+1}" for i in range(50)])
nodes_pca_df.insert(0, "node_id", col1)

nodes_pca_df.set_index(nodes_pca_df.columns[0], inplace=True)

print(nodes_pca_df)

edges_df = edges_df.drop_duplicates()
edges_df.columns = ['source', 'target']

robokop_stellargraph = StellarGraph(nodes = nodes_pca_df, edges = edges_df)

generator = GraphSAGELinkGenerator(robokop_stellargraph, batch_size, num_samples)

graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=dropout, normalize="l2"
)

x_inp, x_out = graphsage.in_out_tensors()

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]

embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)


node_ids = nodes_pca_df.index.tolist()
node_id_batches= np.array_split(node_ids, 3)
node_id_batches=node_id_batches[2]

print('first 5 node ids:', node_ids[:5])
print(len(node_id_batches))

columns = ['node_id'] + [f'feat{i+1}' for i in range(feature_dim)]

node_gen = GraphSAGENodeGenerator(robokop_stellargraph, batch_size, num_samples).flow(node_id_batches)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
node_ids = np.array(node_id_batches).reshape(-1, 1)
id2feat = np.hstack([node_ids, node_embeddings])
df = pd.DataFrame(id2feat, columns=columns)
print(df)
df.to_parquet(os.path.dirname(os.getcwd())+'/rkg_baseline/graphsage_batch/node_graphsage_embeddings_batch_3.parquet')


parquet_files = [os.path.dirname(os.getcwd())+ '/rkg_baseline/graphsage_batch/node_graphsage_embeddings_batch_1.parquet',os.path.dirname(os.getcwd())+ '/rkg_baseline/graphsage_batch/node_graphsage_embeddings_batch_2.parquet', os.path.dirname(os.getcwd())+ '/rkg_baseline/graphsage_batch/node_graphsage_embeddings_batch_3.parquet']

dataframes = [pd.read_parquet(file) for file in parquet_files]
concatenated_df = pd.concat(dataframes, axis=0, ignore_index=True)

concatenated_df.to_parquet(os.path.dirname(os.getcwd())+f'/rkg_baseline/graphsage_embeddings_biowordvec_feature_dim_{feature_dim}_layer_sizes_{layer_sizes[0]}_{layer_sizes[1]}_batch_size_{batch_size}_num_samples_{num_samples[0]}_{num_samples[1]}_dropout_{dropout}.parquet')
