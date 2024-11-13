from stellargraph import StellarGraph
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator

from tensorflow import keras

import pandas as pd
import os

edges_df = pd.read_csv(os.path.dirname(os.getcwd())+ "/enriched_refined_rkg/enriched_refined_robokop_kg.csv", usecols = ['node1_id', 'node2_id'])
nodes_df = pd.read_parquet(os.path.dirname(os.getcwd())+'/biowordvec_node_embeddings.parquet')
nodes_df['node_id'] = nodes_df['node_id'].str.replace('"', '', regex=False)
nodes_df = nodes_df.drop_duplicates(subset= 'node_id')
nodes_df.set_index(nodes_df.columns[0], inplace=True)

print(nodes_df)

edges_df = edges_df.drop_duplicates()
edges_df.columns = ['source', 'target']

robokop_stellargraph = StellarGraph(nodes = nodes_df, edges = edges_df)

nodes = list(robokop_stellargraph.nodes())
number_of_walks = 1
length = 5

batch_size = 50
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(robokop_stellargraph, batch_size, num_samples)

layer_sizes = [200, 200]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
)

x_inp, x_out = graphsage.in_out_tensors()

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = nodes_df.index.tolist()
print('first 5 node ids:', node_ids[:5])
print('len of node ids', len(node_ids))
node_gen = GraphSAGENodeGenerator(robokop_stellargraph, batch_size, num_samples).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

node_embeddings_df = pd.DataFrame(node_embeddings)
node_embeddings_df.columns = [f"col{x}" for x in range(200)]

node_embeddings_df['node_id'] = node_ids
node_embeddings_df.insert(0, 'node_id', node_embeddings_df.pop('node_id'))

node_embeddings_df.to_parquet(os.path.dirname(os.getcwd())+'/node_graphsage_embeddings.parquet')
