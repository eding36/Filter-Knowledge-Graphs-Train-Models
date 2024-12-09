# %%
####PLEASE CHECK OUT requirements.rtf for all dependencies and how to run a uv venv before executing this notebook. Fasttext only runs on python3.11.

###Download BioWordVec Model from
"https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin"

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fasttext

model = fasttext.load_model(os.path.dirname(os.getcwd())+"/biowordvec model/BioWordVec_PubMed_MIMICIII_d200.bin")


def generate_phrase_vector(phrase):
    try:
        vector = model[phrase]
    except TypeError:
        vector = np.random.rand(200).tolist()
    return vector

input_file = os.path.dirname(os.getcwd())+"/rkg-text_mining/all_node_ids_mapped_to_node_names.parquet"
df = pd.read_parquet(input_file)

vectors = df['node_name'].apply(generate_phrase_vector)

# Convert the list of vectors into a DataFrame with 200 columns
vectors_df = pd.DataFrame(vectors.tolist(), columns=[f'vector_{i+1}' for i in range(200)])

# Combine the original node_id with the 200-dimensional vector DataFrame
output_df = pd.concat([df['node_id'], vectors_df], axis=1)

# Save the resulting DataFrame to a new Parquet file
output_file = os.path.dirname(os.getcwd())+"/rkg-text_mining/biowordvec_node_embeddings.parquet"
output_df.to_parquet(output_file, index=False)
