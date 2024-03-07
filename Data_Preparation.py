import pandas as pd
import numpy as np
import torch
import esm
# import psycopg2 as ps
from IPython.display import HTML

def Check_Mut(sc_seq): # Shortcut sequence ---> check a number of Mutation
    n_mut = []
    count = 0
    for index, alph in enumerate(sc_seq): # We try to split each mutation by sign ';'
        if alph == ';':
            n_mut.append(sc_seq[count:index])
            count = index + 1
    
    n_mut.append(sc_seq[count:len(sc_seq)])
    return n_mut

def Transformation(dataset, sequence): # Transform shortcut mutation to fully alphabet sequence
    df = pd.DataFrame(dataset)
    signs = df.iloc[:,0]
    f_scores = df.iloc[:,1]
    Sequence_df = []
    
    for index, sign in enumerate(signs): # Sequences in DataFrame
        
        Mutation = Check_Mut(sign)
        Mut_seq = sequence
        for pos in Mutation:
            index = int(pos[1:-1])
            Mut_seq = Mut_seq[:index-1] + pos[-1] + Mut_seq[index:]
        
        Sequence_df.append(Mut_seq)
    
    df.iloc[:,0] = Sequence_df # Substitude shortcut seq by full sequence
    return df

# Encoding
def Encode(Data_Trans):

    data = [['ID'+str(i+1), seq] for i, seq in enumerate(Data_Trans)] # Adjust and Rearrange data corresponding to model

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # Claim a model from ESM2
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad(): #less compute
        results = model(batch_tokens, repr_layers=[33])  # Specify the layer(s) from which to get embeddings
    embeddings = results["representations"][33]
    
    
    return embeddings
