import esm
import numpy as np
import torch

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic resul
model = model.to("cuda") # model to GPU
def extract_seq(seq,output_file_path,block_layer=33):
    """
    Input
        seq : str of sequences or list of sequences
        output_file_path : pkl file for embedding
        block_layer
        - 33 for esm2_t33_650M_UR50D
        - 36 for esm2_t36_3B_UR50D
        - 48 for esm2_t48_15B_UR50D
    Output : {'esm2_embedding':[tensor_array]}
    
    """
    if isinstance(seq,str):
        data = [("protein1",seq)]
    elif isinstance(seq,list):
        data = [("protein"+str(i),seq[i]) for i in range(len(seq))]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to("cuda")
    # Extract per-residue representations (on GPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[block_layer], return_contacts=True)
    token_representations = results["representations"][block_layer]
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0)) #[Length, dim]
    np.save(output_file_path,{"esm2_embedding":sequence_representations},allow_pickle=True)


if __name__ == "__main__":
    seq = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
    output_file_path = "./test"
    extract_seq(seq,output_file_path)

    