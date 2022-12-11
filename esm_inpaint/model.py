import matplotlib.pyplot as plt
import torch
import esm
from typing import List, Optional, Tuple

# A number of functions/classes are adopted from: https://github.com/facebookresearch/esm


def esm_2():
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask",
         "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3",  "K A <mask> I S Q"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1: tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()


def esmfold_inpaint(sequence: str):
    model.eval()  # disables dropout for deterministic results
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    with torch.no_grad():
        output = model.infer_pdb(sequence)
