# GPDL (Generative Protein Design by Language-model)

[Protein Language Model Supervised Precise and Efficient Protein Backbone Design Method
](https://www.biorxiv.org/content/10.1101/2023.10.26.564121v1)

![GPDL](./img/img.png)

GPDL is a deep learning method to **design novel and high quality scaffold backbone** given the desired motif residue topologies and sequences. Included in this code repository are two distinct methods, each offering a balance between generation speed and output quality.

## 💻 Environment set-up

### Conda environment

```
# install esmfold and openfold 
conda create -n gpdl python=3.8
conda activate gpdl
conda install pip
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge biotite
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

### Third party source code

Our repo keeps a fork of ProteinMPNN in `./ProteinMPNN`. Our conda environment is sufficient for running the ProteinMPNN codes to generate sequences compatible with our backbones.

```
git clone https://github.com/dauparas/ProteinMPNN.git
```

## 🚀 Inpainting tutorial

- GPDL-Inpainting employs a fine-tuned ESMFold module, specializing in the generation of scaffold proteins tailored to functional sites, a process also known as $s,t \sim f_{\theta}(\hat{s},\hat{t})$ , where  $s,\hat{s},t,\hat{t}$ represent the entire sequence, the complete structures, motif sequences, and motif structures, respectively. Here, $f_{\theta}$ denotes the ESM-Inpainting network along with its parameters.
- During the fine-tuning process, only the structure module and certain linear projection layers (such as the distance linear layer and sequence output embedding) undergo training. Other components, including ESM2 and FoldingTrunk, remain fixed.
- Typically, GPDL-Inpainting outperforms other backbone generation methods in terms of speed. It is capable of generating approximately 10,000 backbones, each 200 amino acids long, within a single day using a single V100 GPU.

Run the bash file `./gpdl_inpaint_sample.sh` in a slrum system or using the following command:

```
python3 ./gpdl_inpainting/esm_inference_v2.py  \
    --input ./gpdl_inpainting/benchmark_set/1BCF.pdb \
    --output_prefix "./gpdl_inpainting/design/1bcf/1BCF" \
    --inpaint_seq "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15" \
    --sample_steps 1 \
    --num_design 100 \
    --fpath "./gpdl_inpainting/design/1bcf/1BCF.txt"
```

The following parameters can be specified in the Python command :

- `input` - This is the target protein data structure that contains motif information.
- `output_prefix` - The format for the backbone output, such as `output_prefix_1.pdb`,`output_prefix_2.pdb`.
- `inpaint_seq` -  This defines the motif information format, like `"x-y,Ax-y,m-n"`, where x,y indicate the length range of scaffold samples needed. For each design ,a random number N is sampled from  $N \sim $ Uniform[x,y]. The motif position begins from A chain residue x to y residue y.
- `sample_steps` - The number of ESMFold cycles in the inpainting program. It is generally recommended to set this to 1 for most tasks.
- `num_design` - The number of designs to be generated.
- `fpath` - The file path for outputting detailed design information, which is useful for computing metrics like RMSD and pLDDT.

## 🔮 Hallucination tutorial

- GPDL-Hallucination utilized the ESMFold to generate backbones prediction and optimize the intermediate iteratively. The optimization involved introducing mutations to a previously accepted sequence and updating the sequence based on pre-defined criteria (motif RMSD and pLDDT, van der waal radius in some cases)

To quickly run 100 design trajectories for scaffolding the 1BCF binding interface using 1500 steps of gradient descent, one can run the bash file `./gpdl_inpaint_sample.sh` in a slrum system or using the following command:

```
python3 ./gpdl_hallucination/hallucination_v1.py \
    --reference /path_of_reference.pdb \
    --output_dir /output_folder \
    --step 1500 \
    --loss 10 \
    --t1 1 \
    --t2 500 \
    --n_mut_max 20 \
    --number 100 \
    --mask_len 10,20,20,20,10 \
    --motif_id A92-99,A123-130,A47-54,A18-25 \
    --atom N,CA,C,O
```

- `reference`: Specifies the path to the template protein structure file.
- `motif_id`: Identifies the residue ranges within the template that should remain fixed during hallucination.
- `mask_len`: Defines the number of residues to be hallucinated within each specified range. For example, using --mask_len 10,20,20,20,10 along with --motif_id A92-99, A123-130, A47-54, A18-25 instructs the script to hallucinate sequences of 10, 20, 20, 20, and 10 residues in length, corresponding to and scaffolding around the specified residue ranges in chain A of the template.
- `t1` and `t2`: These parameters set the initial temperature for simulated annealing (--t1) and the half-life for the temperature decay (--t2), respectively.
- `step`: Determines the number of steps to be executed during the simulated annealing process.
- `atom`: Designates which atoms are to be considered when calculating the RMSD in the loss function.

---

## ✏️ Citation

If you use the framework in your research, please cite the following paper.

```
@article {GPDL,
    Author={Bo, Zhang and Kexin, Liu and Zhuoqi, Zheng and Yunfeiyang, Liu and Ting, Wei and Haifeng, Chen},  
    title={Protein Language Model Supervised Precise and Efficient Protein Backbone Design Method},  
    year={2023},
    doi={10.1101/2023.10.26.564121},
    url={https://www.biorxiv.org/content/10.1101/2023.10.26.564121v1},
    journal={bioRxiv}
}
```

---

Github codebase author : Bo Zhang, Kexin Liu, Zhuoqi Zheng

E-mail : {zhangbo777,lkxlkx,h2knight}@sjtu.edu.cn
