# Deep-Learning-for-Molecular-Property-Prediction-in-the-Foundation-Model-Era

<p align="center">
  <img src="Figure\fig1_intro.png" alt="Overview and contributions of this survey. The survey constructs a unified framework that organizes over 100 deep learning methods for molecular property prediction across four axes: evolution, taxonomy, capability, and roadmap. " width="1000"/>
</p>
<p align="center">Overview and contributions of this survey. The survey constructs a unified framework that organizes over 100 deep learning methods for molecular property prediction across four axes: evolution, taxonomy, capability, and roadmap.</p>

## About This Project
This repository is a continuously updated collection of papers and resources dedicated to "A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era".

For in-depth knowledge, check out our survey paper: "A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era".

If you find this project helpful, we kindly ask you to consider citing our work:
```bibtex
@article{li_mpp_foundation,
  title={A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era},
  author={Li, Zongru and Chen, Xingsheng and Wen, Honggang and Zhang, Regina and Li, Ming and Zhang, Xiaojin and Yin, Hongzhi and Yang, Qiang and Lam, Kwok-Yan and Lio, Pietro and Yiu, Siu-Ming},
}
```

<p align="center">
  <img src="Figure\fig5_pipeline.png" alt="The pipeline of deep learning-driven MPP" width="1000"/>
</p>
<p align="center">The pipeline of deep learning-driven MPP</p>

## Table of Contents

- [Representation Modalities](#representation-modalities)
  - [1D Representations](#1d-representations)
    - [SMILES-based Models](#smiles-based-models)
    - [SELFIES-based Models](#selfies-based-models)
    - [Other Sequence Representations](#other-sequence-representations)
  - [Molecule Topological Graph (2D)](#molecule-topological-graph-2d)
  - [Geometric Conformation (3D)](#geometric-conformation-3d)
  - [Multimodal Representations](#multimodal-representations)
- [Model Architectures](#model-architectures)
  - [Geometric GNNs](#geometric-gnns)
  - [Graph Transformers](#graph-transformers)
  - [Hybrid Architectures](#hybrid-architectures)
  - [Quantum Hybrid Models](#quantum-hybrid-models)
- [Applications](#applications)
  - [Drug Discovery](#drug-discovery)
  - [Materials Design](#materials-design)
  - [Other Applications](#other-applications)

------

## Representation Modalities

<p align="center">
  <img src="Figure\fig2_representation.png" alt="The representations of molecules" width="1000"/>
</p>
<p align="center">The representations of molecules</p>

### 1D Representations

#### SMILES-based Models

- SimSon: Simple contrastive learning of SMILES for molecular property prediction
- ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction 
- Molecular representation learning with language models and domain-relevant auxiliary tasks 
- Transformers for molecular property prediction: Domain adaptation efficiently improves performance 
- Convolutional neural network based on SMILES representation of compounds for detecting chemical motif 
- DeepSMILES: An Adaptation of SMILES for Use in Machine-Learning of Chemical Structures 
- SMILES Pair Encoding: A Data-Driven Substructure Tokenization Algorithm for Deep Learning 
- SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction 
- SPVec: A Word2vec-Inspired Feature Representation Method for Drug-Target Interaction Prediction 
- Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition 
- Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources 
- DeepDTA: deep drug–target binding affinity prediction 
- Chemberta-2: Towards chemical foundation models 
- Mol-BERT: An Effective Molecular Representation with BERT for Molecular Property Prediction 
- Chemformer: A Pre-Trained Transformer for Computational Chemistry 
- Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction 
- SMILES2Vec: An Interpretable General-Purpose Deep Neural Network for Predicting Chemical Properties 
- ReactionT5: a pre-trained transformer model for accurate chemical reaction prediction with limited data 
- Chemical representation learning for toxicity prediction 
- MolTrans: Molecular Interaction Transformer for drug–target interaction prediction 

#### SELFIES-based Models

- Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources 
- Group SELFIES: a robust fragment-based molecular string representation 
- SELFormer: Molecular representation learning via SELFIES language models 
- Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation 

#### Other Sequence Representations

- Distributed Representations of Words and Phrases and their Compositionality 
- InChI, the IUPAC International Chemical Identifier 
- DeepTox: Toxicity Prediction using Deep Learning 

------

### Molecule Topological Graph (2D)

- Graph attention networks 
- Do Transformers Really Perform Badly for Graph Representation? 
- MolE: a foundation model for molecular graphs using disentangled attention 
- Language models can explain neurons in language models 
- GNN-SKAN: Advancing Molecular Representation Learning with SwallowKAN 
- Semi-supervised classification with graph convolutional networks 
- SPECTRA: Spectral Target-Aware Graph Augmentation for Imbalanced Molecular Property Regression 
- Recipe for a general, powerful, scalable graph transformer 
- GraphMAE: Self-Supervised Masked Graph Autoencoders 
- A compact review of molecular property prediction with graph neural networks 
- N-gram graph: Simple unsupervised representation for graphs, with applications to molecules 
- Neural message passing for Quantum chemistry 
- Chemical Graph-Based Transformer Models for Yield Prediction of High Throughput Cross-Coupling Reaction Datasets 
- Self-supervised graph transformer on large-scale molecular data 
- How powerful are graph neural networks? 
- Kagnns: Kolmogorov-arnold networks meet graph learning 
- Graphkan: Graph kolmogorov arnold network for small molecule-protein interaction predictions 

------

### Geometric Conformation (3D)

- Exploring chemical compound space with quantum-based machine learning 
- Directional Message Passing for Molecular Graphs 
- SchNet: a continuous-filter convolutional neural network for modeling quantum interactions 
- DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking 
- Spherical Message Passing for 3D Molecular Graphs 
- Highly accurate quantum chemical property prediction with uni-mol+ 
- Fast and uncertainty-aware directional message passing for non-equilibrium molecules 
- Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds 
- Torchmd-net: equivariant transformers for neural network based molecular potentials 
- Benchmarking graphormer on large-scale molecular modeling datasets 
- Gemnet: Universal directional graph neural networks for molecules 
- Gps++: An optimised hybrid mpnn/transformer for molecular property prediction 
- E(n) Equivariant Graph Neural Networks 
- Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations 
- Molecular geometry-aware transformer for accurate 3d atomic system modeling 
- Molecule attention transformer 
- Geometry-enhanced molecular representation learning for property prediction 

------

### Multimodal Representations

- Pre-training molecular graph representation with 3d geometry 
- Graph-BERT and language model-based framework for protein--protein interaction identification 
- Holo-Mol: An explainable hybrid deep learning framework for predicting reactivity of hydroxyl radical to water contaminants based on holographic fused molecular representations 
- GraSeq: graph and sequence fusion learning for molecular property prediction 

------

## Model Architectures

### Geometric GNNs

<p align="center">
  <img src="Figure\fig3_GNN.png" alt="Geometric GNN in MPP" width="1000"/>
</p>
<p align="center">Geometric GNN in MPP</p>

- SchNet: a continuous-filter convolutional neural network for modeling quantum interactions 
- E(n) Equivariant Graph Neural Networks 
- Gemnet: Universal directional graph neural networks for molecules 
- Se (3)-transformers: 3d roto-translation equivariant attention networks 
- Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations 
- Spherical Message Passing for 3D Molecular Graphs 
- Geometry-enhanced molecular representation learning for property prediction 
- Molecular geometry-aware transformer for accurate 3d atomic system modeling 
- Fast and uncertainty-aware directional message passing for non-equilibrium molecules 
- DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking 
- Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds 

------

### Graph Transformers

<p align="center">
  <img src="Figure\fig4_Transformer.png" alt="Transformer in MPP" width="1000"/>
</p>
<p align="center">Transformer in MPP</p>

- MolE: a foundation model for molecular graphs using disentangled attention 
- SimSon: Simple contrastive learning of SMILES for molecular property prediction 
- Do Transformers Really Perform Badly for Graph Representation? 
- Benchmarking graphormer on large-scale molecular modeling datasets 
- Molecular geometry-aware transformer for accurate 3d atomic system modeling 
- Torchmd-net: equivariant transformers for neural network based molecular potentials 
- Molecule attention transformer 
- Chemical Graph-Based Transformer Models for Yield Prediction of High-Throughput Cross-Coupling Reaction Datasets 
- Recipe for a general, powerful, scalable graph transformer 
- Directed message passing based on attention for prediction of molecular properties 
- Self-supervised graph transformer on large-scale molecular data 

------

### Hybrid Architectures

- Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models 
- Gps++: An optimised hybrid mpnn/transformer for molecular property prediction 
- Pre-training molecular graph representation with 3d geometry 
- GraSeq: graph and sequence fusion learning for molecular property prediction 
- Kagnns: Kolmogorov-arnold networks meet graph learning 
- GNN-SKAN: Advancing Molecular Representation Learning with SwallowKAN 
- Graph-BERT and language model-based framework for protein--protein interaction identification 
- Graphkan: Graph kolmogorov arnold network for small molecule-protein interaction predictions 
- Holo-Mol: An explainable hybrid deep learning framework for predicting reactivity of hydroxyl radical to water contaminants based on holographic fused molecular representations 

------

### Quantum Hybrid Models

- NeuroQ: Quantum-Inspired Brain Emulation 
- Differentiable quantum computational chemistry with PennyLane 
- Generalizing neural wave functions 

------

## Applications

### Drug Discovery

- ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction 
- Molecular representation learning with language models and domain-relevant auxiliary tasks 
- Chemberta-2: Towards chemical foundation models 
- DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking 
- E(n) Equivariant Graph Neural Networks 
- GraphDTA: predicting drug–target binding affinity with graph neural networks 
- SELFormer: Molecular representation learning via SELFIES language models 
- MolE: a foundation model for molecular graphs using disentangled attention 
- Neural message passing for Quantum chemistry 
- Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources 
- MolTrans: Molecular Interaction Transformer for drug–target interaction prediction 
- ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries 
- Fast and uncertainty-aware directional message passing for non-equilibrium molecules 
- Molecular geometry-aware transformer for accurate 3d atomic system modeling 
- Self-supervised graph transformer on large-scale molecular data 
- SPECTRA: Spectral Target-Aware Graph Augmentation for Imbalanced Molecular Property Regression 
- DeepDTA: deep drug–target binding affinity prediction 
- Graph attention networks 
- Chemprop: A Machine Learning Package for Chemical Property Prediction 
- Fate-tox: fragment attention transformer for E(3)-equivariant multi-organ toxicity prediction 
- Geometry-enhanced molecular representation learning for property prediction 
- Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models 
- Molecular contrastive learning of representations via graph neural networks 

------

### Materials Design

- Exploring chemical compound space with quantum-based machine learning 
- SchNet: a continuous-filter convolutional neural network for modeling quantum interactions 
- Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations 
- Generalizing neural wave functions 
- PND: Physics-informed neural-network software for molecular dynamics applications 
- Scaling deep learning for materials discovery 
- Catalyst Energy Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models
- CataLM: Empowering Catalyst Design Through Large Language Models 
- Uni-Electrolyte: An Artificial Intelligence Platform for Designing Electrolyte Molecules for Rechargeable Batteries 
- A predictive machine learning force-field framework for liquid electrolyte development 
- Generative Pretrained Transformer for Heterogeneous Catalysts 

------

### Other Applications

- Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds
- Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction 
- Chemformer: A Pre-Trained Transformer for Computational Chemistry 
- Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction 
- Integration of Transfer Learning and Multitask Learning To Predict the Potential of Per/Polyfluoroalkyl Substances in Activating Multiple Nuclear Receptors Associated with Hepatic Lipotoxicity 
- Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction 
- Chemformer: A Pre-Trained Transformer for Computational Chemistry 
- Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction 
- Integration of Transfer Learning and Multitask Learning To Predict the Potential of Per/Polyfluoroalkyl Substances in Activating Multiple Nuclear Receptors Associated with Hepatic Lipotoxicity 
