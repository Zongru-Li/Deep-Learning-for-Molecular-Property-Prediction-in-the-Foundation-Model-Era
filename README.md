# Deep Learning for Molecular Property Prediction in the Foundation Model Era

<p align="center">
  <img src="Figure\fig1_intro.png" alt="Overview and contributions of this survey. The survey constructs a unified framework that organizes over 100 deep learning methods for molecular property prediction across four axes: evolution, taxonomy, capability, and roadmap. " width="1000"/>
</p>
<p align="center">Overview and contributions of this survey. The survey constructs a unified framework that organizes over 100 deep learning methods for molecular property prediction across four axes: evolution, taxonomy, capability, and roadmap.</p>

## About This Project
This repository is a continuously updated collection of papers and resources dedicated to "A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era".

For in-depth knowledge, check out our survey paper: "A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era".

If you find this project helpful, we kindly ask you to consider citing our work:
```bibtex
@article{lisystematic,
  title={A Systematic Survey and Benchmarks of Deep Learning for Molecular Property Prediction in the Foundation Model Era},
  author={Li, Zongru and Chen, Xingsheng and Wen, Honggang and Zhang, Regina and Li, Ming and Zhang, Xiaojin and Yin, Hongzhi and Yang, Qiang and Lam, Kwok-Yan and Lio, Pietro and others}
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

- [SimSon: Simple contrastive learning of SMILES for molecular property prediction](https://academic.oup.com/bioinformatics/article/41/5/btaf275/8127203)
- [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- [Molecular representation learning with language models and domain-relevant auxiliary tasks](https://arxiv.org/abs/2011.13230)
- [Transformers for molecular property prediction: Domain adaptation efficiently improves performance](https://arxiv.org/abs/2503.03360)
- [Convolutional neural network based on SMILES representation of compounds for detecting chemical motif](https://link.springer.com/article/10.1186/s12859-018-2523-5)
- [DeepSMILES: An Adaptation of SMILES for Use in Machine-Learning of Chemical Structures](https://chemrxiv.org/doi/10.26434/chemrxiv.7097960)
- [SMILES Pair Encoding: A Data-Driven Substructure Tokenization Algorithm for Deep Learning](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01127)
- [SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction](https://dl.acm.org/doi/10.1145/3307339.3342186)
- [SPVec: A Word2vec-Inspired Feature Representation Method for Drug-Target Interaction Prediction](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2019.00895/full)
- [Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616)
- [Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources](https://www.nature.com/articles/s41598-025-05017-w)
- [DeepDTA: deep drug–target binding affinity prediction](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
- [ChemBERTa-2: Towards chemical foundation models](https://arxiv.org/abs/2209.01712)
- [Mol-BERT: An Effective Molecular Representation with BERT for Molecular Property Prediction](https://onlinelibrary.wiley.com/doi/10.1155/2021/7181815)
- [Chemformer: A Pre-Trained Transformer for Computational Chemistry](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b)
- [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](https://arxiv.org/abs/1908.06760)
- [SMILES2Vec: An Interpretable General-Purpose Deep Neural Network for Predicting Chemical Properties](https://arxiv.org/abs/1712.02034)
- [ReactionT5: a pre-trained transformer model for accurate chemical reaction prediction with limited data](https://link.springer.com/article/10.1186/s13321-025-01075-4)
- [Chemical representation learning for toxicity prediction](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00099g)
- [MolTrans: Molecular Interaction Transformer for drug–target interaction prediction](https://academic.oup.com/bioinformatics/article/37/6/830/5929692)

#### SELFIES-based Models

- [Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources](https://www.nature.com/articles/s41598-025-05017-w)
- [Group SELFIES: a robust fragment-based molecular string representation](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e)
- [SELFormer: Molecular representation learning via SELFIES language models](https://arxiv.org/abs/2304.04662)
- [Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation](https://arxiv.org/abs/1905.13741)

#### Other Sequence Representations

- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [InChI, the IUPAC International Chemical Identifier](https://link.springer.com/article/10.1186/s13321-015-0068-4)
- [DeepTox: Toxicity Prediction using Deep Learning](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2015.00080/full)

------

### Molecule Topological Graph (2D)

- [Graph attention networks](https://arxiv.org/abs/1710.10903)
- [Do Transformers Really Perform Badly for Graph Representation?](https://arxiv.org/abs/2106.05234)
- [MolE: a foundation model for molecular graphs using disentangled attention](https://www.nature.com/articles/s41467-024-53751-y)
- [Language models can explain neurons in language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- [GNN-SKAN: Advancing Molecular Representation Learning with SwallowKAN](https://arxiv.org/abs/2408.01018)
- [Semi-supervised classification with graph convolutional networks](https://arxiv.org/abs/1609.02907)
- [SPECTRA: Spectral Target-Aware Graph Augmentation for Imbalanced Molecular Property Regression](https://arxiv.org/abs/2511.04838)
- [Recipe for a general, powerful, scalable graph transformer](https://arxiv.org/abs/2205.12454)
- [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803)
- [A compact review of molecular property prediction with graph neural networks](https://www.sciencedirect.com/science/article/pii/S1740674920300305)
- [N-gram graph: Simple unsupervised representation for graphs, with applications to molecules](https://arxiv.org/abs/1806.09206)
- [Neural message passing for Quantum chemistry](https://arxiv.org/abs/1704.01212)
- [Chemical Graph-Based Transformer Models for Yield Prediction of High Throughput Cross-Coupling Reaction Datasets](https://pubs.acs.org/doi/10.1021/acsomega.4c06113)
- [Self-supervised graph transformer on large-scale molecular data](https://arxiv.org/abs/2007.02835)
- [How powerful are graph neural networks?](https://arxiv.org/abs/1810.00826)
- [KAGNNs: Kolmogorov-arnold networks meet graph learning](https://arxiv.org/abs/2406.18380)
- [GraphKAN: Graph kolmogorov arnold network for small molecule-protein interaction predictions](https://openreview.net/pdf?id=d5uz4wrYeg)

------

### Geometric Conformation (3D)

- [Exploring chemical compound space with quantum-based machine learning](https://www.nature.com/articles/s41570-020-0189-9)
- [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123)
- [SchNet: a continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566)
- [DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking](https://arxiv.org/abs/2210.01776)
- [Spherical Message Passing for 3D Molecular Graphs](https://arxiv.org/abs/2102.05013)
- [Highly accurate quantum chemical property prediction with uni-mol+](https://arxiv.org/abs/2303.16982)
- [Fast and uncertainty-aware directional message passing for non-equilibrium molecules](https://arxiv.org/abs/2011.14115)
- [Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds](https://arxiv.org/abs/1802.08219)
- [Torchmd-net: equivariant transformers for neural network based molecular potentials](https://arxiv.org/abs/2202.02541)
- [Benchmarking graphormer on large-scale molecular modeling datasets](https://arxiv.org/abs/2203.04810)
- [GemNet: Universal directional graph neural networks for molecules](https://arxiv.org/abs/2106.08903)
- [Gps++: An optimised hybrid mpnn/transformer for molecular property prediction](https://arxiv.org/abs/2212.02229)
- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)
- [Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations](https://arxiv.org/abs/2502.06073)
- [Molecular geometry-aware transformer for accurate 3d atomic system modeling](https://arxiv.org/abs/2302.00855)
- [Molecule attention transformer](https://arxiv.org/abs/2002.08264)
- [Geometry-enhanced molecular representation learning for property prediction](https://www.nature.com/articles/s42256-021-00438-4)

------

### Multimodal Representations

- [Pre-training molecular graph representation with 3d geometry](https://arxiv.org/abs/2110.07728)
- [Graph-BERT and language model-based framework for protein--protein interaction identification](https://www.nature.com/articles/s41598-023-31612-w)
- [Holo-Mol: An explainable hybrid deep learning framework for predicting reactivity of hydroxyl radical to water contaminants based on holographic fused molecular representations](https://www.sciencedirect.com/science/article/pii/S1385894724001372)
- [GraSeq: graph and sequence fusion learning for molecular property prediction](https://dl.acm.org/doi/10.1145/3340531.3411981)

------

## Model Architectures

### Geometric GNNs

<p align="center">
  <img src="Figure\fig3_GNN.png" alt="Geometric GNN in MPP" width="1000"/>
</p>
<p align="center">Geometric GNN in MPP</p>

- [SchNet: a continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566)
- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)
- [GemNet: Universal directional graph neural networks for molecules](https://arxiv.org/abs/2106.08903)
- [SE(3)-transformers: 3d roto-translation equivariant attention networks](https://arxiv.org/abs/2006.10503)
- [Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations](https://arxiv.org/abs/2502.06073)
- [Spherical Message Passing for 3D Molecular Graphs](https://arxiv.org/abs/2102.05013)
- [Geometry-enhanced molecular representation learning for property prediction](https://www.nature.com/articles/s42256-021-00438-4)
- [Molecular geometry-aware transformer for accurate 3d atomic system modeling](https://arxiv.org/abs/2302.00855)
- [Fast and uncertainty-aware directional message passing for non-equilibrium molecules](https://arxiv.org/abs/2011.14115)
- [DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking](https://arxiv.org/abs/2210.01776)
- [Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds](https://arxiv.org/abs/1802.08219)

------

### Graph Transformers

<p align="center">
  <img src="Figure\fig4_Transformer.png" alt="Transformer in MPP" width="1000"/>
</p>
<p align="center">Transformer in MPP</p>

- [MolE: a foundation model for molecular graphs using disentangled attention](https://www.nature.com/articles/s41467-024-53751-y)
- [SimSon: Simple contrastive learning of SMILES for molecular property prediction](https://academic.oup.com/bioinformatics/article/41/5/btaf275/8127203)
- [Do Transformers Really Perform Badly for Graph Representation?](https://arxiv.org/abs/2106.05234)
- [Benchmarking graphormer on large-scale molecular modeling datasets](https://arxiv.org/abs/2203.04810)
- [Molecular geometry-aware transformer for accurate 3d atomic system modeling](https://arxiv.org/abs/2302.00855)
- [Torchmd-net: equivariant transformers for neural network based molecular potentials](https://arxiv.org/abs/2202.02541)
- [Molecule attention transformer](https://arxiv.org/abs/2002.08264)
- [Chemical Graph-Based Transformer Models for Yield Prediction of High Throughput Cross-Coupling Reaction Datasets](https://pubs.acs.org/doi/10.1021/acsomega.4c06113)
- [Recipe for a general, powerful, scalable graph transformer](https://arxiv.org/abs/2205.12454)
- [Directed message passing based on attention for prediction of molecular properties](https://arxiv.org/abs/2305.14819)
- [Self-supervised graph transformer on large-scale molecular data](https://arxiv.org/abs/2007.02835)

------

### Hybrid Architectures

- [Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models](https://arxiv.org/abs/1706.06689)
- [Gps++: An optimised hybrid mpnn/transformer for molecular property prediction](https://arxiv.org/abs/2212.02229)
- [Pre-training molecular graph representation with 3d geometry](https://arxiv.org/abs/2110.07728)
- [GraSeq: graph and sequence fusion learning for molecular property prediction](https://dl.acm.org/doi/10.1145/3340531.3411981)
- [KAGNNs: Kolmogorov-arnold networks meet graph learning](https://arxiv.org/abs/2406.18380)
- [GNN-SKAN: Advancing Molecular Representation Learning with SwallowKAN](https://arxiv.org/abs/2408.01018)
- [Graph-BERT and language model-based framework for protein--protein interaction identification](https://www.nature.com/articles/s41598-023-31612-w)
- [GraphKAN: Graph kolmogorov arnold network for small molecule-protein interaction predictions](https://openreview.net/pdf?id=d5uz4wrYeg)
- [Holo-Mol: An explainable hybrid deep learning framework for predicting reactivity of hydroxyl radical to water contaminants based on holographic fused molecular representations](https://www.sciencedirect.com/science/article/pii/S1385894724001372)

------

### Quantum Hybrid Models

- [NeuroQ: Quantum-Inspired Brain Emulation](https://www.mdpi.com/2313-7673/10/8/516)
- [Differentiable quantum computational chemistry with PennyLane](https://arxiv.org/abs/2111.09967)
- [Generalizing neural wave functions](https://arxiv.org/abs/2302.04168)

------

## Applications

### Drug Discovery

- [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- [Molecular representation learning with language models and domain-relevant auxiliary tasks](https://arxiv.org/abs/2011.13230)
- [ChemBERTa-2: Towards chemical foundation models](https://arxiv.org/abs/2209.01712)
- [DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking](https://arxiv.org/abs/2210.01776)
- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)
- [GraphDTA: predicting drug–target binding affinity with graph neural networks](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)
- [SELFormer: Molecular representation learning via SELFIES language models](https://arxiv.org/abs/2304.04662)
- [MolE: a foundation model for molecular graphs using disentangled attention](https://www.nature.com/articles/s41467-024-53751-y)
- [Neural message passing for Quantum chemistry](https://arxiv.org/abs/1704.01212)
- [Domain adaptation of a SMILES chemical transformer to SELFIES with limited computational resources](https://www.nature.com/articles/s41598-025-05017-w)
- [MolTrans: Molecular Interaction Transformer for drug–target interaction prediction](https://academic.oup.com/bioinformatics/article/37/6/830/5929692)
- [ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)
- [Fast and uncertainty-aware directional message passing for non-equilibrium molecules](https://arxiv.org/abs/2011.14115)
- [Molecular geometry-aware transformer for accurate 3d atomic system modeling](https://arxiv.org/abs/2302.00855)
- [Self-supervised graph transformer on large-scale molecular data](https://arxiv.org/abs/2007.02835)
- [SPECTRA: Spectral Target-Aware Graph Augmentation for Imbalanced Molecular Property Regression](https://arxiv.org/abs/2511.04838)
- [DeepDTA: deep drug–target binding affinity prediction](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
- [Graph attention networks](https://arxiv.org/abs/1710.10903)
- [Chemprop: A Machine Learning Package for Chemical Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250)
- [Fate-tox: fragment attention transformer for E(3)-equivariant multi-organ toxicity prediction](https://link.springer.com/article/10.1186/s13321-025-01012-5)
- [Geometry-enhanced molecular representation learning for property prediction](https://www.nature.com/articles/s42256-021-00438-4)
- [Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of Expert-developed QSAR/QSPR Models](https://arxiv.org/abs/1706.06689)
- [Molecular contrastive learning of representations via graph neural networks](https://www.nature.com/articles/s42256-022-00447-x)

------

### Materials Design

- [Exploring chemical compound space with quantum-based machine learning](https://www.nature.com/articles/s41570-020-0189-9)
- [SchNet: a continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566)
- [Allegro-FM: Toward an Equivariant Foundation Model for Exascale Molecular Dynamics Simulations](https://arxiv.org/abs/2502.06073)
- [Generalizing neural wave functions](https://arxiv.org/abs/2302.04168)
- [PND: Physics-informed neural-network software for molecular dynamics applications](https://www.sciencedirect.com/science/article/pii/S2352711021000972)
- [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9)
- [Catalyst Energy Prediction with CatBERTa: Unveiling Feature Exploration Strategies through Large Language Models](https://pubs.acs.org/doi/10.1021/acscatal.3c04956)
- [CataLM: Empowering Catalyst Design Through Large Language Models](https://arxiv.org/abs/2405.17440)
- [Uni-Electrolyte: An Artificial Intelligence Platform for Designing Electrolyte Molecules for Rechargeable Batteries](https://arxiv.org/abs/2412.00498)
- [A predictive machine learning force-field framework for liquid electrolyte development](https://www.nature.com/articles/s42256-025-01009-7)
- [Generative Pretrained Transformer for Heterogeneous Catalysts](https://pubs.acs.org/doi/10.1021/jacs.4c11504)

------

### Other Applications

- [Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds](https://arxiv.org/abs/1802.08219)
- [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](https://arxiv.org/abs/1908.06760)
- [Chemformer: A Pre-Trained Transformer for Computational Chemistry](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b)
- [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://pubs.acs.org/doi/10.1021/acscentsci.9b00576)
- [Integration of Transfer Learning and Multitask Learning To Predict the Potential of Per/Polyfluoroalkyl Substances in Activating Multiple Nuclear Receptors Associated with Hepatic Lipotoxicity](https://pubs.acs.org/doi/10.1021/acs.est.5c07895)
- [Self-Attention Based Molecule Representation for Predicting Drug-Target Interaction](https://arxiv.org/abs/1908.06760)
- [Chemformer: A Pre-Trained Transformer for Computational Chemistry](https://chemrxiv.org/engage/chemrxiv/article-details/60ee8a3eb95bdd06d062074b)
- [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://pubs.acs.org/doi/10.1021/acscentsci.9b00576)
- [Integration of Transfer Learning and Multitask Learning To Predict the Potential of Per/Polyfluoroalkyl Substances in Activating Multiple Nuclear Receptors Associated with Hepatic Lipotoxicity](https://pubs.acs.org/doi/10.1021/acs.est.5c07895)
