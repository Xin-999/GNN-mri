GNN Model Sources and Characteristics (fMRI Brain Graphs)
=========================================================

This note summarizes the main sources and characteristics for the three GNN families used in this project: FBNetGen, BrainGNN, and GATv2. It focuses on what each architecture contributes conceptually and why it is relevant for fMRI functional brain graphs.

FBNetGen (Task-aware graph generation for fMRI)
------------------------------------------------
Primary source: FBNETGEN: Task-aware GNN-based fMRI Analysis via Functional Brain Network Generation (MIDL 2022).

Core idea
- Learns a task-aware graph generator from fMRI time-series, then applies a GNN to the learned graph for prediction.
- Designed to make the graph itself more task-relevant than fixed correlation-based graphs.

Key characteristics
- End-to-end pipeline: ROI time-series -> graph generation -> GNN prediction.
- Learned graphs can emphasize task-relevant connectivity patterns.
- Intended to improve interpretability by highlighting important regions/edges.

Implication for this project
- Best fit when you believe the connectivity graph should be learned or denoised rather than fixed.
- More parameters and compute due to the graph generation component.

BrainGNN (Brain-specific interpretable GNN)
------------------------------------------
Primary source: BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis (Medical Image Analysis 2021).

Core idea
- Adds brain-structure priors through ROI-aware convolution and ROI-selection pooling to improve interpretability.

Key characteristics
- ROI-aware graph convolution (Ra-GConv) for region-specific transformations.
- ROI-selection pooling (R-pool) to identify salient brain regions.
- Additional regularization losses to stabilize ROI selection and encourage meaningful biomarkers.

Implication for this project
- Strong when interpretability and neurobiological relevance are important.
- Slightly heavier to train/tune because of extra regularizers and pooling.

GATv2 (Dynamic graph attention)
-------------------------------
Primary source: How Attentive are Graph Attention Networks? (ICLR 2022).

Core idea
- GATv2 fixes the static-attention limitation of original GAT by reordering operations to enable dynamic attention.

Key characteristics
- More expressive attention mechanism than GAT (dynamic attention).
- Often improves performance without significantly increasing parameter count.

Implication for this project
- Good baseline for weighted brain graphs when you want attention without explicit brain-specific priors.
- Typically faster to tune than BrainGNN or FBNetGen.

Quick comparison (high-level)
-----------------------------
- FBNetGen: learns the graph itself; good when fixed correlation graphs are noisy.
- BrainGNN: brain-structure priors + ROI interpretability.
- GATv2: expressive attention baseline with good performance/efficiency trade-off.

References
----------
- FBNETGEN (MIDL 2022): https://2022.midl.io/papers/b1
- BrainGNN (MedIA 2021): https://doi.org/10.1016/j.media.2021.102233
- BrainGNN (PubMed entry): https://pubmed.ncbi.nlm.nih.gov/33636723/
- GATv2 (ICLR 2022): https://iclr.cc/virtual/2022/poster/6366
- GATv2 (Papers With Code summary): https://paperswithcode.com/paper/how-attentive-are-graph-attention
