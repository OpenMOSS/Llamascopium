# Llamascopium

**Start here:** [Install](#installation) · [Documentation](https://openmoss.ai/Llamascopium/) · [Run an experiment](#launch-an-experiment) · [Visualize features](#visualizing-the-learned-dictionary) · [Examples](examples/)

> [!NOTE]
> This package was previously published as **`lm-saes`** (project name: **Language-Model-SAEs**) and now it has been renamed to `llamascopium`.

`llamascopium` is a comprehensive, **fully-distributed** framework designed for **training, analyzing and visualizing Sparse Autoencoders (SAEs)**, empowering scalable and systematic **Mechanistic Interpretability** research.

## News

- 2026.2.12 We introduce **Complete Replacement Models (CRMs)**, which combine transcoders and Lorsas to fully sparsify language models. Link: [Bridging the Attention Gap: Complete Replacement Models for Complete Circuit Tracing](https://interp.open-moss.com/posts/complete-replacement).

- 2025.9.23 We leverage **Crosscoder** to track feature evolution across pre-training snapshots. Link: [Evolution of Concepts in Language Model Pre-Training](https://www.arxiv.org/abs/2509.17196) (ICLR 2026).

- 2025.8.23 We identify a prevalent low-rank structure in attention outputs as the key cause of dead features, and propose **Active Subspace Initialization** to improve sparse dictionary learning on these low-rank activations. Link: [Dimensional Collapse in Transformer Attention Outputs: A Challenge for Sparse Dictionary Learning](https://arxiv.org/abs/2508.16929) (ICML 2026).

<details>
<summary>Earlier updates</summary>

- 2025.4.29 We introduce **Low-Rank Sparse Attention (Lorsa)** to attack attention superposition, extracting tens of thousands of true attention units from LLM attention layers. Link: [Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](https://arxiv.org/abs/2504.20938) (ICLR 2026).

- 2024.10.29 We introduce **Llama Scope**, our first contribution to the open-source Sparse Autoencoder ecosystem. Stay tuned! Link: [Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders](http://arxiv.org/abs/2410.20526).

- 2024.10.9 Transformers and Mambas are mechanistically similar in both feature and circuit level. Can we follow this line and find **universal motifs and fundamental differences between language model architectures**? Link: [Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures](https://arxiv.org/pdf/2410.06672) (ICLR 2025).

- 2024.5.22 We propose hierarchical tracing, a promising method to **scale up sparse feature circuit analysis** to industrial sized language models! Link: [Automatically Identifying Local and Global Circuits with Linear Computation Graphs](https://arxiv.org/pdf/2405.13868) (ICML 2024 MI Workshop).

- 2024.2.19 Our first attempt on SAE-based circuit analysis for Othello-GPT leads us to **an example of Attention Superposition in the wild**! Link: [Dictionary learning improves patch-free circuit discovery in mechanistic interpretability: A case study on othello-gpt](https://arxiv.org/pdf/2402.12201).

</details>

## Features

- **Scalability**: Our framework is fully distributed with arbitrary combinations of data, model, and head parallelism for both training and analysis. Enjoy training SAEs with millions of features!
- **Flexibility**: We support a wide range of SAE variants, including vanilla SAEs, Lorsa (Low-rank Sparse Attention), CLT (Cross-layer Transcoder), MOLT (Mixture of Linear Transforms), Crosscoder, and more. Each variant can be combined with different activation functions (e.g., ReLU, JumpReLU, TopK, BatchTopK) and sparsity penalties (e.g., L1, Tanh).
- **Easy to Use**: We provide high-level `runners` APIs to quickly launch experiments with simple configurations. Check our [examples](examples) for verified hyperparameters.
- **Visualization**: We provide a unified web interface to visualize learned SAE variants and their features.

## Installation

Use [pip](https://pypi.org/project/pip/) to install Llamascopium:

```bash
pip install llamascopium==2.0.0b34
```

We also highly recommend using [uv](https://docs.astral.sh/uv/) to manage your own project dependencies. You can use

```bash
uv add llamascopium==2.0.0b34
```

to add Llamascopium as your project dependency.

## Development

We use [uv](https://docs.astral.sh/uv/) to manage the dependencies, which is an alternative to [poetry](https://python-poetry.org/) or [pdm](https://pdm-project.org/). To install the required packages, just install [uv](https://docs.astral.sh/uv/getting-started/installation/), and run the following command:

```bash
uv sync
```

This will install all the required packages for the codebase in `.venv` directory. For Ascend NPU support, run

```bash
uv sync --extra npu
```

If you want to use the visualization tools, you also need to install the required packages for the frontend, which uses [bun](https://bun.sh/) for dependency management. Follow the instructions on the website to install it, and then run the following command:

```bash
cd ui
bun install
```

## Launch an Experiment

Explore the `examples` to check the basic usage of training/analyzing SAEs in different configurations. Note a MongoDB is recommended for recording the model/dataset/SAE configurations and required for storing analyses. For more advanced usage, you may explore `src/llamascopium/runners` folder for the interface for generating activations and training & analyzing SAE variants, and directly write your own variant of training/analyzing script at the runner level.

## Visualizing the Learned Dictionary

The analysis results will be saved using MongoDB, and you can use the provided visualization tools to visualize the learned dictionary. First, start the FastAPI server by running the following command:

```bash
uvicorn server.app:app --port 24577 --env-file server/.env
```

Then, copy the `ui/.env.example` file to `ui/.env` and modify the `BACKEND_URL` to fit your server settings (by default, it's `http://localhost:24577`), and start the frontend by running the following command:

```bash
cd ui
bun dev --port 24576
```

That's it! You can now go to `http://localhost:24576` to visualize the learned dictionary and its features.

## Development

We highly welcome contributions to this project. If you have any questions or suggestions, feel free to open an issue or a pull request. We are looking forward to hearing from you!

TODO: Add development guidelines

## Acknowledgement

The design of the pipeline (including the configuration and some training details) is highly inspired by the [mats_sae_training
](https://github.com/jbloomAus/mats_sae_training) project (now known as [SAELens](https://github.com/jbloomAus/SAELens)) and heavily relies on the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library. We thank the authors for their great work.

## Citation

Please cite this library as:

```
@misc{Ge2024OpenMossSAEs,
    title  = {OpenMoss Language Model Sparse Autoencoders},
    author = {Xuyang Ge, Wentao Shu, Junxuan Wang, Guancheng Zhou, Jiaxing Wu, Fukang Zhu, Lingjie Chen, Zhengfu He},
    url    = {https://github.com/OpenMOSS/Llamascopium},
    year   = {2024}
}
```
