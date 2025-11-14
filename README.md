<div align="center">

# Nano-GPT-OSS Language Model

**An open-source transformer that balances full-context and sliding-window attention for efficient, scalable LLM training and inference.**


---
## Key Improvements of GPT-OSS over GPT-2

### 🏗️ Architecture Enhancements
- **Mixture of Experts (MoE) in MLP** with a Router → Sparse experts active per token (big model capacity, low active FLOPs)
- **Gated Router** → Token-dependent routing to experts (shown inside MoE block)
- **SwiGLU Feed-Forward (FFN) modules** → Modern activation in FFN instead of GELU
- **Grouped Query Attention + RoPE** → Alternate attention that supports longer context and stable queries
- **Sliding Window Attention** → Efficient attention pattern that reduces computation while maintaining context
- **Sink Slots in Attention** → Learned aggregation slots for global context stability
- **RMSNorm** → More stable normalization layer


## Dependencies
- [pytorch](https://pytorch.org) <3
-  `datasets` for huggingface datasets <3 (for loading datasets)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3
-  `ipywidgets` for optional jupyter notebook support 

## 📊 Dataset and Format

TinyStories can be found at [HuggingFace Datasets](https://huggingface.co/datasets/roneneldan/TinyStories).

### Data Fields:

Each story entry contains:

- `story`: The main story text
<details>
<summary>📝 Click to see example story</summary>

**Story:**

```
Once upon a time, there was a big, red ball that could bounce very high...
```

\[Rest of the example story\]

</details>

## 🚀 Installation


### 📦 Pip Installation

```


# create conda environment
conda create -n myenv python=3.10
conda activate myenv
# Install requirements
pip install -r requirements.txt
pip install -r requirements.txt
```


## How to run Training

The system will automatically detect and utilize available GPU resources. To train the GPT-OSS model, choose one of the following methods:

### Option 1: Command Line Interface

1. Navigate to the project directory:
   ```bash
   cd gpt-oss
   ```

2. Start training with default configuration:
   ```bash
   python train.py
   ```

### Option 2: Jupyter Notebook

1. Launch Jupyter from the project directory:
   ```bash
   jupyter notebook
   ```

2. Open `trains.ipynb`
3. Run all cells sequentially using `Cell > Run All`

### Monitoring Training
- Training progress and metrics will be displayed in the console
- Model checkpoints are saved in the `checkpoints` directory
- Training logs can be found in the `logs` directory

