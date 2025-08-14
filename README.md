# From IM to PIM: Revolutionizing Influence Maximization with Personalized Seed Generation

This repository contains the source code for the paper  
**"From IM to PIM: Revolutionizing Influence Maximization with Personalized Seed Generation"** 
submitted to *IEEE Transactions on Mobile Computing*.

---

## 1. Environment Setup
Please create and activate the environment using the provided `conda_env.yml`:

```bash
conda env create -f conda_env.yml
conda activate pim_env
```

## 2. Repository Structure
```
.
├── encoder/                  # Implements the Dual-View Influence Encoder module
├── decoder/                  # Implements the Adaptive Influencer Decoder module
│   └── main.py               # Main entry point for running experiments
├── conda_env.yml             # Conda environment configuration
├── README.md                 # Project documentation

```
## 3. Running the Code
Run experiments by executing the main.py script in the decoder/ folder:
```bash
cd decoder
python main.py
```
Key arguments:

--dataset: Name of the dataset (e.g., PPSD)
--model: Model to run (e.g., PIM)
--epochs: Number of training epochs
--seed: Random seed for reproducibility

Example:
```bash
python main.py --dataset PPSD --model PIM --epochs 100 --seed 42
```





