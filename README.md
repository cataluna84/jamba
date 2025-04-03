# Jamba
PyTorch Implementation of Jamba: "Jamba: A Hybrid Transformer-Mamba Language Model"

## Native uv Python and package management

### 1. Install uv

Uv can be installed as follows, depending on your operating system.

**macOS and Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

> **Note:**
> For more installation options, please refer to the official [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

### 2. Install Python packages and dependencies

Recommended to use Python 3.12 or 3.13
```bash
uv venv --python 3.12
```

After that do
```bash
source .venv/bin/activate
```

Install dependencies from requirements.txt
```bash
uv run python -m pip install -U -r requirements.txt
```

You can install new packages, that are not specified in the `pyproject.toml` via `uv add`, for example:

```bash
uv add packaging
```

And you can remove packages via `uOn Windows (PowerShell):

```bash
.venv\Scripts\activate
```

uv remove packaging
```

## Train

```bash
uv run python train.py
```


**Skipping the `uv run` command**

If you find typing `uv run` cumbersome, you can manually activate the virtual environment as described below.

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then, you can run scripts via

```bash
python script.py
```

and launch JupyterLab via

```bash
jupyter lab
```


## Usage

```python
# Import the torch library, which provides tools for machine learning
import torch

# Import the Jamba model from the jamba.model module
from jamba.model import Jamba

# Create a tensor of random integers between 0 and 100, with shape (1, 100)
# This simulates a batch of tokens that we will pass through the model
x = torch.randint(0, 100, (1, 100))

# Initialize the Jamba model with the specified parameters
# dim: dimensionality of the input data
# depth: number of layers in the model
# num_tokens: number of unique tokens in the input data
# d_state: dimensionality of the hidden state in the model
# d_conv: dimensionality of the convolutional layers in the model
# heads: number of attention heads in the model
# num_experts: number of expert networks in the model
# num_experts_per_token: number of experts used for each token in the input data
model = Jamba(
    dim=512,
    depth=6,
    num_tokens=100,
    d_state=256,
    d_conv=128,
    heads=8,
    num_experts=8,
    num_experts_per_token=2,
)

# Perform a forward pass through the model with the input data
# This will return the model's predictions for each token in the input data
output = model(x)

# Print the model's predictions
print(output)

```


# License
MIT
