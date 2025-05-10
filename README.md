# HF_Qwen_Double_Quantization
Qwen2-7B-Instruct double quantized model
# HF QWEN Double Quantization Notebook

This notebook demonstrates how to apply **double quantization** to the Qwen 2-7B Instruct model using Hugging Face Transformers and BitsAndBytes. Double quantization reduces model size and accelerates inference by quantizing weights to 4-bit precision with additional optimization.

## Prerequisites

- Python 3.8+
- GPU with CUDA support and sufficient memory
- Hugging Face account with an access token (`HF_TOKEN`)
- GitHub token stored as a Kaggle secret (`GITHUB_TOKEN`)
- Kaggle environment (optional) or a local setup configured with your tokens

## Notebook Overview

**Cell 1: Install Dependencies**  
```bash
!pip install -q requests torch bitsandbytes transformers sentencepiece accelerate
```
Installs core libraries:  
- `torch` for PyTorch  
- `bitsandbytes` for quantization support  
- `transformers`, `sentencepiece`, and `accelerate` for model and tokenization  

---

**Cell 2: Upgrade RAPIDS Dependencies**  
```bash
!pip install --upgrade \
    pylibraft-cu12==24.12.0 \
    rmm-cu12==24.12.0
```
Upgrades `pylibraft` and `rmm` to ensure compatibility with CUDA 12 for memory-managed GPU operations.

---

**Cell 3: Import Libraries**  
```python
from google.colab import userdata
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig
)
import torch
import gc
```
Imports all required modules for tokenization, model loading, inference streaming, and quantization.

---

**Cell 4: Hugging Face Authentication**  
```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token)
```
Retrieves the Hugging Face token from Kaggle secrets and logs in to enable pushing and pulling models.

---

**Cell 5: Quantization Configuration**  
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```
Defines a `BitsAndBytesConfig` for 4-bit quantization with **double quant** using the NF4 quantization type.

---

**Cell 6: Model Identifier**  
```python
QWEN = "Qwen/Qwen2-7B-Instruct"
```
Specifies the Hugging Face repository identifier for the base Qwen 2-7B Instruct model.

---

**Cell 7: Example Chat Messages**  
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of ship captains"}
]
```
Prepares a simple system and user message pair to demonstrate chat-style inference.

---

**Cell 8: Tokenizer Initialization and Input Preparation**  
```python
tokenizer = AutoTokenizer.from_pretrained(QWEN)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
```
- Loads the tokenizer  
- Sets the pad token to the end-of-sequence token  
- Converts messages into model-ready tensors on GPU

---

**Cell 9: Model Loading with Quantization**  
```python
model = AutoModelForCausalLM.from_pretrained(
    QWEN,
    device_map="auto",
    quantization_config=quant_config
)
```
Loads the Qwen model in 4-bit precision with the specified quantization configuration and maps layers across available devices automatically.

---

**Cell 10: Push Quantized Model to Hub**  
```python
model.push_to_hub("premkumarkora/qkora2-7B-Instruct")
tokenizer.push_to_hub("premkumarkora/qkora2-7B-Instruct")
```
Uploads the quantized model and tokenizer to the user’s Hugging Face Hub repository.

---

**Cell 11: GitHub Authentication Setup (Optional)**  
```python
from kaggle_secrets import UserSecretsClient
import os, subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
os.environ["GITHUB_TOKEN"] = token

subprocess.run(["git", "config", "--global", "user.name", "Your Name"])
subprocess.run(["git", "config", "--global", "user.email", "you@example.com"])
subprocess.run(["git", "config", "--global", "credential.helper", "store"])
cred = f"https://{token}:x-oauth-basic@github.com
"
with open(os.path.expanduser("~/.git-credentials"), "w") as f:
    f.write(cred)
```
Configures Git with stored GitHub credentials for seamless pushes to GitHub from the notebook.

---

## Usage

1. Enable GPU runtime (if using Colab or Kaggle).  
2. Ensure `HF_TOKEN` and `GITHUB_TOKEN` are configured in your environment.  
3. Run all cells sequentially.  
4. Check the Hugging Face Hub for the new quantized model repository.

---

## License

This notebook is provided under the MIT License.

