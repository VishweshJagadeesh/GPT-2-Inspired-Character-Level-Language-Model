# GPT-2 Inspired Character-Level Language Model

A Transformer-based language model trained from scratch on Shakespeare's works using character-level tokenization. Built with PyTorch and designed to run efficiently on mid-range GPUs.

## Overview

This project implements a GPT-2 style autoregressive language model using key architectural elements such as:

-  Multi-head self-attention
-  Positional embeddings
-  Transformer blocks with GELU activations
-  Character-level tokenization for simplicity and speed

The model was trained on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and can generate stylistically coherent text resembling Shakespeare's writing.

## Features

-  Pure PyTorch implementation of GPT-like architecture
-  Supports character-level encoding (simpler and faster to train on small datasets)
-  Implements:

  -  Layer normalization
  -  Dropout regularization
  -  Gradient clipping
  -  Best-model checkpoint saving
-  Compatible with CUDA (GPU acceleration)
-  Outputs generated Shakespeare-style text samples

## Setup

1. Clone this repo and navigate to the project directory.
2. Install dependencies:

```
pip -r requirements.txt
```

## Training

To train the model:

```
python gpt.py
```

Model checkpoints will be saved as `best_model.pt` based on validation loss.
If you wanna see a quick demo the `best_model.pt` is also attached in the repo


## Configuration

You can customize the model by adjusting these values in `gpt.py`:

```python
batch_size = 64            # number of sequences per batch
context_length = 256       # sequence length (reduce for smaller GPUs)
n_embd = 256               # embedding size
n_head = 4                 # number of attention heads
n_layer = 4                # number of transformer blocks
```

## Sample Output

### After training, the model generated text like these:

```
This will as i'er some, that virtue shall have so.

DUKE OF AUMERLE:
I will.

LUCIO:
Have you three your lord's felligner.

ISABELLA:
O, prothee, deny his tonguest a comender-hearted become.

DUCHESS OF YORK:
In King of Henry, Ambon's punishes and obsence.
That beauty directied, but and big subject.

DUCHESS OF YORK:
A well, 'tis chasel. When we will at use
This servIce he things to a child hear your eyes?
He's not prode the parchaiuting but well.'
What is he wear thier tempest!
God give them thee
```

```
Once, of thy Isaby doom in my kedge,
That post unworth, whose all my head seems and my son
That doth she doth I food'st forward. Thyself,
Encale rhereformstorse of thy father's eyes
Is not do ask, that twoman strong to thy brothers:
But thou art these extress. For one came none mather.
What. We do ouse that be young
To won defendere wherether thou, forsworn will be
Be in time would death thought we desert unto happy
Camiles the childrens and the soul
Of the phaps and dog of my made them; he
From
```

## Model Saving & Loading

-  Model weights are saved using:

```python
torch.save(model.state_dict(), 'best_model.pt')
```

-  To reload and generate:

```python
model = GPTLanguageModel()
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.eval()
```

## Future Work

-  Add top-k / top-p sampling for more diverse generation
-  Upgrade to BPE tokenization for larger corpora
-  Export to ONNX or integrate with a web UI using Gradio


---
