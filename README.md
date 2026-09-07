Importent experiment scripts is in LoRA modified.
//
scripts end with onlyCNlora is for config 1.
//
scripts end with ctrlora is for config 2.
//
scripts end with addlrSD is for config 3.

# Interactive Stylization of Topographic Maps Using Prompt-Driven Diffusion Models

A research project exploring **text-guided cartographic stylization with diffusion models**, with a focus on adapting pretrained Stable Diffusion and ControlNet architectures to interpret natural-language cartographic instructions while preserving the underlying spatial structure of topographic maps.

The project investigates how parameter-efficient adaptation, alternative text encoders, and structured prompt design can enable diffusion models to perform controllable map-editing operations such as changing **map-element colors, visibility, morphology, and rendering styles**.

---

## Overview

Conventional map styling typically requires users to manually modify cartographic parameters such as colors, line widths, visibility, and symbol styles. This project explores a different interaction paradigm:

> **Can a user describe the desired cartographic modification in natural language and let a generative model perform the corresponding map transformation?**

The framework combines:

* a pretrained **Stable Diffusion** backbone,
* **ControlNet-based spatial conditioning** to preserve map geometry,
* a **BERT-based text encoder** for representing cartographic instructions,
* parameter-efficient **LoRA adaptation**, and
* structured prompt datasets describing map-editing operations.

The resulting experimental pipeline studies how language-conditioned diffusion models can learn mappings such as:

```text
"Set building fill color to yellow"
```

```text
"Modify roads to be wider"
```

```text
"Make forests hidden"
```

and compositional instructions such as:

```text
"Set road color to dark gray and modify roads to be wider"
```

while maintaining the spatial organization of the input map.

---

## Research Questions

The experiments in this repository mainly investigate three questions:

1. **How should a pretrained diffusion model be adapted to cartographic styling tasks?**

   Different LoRA configurations are compared to determine how much adaptation is required in the ControlNet and Stable Diffusion components.

2. **How can natural-language map-editing instructions be represented effectively?**

   Several text-encoder configurations are explored, including CLIP- and BERT-based representations and lightweight trainable adapters.

3. **Can the learned model generalize from atomic styling instructions to compositional prompts?**

   Prompt-composition experiments combine operations involving color, morphology, visibility, and rendering style.

---

## Model Architecture

At a high level, the experimental pipeline follows:

```text
Natural-language instruction
          │
          ▼
   BERT Text Encoder
          │
    Trainable Adapter
          │
          ▼
 ┌─────────────────────┐
 │ Stable Diffusion    │
 │      U-Net          │
 └─────────┬───────────┘
           │
       LoRA modules
           │
           │
Input map ─┼──► ControlNet / CTRLora
           │
           ▼
  Stylized map output
```

The map provides the structural conditioning signal, while the text prompt specifies the requested cartographic transformation.

The BERT backbone is kept frozen in the principal experiments, with a lightweight adapter trained on top of its token-level hidden representations.

---

## Parameter-Efficient Adaptation

A major part of the project investigates where LoRA adaptation should be introduced.

Three principal configurations are implemented.

### Configuration 1 — ControlNet LoRA

Only LoRA modules associated with the ControlNet branch are trained.

Main script:

```text
LoRA_modified/
└── multiseg_19prompts_batch1_shuffle_onlyCNLora.py
```

This configuration tests whether adapting the spatial-conditioning pathway alone is sufficient for learning prompt-controlled cartographic transformations.

---

### Configuration 2 — CTRLora

A CTRLora-style configuration extends parameter-efficient adaptation within the ControlNet architecture.

Main script:

```text
LoRA_modified/
└── multiseg_19prompts_batch1_shuffle_ctrlora.py
```

This configuration provides greater flexibility for adapting the map-conditioning branch while keeping most pretrained diffusion parameters frozen.

---

### Configuration 3 — CTRLora + Stable Diffusion U-Net LoRA

The most flexible configuration additionally introduces low-rank adaptation into the main Stable Diffusion U-Net.

Main script:

```text
LoRA_modified/
└── multiseg_19prompts_batch1_shuffle_addlrSD.py
```

In this setting, training updates include:

* the BERT text adapter,
* selected CTRLora / ControlNet parameters, and
* LoRA layers inserted into the Stable Diffusion U-Net attention projections.

The U-Net LoRA modules are applied to the attention **Q/K/V projections**, allowing limited adaptation of the generative backbone without full model fine-tuning.

---

## Prompt Categories

The training and evaluation experiments organize map-editing instructions into several semantic categories.

### Element Color

Changes the appearance of specific cartographic objects.

Examples:

```text
Set building fill color to yellow
Set road color to dark gray
Set river color to bright blue
```

### Morphology

Changes geometric or visual properties of map elements.

Examples:

```text
Modify buildings to have rounded corners
Modify roads to be wider
Modify streams to be smoother
```

### Visibility

Controls whether selected map elements should be visually present.

Examples:

```text
Make forests hidden
Make lakes visible
```

### Element Styling

Changes rendering conventions rather than only color or geometry.

Examples:

```text
Render rivers in dashed outline style
Render streams in wavy line style
```

---

## Compositional Prompt Evaluation

Beyond individual transformations, the project evaluates whether independently learned concepts can be combined within a single instruction.

Examples include:

```text
Set building fill color to yellow and modify buildings to have rounded corners
```

```text
Set road color to dark gray and modify roads to be wider
```

```text
Make lakes hidden and render rivers in dashed outline style
```

The corresponding experiments are implemented in:

```text
PromptEngineering/
└── Compositionality_TEST.py
```

This experiment is intended to test whether the model learns reusable cartographic concepts rather than simply memorizing complete training prompts.

---

## Text-Encoder Experiments

The repository also contains a series of experiments investigating how text-conditioning representations affect map stylization.

```text
TextEncoder_Finetuning/
├── Training_sd_bert_embfussion.py
├── Training_sd_clip_bert_tokenadd.py
├── Training_sd_clip_bert_tokenjoint.py
├── Training_sd_clip_bert_tokenjoint_77.py
├── Training_sdfusion_bert.py
├── Training_sdfusion_bert_lora.py
├── Training_sdfusion_bert_lora_qkvo.py
├── Training_sdfusion_bert_pooler.py
└── Training_sdfusion_clip.py
```

These experiments explore different strategies for integrating BERT and CLIP representations with the diffusion conditioning pipeline.

In the main BERT-based configuration, `bert-base-uncased` is used as the pretrained language encoder. Its original parameters are frozen and token-level hidden states are passed through a trainable adapter before being supplied to the diffusion model.

---

## Prompt Engineering Experiments

Prompt representation and semantic separability are additionally studied in:

```text
PromptEngineering/
├── Compositionality_TEST.py
├── PromptTest_UMAP.py
└── empty_prompt_data_creation.py
```

These scripts support experiments on:

* prompt-category structure,
* representation visualization,
* compositional instructions, and
* unconditional / empty-prompt samples.

---

## Quantitative Evaluation

Generated maps can be evaluated against target maps using several image-level metrics.

The primary evaluation implementation is:

```text
LoRA_modified/
└── quantitative_metrics.py
```

The current evaluation pipeline reports:

* **MSE** — pixel-level reconstruction error
* **PSNR** — image reconstruction quality
* **LPIPS** — perceptual similarity
* **Color-histogram distance** — difference in global color distributions

Evaluation samples are generated with a fixed seed to support consistent comparisons between adaptation configurations.

Example output:

```text
metrics.csv
```

contains per-sample prompt information and quantitative generation metrics.

---

## Repository Structure

```text
.
├── BaseModel/
│   └── Base diffusion-model related resources
│
├── ControlNet/
│   └── ControlNet implementation and configuration
│
├── LoRA_modified/
│   ├── multiseg_19prompts_batch1_shuffle_onlyCNLora.py
│   ├── multiseg_19prompts_batch1_shuffle_ctrlora.py
│   ├── multiseg_19prompts_batch1_shuffle_addlrSD.py
│   ├── quantitative_metrics.py
│   ├── LoRA_utils.py
│   ├── attention_lora.py
│   └── ...
│
├── PromptEngineering/
│   ├── Compositionality_TEST.py
│   ├── PromptTest_UMAP.py
│   └── empty_prompt_data_creation.py
│
├── SDFusion_bert/
│   └── bert_network/
│
├── TextEncoder_Finetuning/
│   ├── Training_sdfusion_bert.py
│   ├── Training_sdfusion_bert_lora.py
│   ├── Training_sdfusion_clip.py
│   ├── data_utils.py
│   ├── token_utils.py
│   └── vis_metrics.py
│
├── ctrlora/
│   └── CTRLora implementation and configurations
│
└── experiment_overfit/
    └── Diagnostic and overfitting experiments
```

---

## Main Experimental Scripts

For reproducing the principal adaptation experiments, the most relevant scripts are located in `LoRA_modified/`.

| Configuration        | Script                                            |
| -------------------- | ------------------------------------------------- |
| ControlNet LoRA only | `multiseg_19prompts_batch1_shuffle_onlyCNLora.py` |
| CTRLora              | `multiseg_19prompts_batch1_shuffle_ctrlora.py`    |
| CTRLora + U-Net LoRA | `multiseg_19prompts_batch1_shuffle_addlrSD.py`    |

Additional variants evaluate individual prompt categories such as styling, visibility, and color.

---

## Data Format

Training samples are organized through a `pairs.jsonl` metadata file and corresponding map images.

The scripts expect dataset directories containing training pairs and metadata, conceptually following:

```text
dataset/
├── ...
└── meta/
    └── pairs.jsonl
```

Each training example associates:

```text
input spatial condition
        +
natural-language styling prompt
        +
target stylized map
```

The exact dataset used for the experiments is not currently distributed with this repository.

---

## Pretrained Checkpoint

The experiments initialize the diffusion model from a pretrained map-generation checkpoint:

```text
BaseModel/Swisstopo.ckpt
```

or an equivalent local checkpoint path depending on the experiment.

Model checkpoints are not necessarily included in the Git repository and may need to be provided separately.

---

## Environment

The implementation is written in Python and uses PyTorch.

Core dependencies include:

```text
torch
torchvision
transformers
omegaconf
numpy
matplotlib
scikit-image
lpips
```

Some experiments additionally use memory-efficient attention through `xformers` when available.

A CUDA-enabled GPU is strongly recommended for training and inference.

---

## Running the Experiments

### 1. Clone the repository

```bash
git clone <repository>
cd Interactive-Stylization-of-Topographic-Maps-Using-Prompt-Driven-Diffusion-Models
```

### 2. Prepare the pretrained checkpoint

Provide the Stable Diffusion / map-generation checkpoint required by the selected experiment.

### 3. Prepare the dataset

Configure the dataset root containing the map pairs and `pairs.jsonl` metadata.

### 4. Update local paths

The current research scripts were developed in a local Windows environment and therefore contain absolute paths such as:

```python
ROOTDIR = r"..."
OUTDIR = r"..."
CKPT = r"..."
```

Before running an experiment, update these variables to match your local environment.

### 5. Run one of the main configurations

For example:

```bash
python LoRA_modified/multiseg_19prompts_batch1_shuffle_onlyCNLora.py
```

or:

```bash
python LoRA_modified/multiseg_19prompts_batch1_shuffle_ctrlora.py
```

or:

```bash
python LoRA_modified/multiseg_19prompts_batch1_shuffle_addlrSD.py
```

---

## Example Training Configuration

The principal experiments operate on `512 × 512` images with batch size 1 and use parameter-efficient fine-tuning rather than updating the complete diffusion model.

A representative configuration includes:

```text
Image size:           512 × 512
Batch size:           1
Maximum text length:  77 tokens
Optimizer:            AdamW
Text encoder:         bert-base-uncased
Text backbone:        frozen
Adaptation:           LoRA + trainable text adapter
```

Individual scripts contain the exact configuration used for each experiment.

---

## Evaluation Workflow

A typical experiment follows:

```text
                ┌─────────────────┐
                │ Topographic Map │
                └────────┬────────┘
                         │
                         ▼
                 Spatial Condition
                         │
                         │
Text Prompt ──► Text Encoder
     │                   │
     └──────────┬────────┘
                ▼
       Conditioned Diffusion
                │
                ▼
          Stylized Map
                │
                ▼
     Quantitative Evaluation
      MSE / PSNR / LPIPS /
      Color Distribution
```

The experimental design therefore evaluates both:

**structural preservation**, through spatially conditioned generation, and

**instruction following**, through controlled variation of cartographic prompts.

---

## Research Status

This repository contains **research and experimental code** rather than a production-ready software package.

Some folders record intermediate experiments conducted during model development, including alternative text encoders, overfitting diagnostics, prompt-representation experiments, and different LoRA configurations.

For understanding the final experimental pipeline, start with:

```text
LoRA_modified/
```

and in particular the three primary adaptation scripts listed above.

---

## Project Motivation

This project sits at the intersection of:

* **Cartography**
* **GeoAI**
* **Generative AI**
* **Human–map interaction**
* **Diffusion models**
* **Natural-language interfaces**

The broader objective is to investigate how generative models can support more intuitive interaction with geographic visualizations, allowing users to specify desired cartographic transformations through language rather than manually manipulating a large number of graphical parameters.

---

## Acknowledgements

This project builds upon pretrained diffusion-model and ControlNet architectures and investigates their adaptation to prompt-driven cartographic generation.

---

## Notes

The repository currently preserves the original experiment-oriented code structure. Paths, datasets, and pretrained checkpoints may therefore require local configuration before experiments can be reproduced.
