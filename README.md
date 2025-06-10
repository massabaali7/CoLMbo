
# CoLMbo: A Speaker Language Model for Descriptive Profiling

CoLMbo is a **Speaker Language Model (SLM)** designed to go beyond traditional speaker recognition. While most systems stop at identifying “who” the speaker is, CoLMbo answers **“what is this speaker like?”** by generating **context-rich, descriptive captions** from speaker embeddings including **gender, age, personality**, and **dialect**.

---

## Abstract

> Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner.
>
> This paper introduces **CoLMbo**, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to dynamically adapt to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach enhances traditional speaker profiling and performs well in **zero-shot scenarios** across diverse datasets, marking a significant advancement in the field of speaker recognition.

---

## 🗂️ Project Structure

- `run_single_example.py` – script to run CoLMbo on a speaker audio file and a prompt.
- `wrapper.py` – core wrapper class for loading models and generating text.
- `config.yaml` – configuration file to control model settings, prompt, and file paths.

---

## 🚀 Running the Example

To run a single inference example using CoLMbo:

```bash
python run_single_example.py --config config.yaml
```

Inside `config.yaml`, make sure to:
- Set the correct path to your **audio waveform**
- Provide your custom **prompt** (e.g., `"describe the speaker"`)
- Choose the device (`cuda` or `cpu`)

---

## 📥 Checkpoints

You can download pretrained checkpoints for:

- `ECAPA Speaker Encoder`  
- `PDAF Speaker Encoder`  
- `SID_LM`  
- `Mapper_LM`

from this Google Drive:  
🔗 [Download CoLMbo Checkpoints](https://drive.google.com/drive/folders/1OzYxobJ6w1RMZlPHVkX20xcUgSQQPlMC)

---

## 🗃️ Datasets

We use the **TEARS dataset** for evaluation and training. You can get it here:

- 📦 [TEARS Dataset on HuggingFace](https://huggingface.co/datasets/cmu-mlsp/TEARS)

Audio files used:
- [EARS Dataset](https://github.com/facebookresearch/ears_dataset/tree/main)
- [TIMIT Dataset (LDC93S1)](https://catalog.ldc.upenn.edu/LDC93S1)

---

## 💡 Example Prompt Usage

You can run prompts like:

```python
prompts = [
    "Describe the speaker's gender.",
    "What is the speaker's age?",
    "What is the speaker's gender inferred from their speech characteristics?",
    "What is the speaker's nationality?"
]
```

CoLMbo will return natural language responses conditioned on speaker voice.

---

## 📌 Citation

If you find this useful in your research, please cite us:

```bibtex
@misc{colmbopaper,
  title={CoLMbo: A Speaker Language Model for Descriptive Profiling},
  author={Massa Baali et al.},
  year={2025},
  archivePrefix={arXiv},
  eprint={XXXX},
  primaryClass={cs.CL}
}
```
