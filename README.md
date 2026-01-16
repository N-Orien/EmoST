# Generative Error Correction for Emotion-aware Speech-to-text Translation

This repository provides the code for the paper **“[Generative Error Correction for Emotion-aware Speech-to-text Translation](https://aclanthology.org/2025.findings-acl.1047)”**.

It contains a **two-stage** pipeline:

1. **N-best generation** using SeamlessM4T → `SeamlessM4T/`
2. **Generative Error Correction (GER)** fine-tuning + inference → `GER/`

---

## 1. N-best generation (SeamlessM4T/)

### Step 1: Create environment

Create a conda environment from `SeamlessM4T/requirements.yml`:

```bash
conda env create -f SeamlessM4T/requirements.yml
conda activate emost-seamlessm4t
```

---

### Step 2: Convert audio (mp4 → wav)

The original **MELD** data must be converted from **mp4** to **wav** before decoding.

1. Open `SeamlessM4T/convert_audio.sh`
2. Set `MELD_DIR` to your local MELD directory path
3. Run the script:

```bash
bash SeamlessM4T/convert_audio.sh
```

This should produce a directory of `.wav` files that will be used in decoding.

---

### Step 3: Generate N-best lists

Edit the following variables in:

`SeamlessM4T/translate_nbest_encoder_bmeld.py`

- `WAV_DIR`: directory containing wav files
- `TEXT_DIR`: directory containing the BMELD csv files
- `OUTPUT_DIR`: directory for outputs

Then run:

```bash
ls SeamlessM4T
python SeamlessM4T/translate_nbest_encoder_bmeld.py dev medium
```

- Replace `dev` with other dataset splits (e.g., `train`, `test`, ...)
- Replace `medium` with other SeamlessM4T model sizes supported by your setup

The script outputs a `.pt` file used in the next step.

---

### Step 4: Convert generated data into GER format

Convert the `.pt` file from Step 3 into the training/inference format used for GER fine-tuning:

```bash
python convert_data_llama-2/convert_data_conv1d.py $input_path $emotion_path $output_path
```

Where:

- `$input_path`: the `.pt` file generated in Step 3
- `$emotion_path`: the corresponding BMELD csv file containing emotion labels
- `$output_path`: the output path (converted data file)

You can replace `convert_data_conv1d.py` with other scripts under `convert_data_llama-2/`
to reproduce different experimental setups.

---

## 2. Generative Error Correction (GER/)

### Step 1: Create environment

Create a conda environment from `GER/requirements.yml`:

```bash
conda env create -f GER/requirements.yml
conda activate emost-ger
```

---

### Step 2: Fine-tuning

Run fine-tuning:

```bash
ls GER
bash GER/finetune.sh
```

To run other experimental setups, modify the python entrypoint used inside `GER/finetune.sh`,
switching to another script under `finetune/`.

---

### Step 3: Inference

Run inference:

```bash
bash GER/infer.sh
```

Similarly, for other experimental setups, modify the python entrypoint used inside `GER/infer.sh`,
switching to another script under `inference/`.

---

### Extract predictions for evaluation

After inference, extract predictions from the generated JSON file:

```bash
python GER/scripts/extract_data.py $json_file
```

- `$json_file`: the JSON file output by `infer.sh`

---

## Acknowledgements

This codebase is built upon **[GenTranslate](https://github.com/YUCHEN005/GenTranslate)**. We thank the authors for releasing their implementation.
