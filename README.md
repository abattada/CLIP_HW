# CLIP Homework Guide

This project ships two Jupyter notebooks—`Task1.ipynb` and `Task2.ipynb`. They demonstrate zero-shot evaluation of CLIP on flower and bird datasets as well as visual encoder linear probing and LoRA fine-tuning workflows. The guide below explains how to prepare the environment, manage data and models, and reproduce the experiments in the notebooks.

## Environment setup
1. Use Python 3.10 or newer and create a virtual environment (`venv` or `conda` recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
   Open `Task1.ipynb` or `Task2.ipynb` in your browser and execute the cells sequentially.

## Data layout
- `./data`: Datasets and model artifacts are stored here.
  - `Task1.ipynb` and `Task2.ipynb` automatically download Flowers102 and CUB-200-2011, caching everything required by torchvision and 🤗 Datasets inside this folder. The `cat_to_name.json` mapping for Flowers102 should also live here. 
  - Task 2 training outputs—comparison figures and LoRA artifacts—are written into subdirectories of `./data`, such as `linear_birds_result.png` and `fine_flowers_result.png`. 
- `./photo`: Personal photos used for qualitative checks belong here. The notebooks show how to load these files and convert them into CLIP-compatible `pixel_values`. 【F:Ta

## Task 1: Zero-shot classification workflow
1. The notebook loads the pretrained `openai/clip-vit-large-patch14` model and processor, then builds flower and bird evaluation dataloaders. 
2. It evaluates batches with multiple prompt templates. The `zeroshot_eval_all_prompts` helper caches all prompt text embeddings and reports accuracy for each template. 
3. Running `zeroshot_eval_all_prompts` produces accuracy tables for Flowers102 and CUB-200-2011 so you can compare prompt effectiveness.
4. Use `visualize_multi_prompt_compact_indices` to render image grids with predictions under each prompt. You can mix samples from both datasets or custom image collections.
5. To evaluate your own photos, wrap the `./photo` contents with `InlineImageDataset`. The resulting dataset works with the visualization and inference helpers. 

## Task 2: Visual encoder fine-tuning workflow
1. The notebook downloads and preprocesses Flowers102 and CUB-200-2011 train/val/test splits, exposing dataloaders for each partition.
2. `DATASET_REGISTRY` centralizes dataset metadata, class names, and prompt templates, enabling consistent comparisons and visualizations.
3. After linear probing or LoRA fine-tuning, call `register_trained_variant` to log the configuration, then `save_prediction_comparison_grid` to export comparison figures. The notebook demonstrates both flower and bird runs for linear probes and LoRA. 
4. When you finish training and need to reclaim GPU memory, run `free_gpu_safe()`. It transfers registered models to the CPU, clears temporary variables, and helps prevent out-of-memory issues during long interactive sessions. 

## Model persistence
- `save_trained_variant_to_disk` and `save_training_artifacts` are built-in helpers for persisting and restoring models under `./saved_variants/<variant_name>/`. The backbone is exported in Hugging Face format, the head weights go into `head_state.pth`, and `meta.json` records dataset metadata and notes. Loading automatically infers the output dimension and recreates the linear layer. 
- After finishing fine-tuning, call `register_trained_variant(..., save_to_disk=True)` so the notebook tracks the run. Follow up with `save_trained_variant_to_disk` to write artifacts, and use `save_training_artifacts` when you need to reload them to CPU or GPU.

## Releasing GPU resources
- Before starting a new experiment or after a long session, execute `free_gpu_safe()`. The helper moves registered models to the CPU, deletes common temporary variables, and invokes `torch.cuda.empty_cache()` plus `torch.cuda.synchronize()`, making it convenient to free memory inside Jupyter notebooks.

## Testing personal photos
1. Drop JPEG/PNG images into `./photo`, optionally organizing them into subfolders.
2. Load them in the notebook with `glob("./photo/*.jpeg")` or `InlineImageDataset("./photo")`, then convert them using the existing CLIP preprocessing (via `clip_image_transform`) to obtain `(3, 224, 224)` inputs. 
3. Use `visualize_multi_prompt_compact_indices` or your own inference routines to inspect predictions on real-world photos.

Follow the steps above to reproduce zero-shot evaluation, linear probing, and LoRA fine-tuning, while taking advantage of the data/model management utilities for your custom experiments.
