# Construction and Application of the Traditional Chinese Medicine Knowledge Graph Using ChatGLM3-6B by Bo Zhang, Ruifang Li et al.

This repository consists of two parts:
1. **Fine-tuning** the open-source ChatGLM3-6B model with LoRA for TCM-related question-and-answer tasks.  
2. **Building** a TCM Knowledge Graph by extracting key entities  from model outputs via a specialized TCM entity recognition model (TCMER), then storing these in a graph database.

It is based on the article "**Construction and Application of Traditional Chinese Medicine Knowledge Graph Based on Large Language Model**" by Bo Zhang and others. This repository includes Python code and sample data for replicating both the fine-tuning and knowledge graph construction processes.

---

## 1. Download the pre-trained ChatGLM3-6B model

Instead of directly storing large files in this repository, we provide a download link.  
**Please download the ChatGLM3-6B weights** from the Hugging Face model hub:

[**THUDM/chatglm3-6b-128k**](https://huggingface.co/THUDM/chatglm3-6b-128k)

Place the following files into `./base_model/chatglm3-6b/` locally:

```
config.json
pytorch_model-00001-of-00007.bin
pytorch_model-00002-of-00007.bin
pytorch_model-00003-of-00007.bin
pytorch_model-00004-of-00007.bin
pytorch_model-00005-of-00007.bin
pytorch_model-00006-of-00007.bin
pytorch_model-00007-of-00007.bin
tokenizer.model
tokenizer_config.json
```
---

## 2. Fine-tuning ChatGLM3-6B with LoRA on Alibaba Cloud DSW (via LAMMA)

Below is a concise workflow describing how we loaded **LAMMA** within the **Alibaba Cloud DSW** platform to conduct LoRA fine-tuning of **ChatGLM3-6B**, using the hyperparameters specified in `config_lora.json`:

1. **Prepare the DSW Workspace**  
   - In **Alibaba Cloud DSW**, open or create a GPU workspace (e.g., **NVIDIA A100**, 40 GB VRAM, 48-core CPU, 256 GB RAM).  
   - Ensure Python 3.8+ is available. You may create a new Conda environment or install packages via `requirements.txt`.

2. **Upload the Base Model (ChatGLM3-6B)**  
   - Inside the DSW workspace, upload all **ChatGLM3-6B** weights into `./base_model/chatglm3-6b/`.


3.**Invoke LAMMA and Configure LoRA**  
   - Launch or connect to the **LAMMA** environment via the DSW terminal (methods vary by setup).  
   - Navigate to the LoRA scripts directory:
     ```bash
     cd finetune_scripts
     ```
   - Open `config_lora.json` and confirm the hyperparameters match those in your paper .

4.**Run the LoRA Training**  
   - Start fine-tuning using the parameters from `config_lora.json`:
     
   - This will:
     1. Load the base ChatGLM3-6B from `../base_model/chatglm3-6b/`.
     2. Apply LoRA using your specified hyperparameters (from `config_lora.json`).
     3. Train for the configured number of epochs.
     4. Save the LoRA-adapted weights and tokenizer in `./lora_outputs/`.

5.**Monitor and Finalize**  
   - Observe GPU usage, logs, or console output in DSW.  
   - Once training completes, you can download or directly use the **LoRA** weights from `lora_outputs/` in subsequent tasks (e.g., TCM entity recognition, knowledge graph construction).

**Note**  
- If GPU memory is tight, reduce `batch_size` or enable FP16 in `config_lora.json`.  
- For large-scale data, LAMMA may support incremental training or data streaming.  
## 3. Building the Entity Recognition Model

This section explains how to train a Traditional Chinese Medicine (TCM) entity recognition (TCMER) model, identifying crucial entities—such as herbs, symptoms, and treatments—from TCM texts.

### 3.1 Directory Structure

- **data/**: Contains example NER-annotated datasets (`example_train.json`, `example_dev.json`).  
- **tcmer_model.py**: Defines the TCMER architecture (BERT + BiLSTM + CNN + Self-Attention).  
- **train_tcmer.py**: Main script to train or fine-tune the model.

### 3.2 Training Steps

1. **Install Dependencies**  
   ```bash
   cd entity_recognition
   pip install -r requirements.txt
   ```
---
2. **Prepare Training Data** 
Prepare Traditional Chinese Medicine texts and label them in NER format, then place the files in the `data/` directory. Example files are `example_train.json` and `example_dev.json`.

---
3. **Start Training** 
Run `train_tcmer.py` with the specified parameters (model architecture, batch size, learning rate, etc.). Upon completion, an entity recognition model (e.g., `tcmer_model.bin`) will be generated.

---
**Note**  
- This entity recognition model combines **BERT, BiLSTM, CNN**, and **Self-Attention** to extract TCM-related entities from text.  
- Entity recognition is critical for subsequent knowledge graph construction: the model identifies and classifies entities like **“herbs,” “symptoms,”** and **“treatments”** within the generated text, providing the foundation for creating triples.  
## 4. Constructing the Knowledge Graph

In this section, we utilize recognized entities from the TCMER model to build a **Traditional Chinese Medicine Knowledge Graph** in Neo4j.

### 4.1 Directory Structure

- **build_kg.py**: Reads LLM outputs and recognized entity data, then creates Entity-Relationship-Entity triple for insertion into Neo4j.  
- **example_kg.cypher**: An optional Cypher script to query or visualize the TCM knowledge graph in Neo4j.

### 4.2 Building the Graph

```bash
cd knowledge_graph
python build_kg.py \
  --llm_output_file llm_results.json \
  --tcmer_model_path ../entity_recognition/tcmer_model.bin \
  --tokenizer_name bert-base-chinese \
  --num_labels 7 \
  --neo4j_url bolt://localhost:7687 \
  --neo4j_user neo4j \
  --neo4j_password yourpassword
  ```
### 4.3  Querying in Neo4j

- After running build_kg.py, open your Neo4j Browser and execute a sample query:
```bash
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 50
   ```
- You can then explore the TCM knowledge graph, see which herbs treat which symptoms, and perform more advanced queries or visualizations.

**Note**
- Customize build_kg.py to handle different entity types or relationship rules.  
- Update example_kg.cypher with queries specific to your TCM domain requirements.

If you need to access the dataset or other code, please contact Bo Zhang（ 2681308146@qq.com ).