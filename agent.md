# MotiveAi Architecture & Implementation Plan

This document outlines the system architecture and implementation strategy for the MotiveAi project, strictly adhering to the Product Requirements Document (PRD) and repository structure.

## 1. System Architecture Overview

MotiveAi is a containerized LLM agent system exposing two FastAPI endpoints.

### Task A: User Modeling (`POST /simulate-review`)
1. **Pidgin VADER + Levenshtein normalizer**: Pre-processes review text and corrects semantic shifts in Nigerian Pidgin.
2. **NaijaBERT embedding**: Embeds the text using a 110M parameter model fine-tuned on Nigerian Pidgin + English.
3. **APG4RecSim profile extraction**: Extracts structured user profile traits (price sensitivity, quality threshold, etc.).
4. **SASRec temporal attention**: Models recency-weighted preference drift.
5. **LoRA LLM conditioned generation**: Generates a culturally authentic review string.
6. **Rating prediction + structured output**: Combines signals to predict the star rating and outputs the generated review.

### Task B: Recommendation (`POST /recommend`)
1. **Cold-start detection**: Identifies if the user lacks target domain history.
2. **Lasso cross-domain pseudo-interactions**: Generates pseudo-interactions using source domain taste dimensions.
3. **NaijaBERT query embedding**: Combines user profile and pseudo-interactions into a query vector.
4. **ChromaDB ANN retrieval**: Stage 1 retrieval, returning the top-100 candidates from the vector database.
5. **LlamaRec verbalizer reranker**: Stage 2 reranking using the LLM's final hidden layer probabilities.
6. **Top-K ranked output**: Returns the ranked item list along with a 1-2 sentence reasoning.

---

## 2. ⚠️ Model Training Hand-off (FRIEND'S TASK) ⚠️

> **IMPORTANT:** You will **NOT** be doing the model training. Your friend is in charge of this step!

### **When to call your friend:**
You should call your friend **at the start of Day 2** (or as soon as you finish the data preprocessing).

### **What your friend needs to do:**
- **Task:** Run the LoRA fine-tuning job on the **NaijaSenti** dataset to train the culturally-authentic review generation model.
- **Output Needed:** They need to provide you with the LoRA adapter weights.
- **Where to put it:** Once they give you the model weights, place them in the `data/models/` directory so the FastAPI app can load them via `LLM.load()`.

---

## 3. Core Component Specifications

The source code is organized strictly as follows in `src/`:

### `src/app/` (API Layer)
- `main.py`: FastAPI app factory, includes startup events to warm up models (NaijaBERT, LLM) and connect to ChromaDB.
- `routes/simulate.py`: Task A handler.
- `routes/recommend.py`: Task B handler.
- `models/schemas.py`: Pydantic schemas.

### `src/nlp/` (Natural Language Processing)
- `pidgin_vader.py`: Sentiment scoring with Nigerian Pidgin overrides (e.g., "ginger", "hammer", "mumu").
- `levenshtein_normalizer.py`: Phonological spelling normalization to correct Pidgin variants.
- `naija_bert.py`: Batch embedding generation.

### `src/agent/` (Agent Mechanics)
- `apg4recsim.py`: Extracts UserProfiles containing `price_sensitivity`, `quality_threshold`, etc.
- `sasrec.py`: Recency-weighted attention mechanism for sequence encoding.
- `llamarec.py`: The verbalizer reranker scoring candidates based on logit positions without autoregressive generation.
- `lasso.py`: Maps source domain taste dimensions to target domain attributes for zero-shot cold start.
- `retriever.py`: ChromaDB integration for Top-100 ANN retrieval.

## 4. Tech Stack Check
- **Language**: Python 3.11
- **API**: FastAPI
- **Vector DB**: ChromaDB
- **Container**: Docker + docker-compose
- **Base LLM**: Llama 3 / Mistral with LoRA weights
- **Embeddings**: NaijaBERT
