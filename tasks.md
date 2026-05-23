# MotiveAi — Sprint Tracker

**Deadline:** 24 May 2026, 23:59 WAT  
**Updated:** 21 May 2026, 8:15 PM WAT  
**Days remaining:** 3  
**Training:** Friend handles LoRA — we orchestrate everything else

---

## ✅ COMPLETED

- [x] 1. Repo folder structure — matches PRD Section 8
- [x] 2. Dockerfile + docker-compose.yml — **VERIFIED WORKING** ✅
- [x] 3. FastAPI skeleton with stub endpoints returning 200
- [x] 4. Pydantic schemas matching PRD API contracts (Section 2)
- [x] 5. `pidgin_vader.py` — real code, 9 Pidgin overrides
- [x] 6. `pidgin_vader.json` + `pidgin_canonical.json` — lexicon data files
- [x] 7. `levenshtein_normalizer.py` — phonological weighted distance
- [x] 8. `naija_bert.py` — embed/warmup/mean_pool wrapper
- [x] 9. `apg4recsim.py` — full profile extraction + to_prompt_string()
- [x] 10. `sasrec.py` — temporal attention with recency decay
- [x] 11. `lasso.py` — cross-domain simulation, taste dims, pseudo-interactions
- [x] 12. `retriever.py` — ChromaDB vector retrieval (connect/retrieve/upsert)
- [x] 13. `llamarec.py` — structure done (scoring is stub `[1.0]*n`)
- [x] 14. `config.py` — settings with env vars
- [x] 15. Prompt templates — `simulate_review.txt` + `reranker.txt`
- [x] 16. `preprocess_amazon.py` + `preprocess_yelp.py`
- [x] 17. `build_chroma_index.py` + `sample_test_set.py`
- [x] 18. `eval_task_a.py` + `eval_task_b.py` — metric scripts
- [x] 19. 5 test files with 25 test functions
- [x] 20. CI workflow, .gitignore
- [x] 21. NaijaRec → MotiveAi rename
- [x] 22. Docker `docker compose up` — both containers boot, endpoints return JSON ✅
- [x] 23. Split requirements.txt (runtime) + requirements-dev.txt (dev-only)
- [x] 24. CPU-only PyTorch in Docker (200MB vs 779MB)
- [x] 25. Fix docker-compose ChromaDB port (8001:8000) + remove broken healthcheck
- [x] 26. Fix config.py chroma_port → 8000

---

## 🔨 NEXT UP — Wire Real Pipelines (Priority Order)

### Block 1: Wire `/simulate-review` (Task A) — ✅ DONE
**Goal:** Replace the stub with real code that returns a Pidgin review + rating.
- [x] Take the user history from the request
- [x] Run it through Levenshtein → Pidgin VADER → APG4RecSim → SASRec
- [x] Build the prompt using `simulate_review.txt`
- [x] Placeholder function returns a template-based Pidgin review (no LLM needed yet)
- [ ] **Future:** When friend delivers LoRA weights → swap ONE function to use the real model

### Block 2: Wire `/recommend` (Task B) — ✅ DONE
**Goal:** Replace the stub with real code that returns ranked recommendations.
- [x] Detect cold start
- [x] Run Lasso to generate pseudo-interactions
- [x] Embed the user profile with NaijaBERT
- [x] Query ChromaDB for similar items
- [x] Rank results using cosine similarity (no LLM needed yet)
- [ ] **Future:** When friend delivers LoRA weights → swap ONE function for real reranking

### Block 3: Wire `main.py` startup — ✅ DONE
**Goal:** Connect models on boot so they are ready before requests come in.
- [x] Uncomment `naija_bert.warmup()` and `retriever.connect()`

### Block 4: Populate ChromaDB with sample data — ✅ DATA GENERATED
**Goal:** ChromaDB has real Yelp restaurants so `/recommend` returns real items.
- [x] Generate 30,000 simulated Jumia reviews (Source Domain)
- [x] Generate 2,000 simulated Yelp restaurants (Target Domain)
- [x] Move clean datasets to `data/test/` and update `.gitignore`
- [ ] Run `python data/build_chroma_index.py --input data/test/yelp_simulated_reviews.jsonl` to push data into DB

### Block D: Missing Tests (3 files)
- [ ] D1. Create `test_retriever.py` — connect, retrieve, upsert, collection_size
- [ ] D2. Create `test_sasrec.py` — encode_sequence, recency_weights, empty input
- [ ] D3. Create `test_llamarec.py` — rerank, build_prompt, softmax_normalize

### Block E: Friend's LoRA Integration (when weights arrive)
- [ ] E1. Friend delivers LoRA adapter weights → `data/models/`
- [ ] E2. Add `peft` back to `requirements.txt`
- [ ] E3. Implement real `llamarec.py` `_score_batch()` with forward pass
- [ ] E4. Implement real LLM generation in simulate pipeline
- [ ] E5. Re-run Docker build to verify

---

## 📅 DAY 3 (22 May) — Metrics & Polish

- [ ] F1. Run `sample_test_set.py` → generate `data/test/task_a_test.jsonl` + `task_b_test.jsonl`
- [ ] F2. Run `eval_task_a.py` — first ROUGE-L, BERTScore, RMSE numbers
- [ ] F3. Run `eval_task_b.py` — first NDCG@10, Hit Rate numbers
- [ ] F4. Profile `_score_batch()` latency — must be < 2s for 100 candidates
- [ ] F5. Fix Dockerfile — add NaijaBERT pre-download at build time (PRD 6.1)

## 📅 DAY 4 (23-24 May) — Final Submission

- [ ] G1. Swap LoRA weights if friend delivers them
- [ ] G2. Run all 4 ablation notebooks
- [ ] G3. Save results → `metrics/ablation_results.json`
- [ ] G4. Final `docker compose up --build` on CLEAN machine
- [ ] G5. README final pass — verify curl examples work
- [ ] G6. Code cleanup — type hints, docstrings, remove debug prints
- [ ] G7. **SUBMIT** — GitHub repo public + solution paper + model link

---

## Exit Criteria

| Day | Gate |
|---|---|
| Day 2 (today) | Both endpoints return real non-stub responses. Docker boots. |
| Day 3 | ChromaDB populated. Metrics exist. Latency < 2s. |
| Day 4 | Clean docker boot. Submitted before 23:59 WAT. |
