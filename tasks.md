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

### Block A: Wire `/simulate-review` (Task A — earns ROUGE + BERTScore + RMSE points)
**Goal:** When judge POSTs user history + target item, return a REAL Pidgin review + rating

- [ ] A1. **Wire `main.py` startup** — uncomment: `naija_bert.warmup()`, `retriever.connect()`, `pidgin_vader.load()`, `levenshtein.warmup_cache()`
- [ ] A2. **Wire `simulate.py` pipeline** — connect the 6 components in order:
  1. Normalize review texts (Levenshtein → Pidgin VADER)
  2. Embed normalized texts (NaijaBERT)
  3. Extract user profile (APG4RecSim)
  4. Temporal attention weighting (SASRec)
  5. Build LLM prompt from profile + target item (using `simulate_review.txt`)
  6. **PLACEHOLDER**: Return prompt-based output (until LoRA model arrives)
  7. Parse rating from output
- [ ] A3. **Create LLM placeholder** — a function that takes a prompt and returns a reasonable review WITHOUT a real model. This keeps the pipeline working end-to-end. When friend delivers LoRA weights, we swap this one function.

### Block B: Wire `/recommend` (Task B — earns 30+25+20 = 75 rubric points!)
**Goal:** When judge POSTs user persona, return ranked Yelp restaurant recommendations

- [ ] B1. **Wire `recommend.py` pipeline** — connect the components:
  1. `lasso.detect_cold_start(target_history)`
  2. If cold-start: `lasso.simulate(source_history, target_domain)`
  3. Combine real + pseudo history, embed with NaijaBERT
  4. `retriever.retrieve(query_vec, top_k=100)`
  5. `verbalizer.rerank(profile, candidates)` (uses embedding similarity for now)
  6. Return top-K with reasoning strings
- [ ] B2. **Make `llamarec.py` work without LLM** — use embedding cosine similarity as the scoring function (PRD's fallback strategy). When LoRA model arrives, swap in real logit extraction.

### Block C: Data & ChromaDB Population
**Goal:** ChromaDB has real Yelp restaurants so `/recommend` returns real items

- [ ] C1. **Extract Amazon zip** → `data/raw/` (you downloaded it)
- [ ] C2. **Extract Yelp zip** → `data/raw/`
- [ ] C3. Run `python data/preprocess_yelp.py --business-file data/raw/yelp_academic_dataset_business.json`
- [ ] C4. Run `python data/preprocess_amazon.py --electronics-file data/raw/amazon_electronics.jsonl --movies-file data/raw/amazon_movies.jsonl`
- [ ] C5. Run `python data/build_chroma_index.py` — populates ChromaDB with embedded Yelp businesses

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
