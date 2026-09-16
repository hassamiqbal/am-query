# Validation record

2026-09-16. Sixteen CPU test cases passed via `python -m unittest test_am_query -v`.

- 1,200 generated SQL answers matched an independent Python implementation across all fourteen task families.
- Dangerous queries, writes, multiple statements, file-extension loading, non-finite results and excessive computation were rejected.
- Evaluation and sampling caches resumed without duplicating completed work.
- Changed models or training settings invalidated incompatible cached state.
- Promotion rejected ties and excessive per-family regressions.
- Final freeze prevented further training.
- All 22 notebook code cells parsed successfully; its generated runner matches the delivered runner's AST.
- A filesystem with 6.47 GiB free was rejected before creating output/cache directories.
- Explicit assigned storage relocates both run and model-cache destinations without changing the 35 GiB minimum.
- User-owned allocation storage can be selected automatically; existing run records are preserved.
- A model cache symlinked to a separate full filesystem is rejected independently of output-folder capacity.

Storage tests use simulated filesystem capacities. They do not measure the user's remote
server or its personal/project quota. `validate_notebook.py` confirms the public notebook
has no execution outputs and contains the same definitions and CLI as the runner.

The CUDA, bitsandbytes, Transformers and QLoRA training path was NOT executed in this environment.
No base or fine-tuned model accuracy is reported. Run the notebook smoke profile on the allocated L40S before the pilot.
