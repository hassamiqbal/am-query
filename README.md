# AM-Query

**Execution-verified SQL self-training on one allocated GPU.**

AM-Query is Hassam Iqbal's independent research prototype for translating business
questions into read-only SQLite queries. It starts with
[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
trains a LoRA adapter, checks generated answers against a fixed execution verifier,
and accepts candidate adapters only when they pass a development-set gate.

**Status:** notebook and CPU checks available. GPU training and model quality have
not been validated by this repository's maintainers. No trained model, production
service, customer-data result or general-intelligence claim is included.

## Start in Jupyter

1. Download [AM_Query_Master.ipynb](AM_Query_Master.ipynb) and open it in the
   Jupyter kernel attached to your allocated NVIDIA GPU.
2. Run Cells 01–04; retain `PROFILE = "smoke"` for the first run.
3. Run the definitions and CPU data check through Cell 15.
4. In Cell 16, select storage with sufficient free space and quota (see below).
5. Run Cell 17 to start or resume the smoke experiment.
6. Inspect Cell 18 before starting a separate pilot experiment.

The notebook has 22 numbered code cells and their explanations. The same cells are
available as a [copy-and-paste guide](AM_Query_Cells.md).

## Cell 16: disk-space fix

The 7B experiment requires **at least 35 GiB free on the storage used by the run
and model cache**, plus sufficient individual/project quota. A longer run may need
more. Free GPU memory and disk capacity are separate resources.

In the updated notebook, set `STORAGE_BASE` to an **existing directory assigned to
your account**. Leave it blank to retain adequate current storage or check
the explicitly configured `AM_QUERY_STORAGE` directory, or user-owned allocation
directories in `SCRATCH`, `WORK` and `PROJECT`. If none works, the cell prints
diagnostics and stops before downloading.

For an already-running older notebook, download
[Cell_16_Storage_Fix.py](Cell_16_Storage_Fix.py), upload it beside your notebook,
and replace Cell 16 with:

```python
STORAGE_BASE = ""  # Or your existing assigned project/scratch directory.
%run -i Cell_16_Storage_Fix.py
```

On Windows, diagnostics use Python's drive listing rather than the Unix `df`
command. The first check prints this kernel's Python executable and visible GPU.
If CUDA is unavailable, switch to the kernel on the allocated GPU server. The
location of the browser does not determine where the notebook kernel runs.

To use another Windows drive, first identify an existing folder assigned to you
on a drive with enough space and quota. Set `STORAGE_BASE` using a raw string such
as `r"D:\AM_Query"` **only if that folder actually exists and is yours to use**.
Another folder on the same full C: drive does not provide additional disk space.
The updated replacement also installs a Windows-compatible experiment lock.

The replacement uses the `CFG` and definitions already in your kernel. It checks
both run and cache paths, including symlinks. It does not lower the disk threshold,
delete files or silently switch away from an existing experiment. Changing
`HF_HOME` alone does not override the explicit `cache_dir` used here; see the
[Hugging Face cache documentation](https://huggingface.co/docs/huggingface_hub/v0.28.1/guides/manage-cache).

Filesystem free space is not a per-user quota measurement. Check your assigned
storage's retention policy; allocation-local storage may disappear. If the server
has no suitable storage, request additional capacity before continuing.

## Terminal / allocated batch job

Install dependencies using Cell 02 in your project kernel first; it retains your
working Torch build. The recorded compatibility target is Python 3.12,
Torch 2.5.1+cu121 and an L40S-class GPU. The preflight measures actual free VRAM.

```bash
python am_query_runner.py --action check
python am_query_runner.py --profile smoke --root /YOUR/ASSIGNED/STORAGE/am_query_runs/smoke --hours 2
```

Replace the placeholder path with your real assigned directory. Run inside an
approved GPU allocation and use that kernel's Python executable. Cell 22 prints
the precise command for the notebook's current configuration. Use the **same root**
when resuming. Do not run the notebook and runner on one experiment simultaneously.

The default deadline is **28 September 2026, 23:59 Melbourne time**; each session
also has a shorter wall-time budget and a five-minute checkpoint reserve. The
institution's scheduler controls actual allocation. The code does not reserve,
extend or automatically resubmit GPU jobs.

## Experiment design

- Procedural business records; no customer, patient or PhD data.
- Frozen pretrained base with NF4 loading and trainable BF16 LoRA adapters.
- Independent Python reference checks for the generated SQL tasks.
- Seed-supervised control, verified self-training and retained incumbent.
- Development gate with minimum gain, paired bootstrap and family regression checks.
- Separate schema vocabularies and held-out query compositions.
- One final evaluation that freezes the selected experiment against further training.
- Local adapter exports, provenance, resume checkpoints and a cooperative STOP file.

The verifier runs generated queries against generated in-memory databases with
query restrictions and resource limits. It is **not a production SQL sandbox**.
Template-generated tasks do not establish reliability on real customer databases.
Candidates may fail to improve; negative results should be retained and reported.

## Validation

```bash
python -m unittest -v test_am_query
python validate_notebook.py
```

Twenty-one CPU tests cover 1,200 independent SQL checks, query restrictions, stop/resume,
selection/freeze controls, low disk space, explicit and assigned storage, existing
run preservation, a cache linked to another full filesystem, Windows drive checks,
kernel selection and native process locking. CI repeats these checks on Linux and
Windows without downloading models. The symlink test is skipped if OS permissions
do not allow creating a test symlink. GPU loading, training, throughput and accuracy
remain to be measured using the smoke test.

## Files

| File | Purpose |
|---|---|
| `AM_Query_Master.ipynb` | Complete beginner notebook |
| `AM_Query_Cells.md` | Numbered cells and explanations |
| `Cell_16_Storage_Fix.py` | Replacement for Cell 16 in an existing session |
| `am_query_runner.py` | Equivalent allocated-job runner |
| `test_am_query.py` | CPU regression tests |
| `validate_notebook.py` | Notebook syntax, clean-output and runner consistency checks |
| `build_notebook.py` | Rebuild notebook and guide from the implementation |
| `requirements.txt` | Pinned project libraries; keep the existing Torch build |
| `TEST_REPORT.md` | Validation scope and limitations |

## Publication and commercial work

This public repository contains source and documentation only. Training folders,
model caches, checkpoints, credentials and local datasets are excluded. Nothing in
the training code automatically publishes adapters or opens a network service.

A potential product is a private business analytics assistant. Commercial value
must be demonstrated with real requirements, permission-aware access, independent
evaluation, deployment controls and paying customers. Training alone does not
establish revenue, novelty or a valuation.

No project source-code licence has been selected yet. See [RIGHTS.md](RIGHTS.md).
The upstream model and dependencies retain their own licences. Confirm compute
and intellectual-property terms for institutional resources before commercial work.

Maintainer: [Hassam Iqbal](https://github.com/hassamiqbal).
