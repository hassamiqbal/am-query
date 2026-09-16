"""Assemble the delivery notebook and its complete numbered cell transcript."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
source = (ROOT/"am_query_runner.py").read_text()
parts = re.split(r"^# %% (.+)$",source,flags=re.M)
sections = [(parts[i].strip(),parts[i+1].strip()+"\n") for i in range(1,len(parts),2)]
cells = []
cell_no = 0
transcript = []

def md(text):
    cells.append({"cell_type":"markdown","id":f"md-{len(cells):03d}","metadata":{},
                  "source":text.splitlines(keepends=True)})

def code(title,body,explanation="",role="definitions"):
    global cell_no
    cell_no += 1
    heading = f"## Cell {cell_no:02d} — {title}\n\n{explanation}\n"
    md(heading)
    cells.append({"cell_type":"code","id":f"code-{cell_no:03d}",
                  "metadata":{"am_query_role":role,"am_query_cell":cell_no},
                  "execution_count":None,"outputs":[],"source":body.splitlines(keepends=True)})
    ast.parse(body)
    transcript.append(heading+"\n```python\n"+body.rstrip()+"\n```\n")

intro = """# AM-Query — one master GPU notebook

Prepared for Hassam Iqbal · 16 September 2026 · research experiment v0.1.1

**Goal:** learn to translate business questions into read-only SQLite queries, then
test whether execution-verified self-training improves a fixed pretrained model.
The starting model is `Qwen/Qwen2.5-Coder-7B-Instruct`; only its LoRA adapter is trained.
This is a model derivative, not a new foundation model or evidence of superintelligence.

The work is independent of the PhD and uses only generated business records. Its
commercial hypothesis is a privately deployable analytics assistant. The notebook
does not establish customer demand, novel research, production readiness or a valuation.
Confirm that your university/hospital GPU allocation and IP conditions permit this
non-PhD project and its intended commercial use before starting substantial training.

## Start here

1. Upload this file to Jupyter on the server exposing your NVIDIA L40S-48Q, then open it.
2. Run Cells 01–04. Keep `PROFILE = "smoke"` for the first run.
3. Run the definition cells, data checks and GPU preflight in order.
4. Run the training cell. It does actual fine-tuning after baseline evaluation.
5. Inspect `state.json` and the printed metrics. A candidate may be rejected; that is valid.
6. For the larger experiment, change Cell 04 to `PROFILE = "pilot"` and use its separate run folder.
7. Resume with the same configuration/folder after an allocation ends. Never run two workers on it.
8. Run the final-test cell only once model selection is finished. It freezes this experiment.

Long GPU work belongs in your institution's allocated job environment. Keeping a browser
tab open does not reserve a GPU. The companion Python runner supports the same experiment,
checkpoints, a STOP file and deadline. No job is submitted and no remote training is started
by this deliverable. Default cutoff: **28 September 2026, 23:59 Melbourne time**, with a
five-minute checkpoint reserve. This is a cooperative stop; your scheduler's limits prevail.

## What “self-improvement” means here

The current model samples SQL answers for fresh training tasks. A separate, fixed verifier
compares their results with independently checked reference queries over multiple generated
databases. Verified answers plus labelled seed replay train a candidate adapter. The candidate
replaces the incumbent only if it passes the fixed development gate. Otherwise the incumbent
is kept. Three unsuccessful rounds stop the pilot, even if GPU time remains.

The final evaluation compares the untrained adapter/base, supervised-only adapter and selected
champion on held-out schema names and held-out query compositions. The final data never drives
updates. These are small, synthetic, template-generated benchmarks; even a score of 1.00 would
not establish performance on customer databases. Repeated development selection can overfit.

## What was tested here

Sixteen CPU test cases pass, including 1,200 SQL results checked against an independent Python
implementation, query permissions/resource limits, cache recovery, promotion gates and freeze
controls. All numbered code cells were syntax-checked. The L40S training/inference path could
not be executed in this chat environment. Cell 17 is the required GPU smoke test.
"""
md(intro)

hardware = '''import sys
from pathlib import Path
import torch

print("Python:", sys.version.split()[0])
print("Kernel executable:", sys.executable)
print("Torch:", torch.__version__, "| Built CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select the Jupyter kernel on your allocated NVIDIA GPU server.")
print("Visible GPUs:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info(0)
print(f"GPU memory: {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
print("BF16 supported:", torch.cuda.is_bf16_supported())
'''
code("Confirm the GPU kernel",hardware,"Expected: CUDA True and your L40S. This cell makes no installations or training changes.","setup")

install = '''import importlib.metadata as metadata
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED = {
    "transformers": "4.48.3", "peft": "0.14.0", "accelerate": "1.3.0",
    "bitsandbytes": "0.45.2", "huggingface-hub": "0.28.1",
    "safetensors": "0.5.2", "tokenizers": "0.21.0",
}
def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

torch_version = installed_version("torch")
if torch_version is None:
    raise RuntimeError("Use the existing GPU kernel with working PyTorch first.")
changes = [f"{name}=={version}" for name, version in PINNED.items()
           if installed_version(name) != version]
if changes:
    previously_loaded = {"transformers", "peft", "accelerate", "bitsandbytes"} & set(sys.modules)
    with tempfile.TemporaryDirectory() as setup_dir:
        constraint = Path(setup_dir)/"keep_existing_torch.txt"
        constraint.write_text(f"torch=={torch_version}\\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "--constraint",
                        str(constraint), *changes], check=True)
    if previously_loaded:
        raise RuntimeError("Packages installed. Restart this kernel, then continue from Cell 03.")
    print("Packages installed; your existing Torch/CUDA build was constrained and retained.")
else:
    print("Dependency versions already match. Continue.")
'''
code("Install the project libraries",install,
     "Use a project kernel rather than an institution-wide shared Python installation. These versions target the supplied Torch 2.5.1/CUDA 12.1 stack. The installer constrains Torch to its existing build. If it asks for a kernel restart, restart and continue at Cell 03. No paid API key is needed.","setup")

code(*sections[0],explanation="Defines configuration, deadline checks, atomic records and a single-process lock. It does not start training.")
configuration = '''PROFILE = "smoke"  # First run: smoke. After it succeeds: pilot.
RUN_ROOT = f"am_query_runs/{PROFILE}"
SESSION_HOURS = 2.0 if PROFILE == "smoke" else 10.0
# SESSION_HOURS must fit inside your approved allocation, leaving time for startup.
CFG = Config.for_profile(PROFILE, root=RUN_ROOT, hours=SESSION_HOURS)

# Optional changes BEFORE a new experiment starts:
# CFG.micro_batch = 1  # Use this if pilot memory is insufficient.
# CFG.deadline = "2026-09-28T23:59:00+10:00"

print(json.dumps(dataclasses.asdict(CFG), indent=2))
print("Effective training batch:", CFG.micro_batch * CFG.accumulation)
'''
code("Select smoke test or pilot",configuration,
     "The smoke profile uses 48 seed examples, 8 supervised update steps and one small self-training round. Pilot uses 2,048 seed examples, 256 supervised steps, up to 24 rounds and plateau stopping. It does not promise to use the whole period or improve every round. Once a run starts, use a new folder for changed hyperparameters.","configuration")

explanations = {
    "Synthetic business schemas and independent task partitions":"Creates fresh two-table business tasks. Training, development and final partitions use separate schema vocabularies. Two query compositions are reserved for the final test.",
    "Read-only SQL verification with resource limits":"Checks outputs across multiple database variants and validates gold queries against an independent Python reference. Generated SQL cannot write records, attach files or load extensions. This local verifier is not a production database security boundary.",
    "GPU preflight and immutable model provenance":"Checks packages, available VRAM and disk; records the exact model commit for reproducibility. Keep at least 35 GiB free; a longer experiment may need more for adapters and checkpoints.",
    "QLoRA engine: one allocated GPU, assistant-token loss only":"Loads the 7B model in NF4 and trains LoRA weights in BF16. Prompts and padding are masked out of the loss. It uses only logical GPU 0 within your allocation.",
    "Durable training checkpoints and interruption recovery":"Saves adapter, optimizer and RNG state. Recoverable interruptions resume from a complete optimizer checkpoint. SIGKILL/power loss can lose work since the latest checkpoint, not the stored champion.",
    "Evaluation caches, paired selection gates, and honest uncertainty":"Greedy pass@1 is the selection metric. Promotion needs at least 2 percentage points improvement, a positive paired bootstrap lower bound and no large task-family regression. This gate is not a formal claim of generalization.",
    "Model-generated examples: retain only execution-verified answers":"Samples the current model, executes each answer only against generated databases, and retains verified candidates. The gold reference is used for scoring; it is not inserted as a self-generated training answer.",
    "Experiment controller: baseline, supervised seed, then verified self-training":"Runs the complete experiment, records the seed-data origin, replays some seed examples, retains the best model, stops on a plateau and exports its adapter.",
    "One frozen final test: baseline, supervised-only, and selected champion":"Defines final evaluation without running it. Calling it later freezes this run before viewing test results. It compares the base, supervised-only and chosen model. No further training may use the same frozen run.",
    "Export and a synthetic demonstration; nothing is automatically published":"Creates an adapter/model-card bundle and an optional generated-data demo. It does not merge or redistribute the pretrained base model, create a public repository or deploy an endpoint.",
}
for title,body in sections[1:-1]:
    code(title,body,explanations.get(title,""))

code("Check the data and verifier",'''data_self_check()
''',"Run this before downloading model weights. It uses the CPU only.","checks")

storage_cell = '''# Optional: paste an EXISTING storage directory assigned to your account here.
# Leave blank to keep a suitable current folder or inspect allocation variables.
STORAGE_BASE = ""

try:
    configure_storage(CFG, storage_base=STORAGE_BASE)
except RuntimeError:
    storage_diagnostics(CFG)
    raise
GPU_REPORT = preflight(CFG)
'''
code("Check the allocated GPU and disk",storage_cell,
     "If your home folder has too little space, enter your assigned project/scratch directory in STORAGE_BASE. This puts run outputs and the explicit model cache on that storage. Automatic selection considers only existing user-owned directories in AM_QUERY_STORAGE, SCRATCH, WORK or PROJECT. It never selects SLURM_TMPDIR automatically. Check your storage quota and retention policy. A filesystem can report free space beyond your individual quota. Existing run files are never silently abandoned. If no suitable path exists, the cell prints storage diagnostics and stops before downloading. Do not reduce the 35 GiB check to bypass the problem.","checks")

# A standalone replacement for people who already ran the original definition cells.
storage_section = dict(sections)["GPU preflight and immutable model provenance"]
hotfix_definitions = storage_section.split("\ndef prepare_manifest", 1)[0]
hotfix = '''"""Run in the existing notebook with: %run -i Cell_16_Storage_Fix.py"""
import dataclasses
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys

if "CFG" not in globals() or "Budget" not in globals():
    raise RuntimeError("Run your notebook's definition cells through Cell 15 first.")

''' + hotfix_definitions + '''
try:
    configure_storage(CFG, storage_base=globals().get("STORAGE_BASE", ""))
except RuntimeError:
    storage_diagnostics(CFG)
    raise
GPU_REPORT = preflight(CFG)
'''
(ROOT/"Cell_16_Storage_Fix.py").write_text(hotfix, encoding="utf-8")

code("Run or resume the experiment",'''RUN_STATE = run_experiment(CFG)
''',"This is the training cell. It downloads public base-model files on first use, runs a baseline, fine-tunes, verifies sampled answers and evaluates candidates. There is no claimed runtime: the smoke test measures whether your actual environment works. On a new session, run the definition cells and this cell with the SAME config/root. A time-limit pause is expected and resumable.","launch")

code("Inspect progress and decisions",'''root = Path(CFG.root).expanduser().resolve()
if (root/"state.json").exists():
    state = read_json(root/"state.json")
    print("Status:", state["status"])
    print("Best checkpoint:", state["champion"])
    print(json.dumps(state["history"], indent=2))
else:
    print("Initialization has not yet completed. Inspect the preceding cell output.")
''',"A rejected adapter is not a failure of the pipeline. Check whether improvements persist beyond supervised fine-tuning. Loss decreasing alone is not enough.","inspection")

code("Freeze and run the final holdout",'''RUN_FINAL_TEST = False  # Change to True only when this experiment is finished.
if RUN_FINAL_TEST:
    FINAL_REPORT = final_evaluation(CFG)
else:
    print("Final test remains unopened. Continue model selection first.")
''',"This is a deliberate research freeze, not a publishing action. Afterwards the run refuses further training. Finish it before your GPU access ends. An interrupted final test resumes on the exact same models.","final")

code("Export the selected adapter",'''root = Path(CFG.root).expanduser().resolve()
if (root/"state.json").exists():
    export_path = export_bundle(CFG)
    print("Keep the complete run folder and export folder on persistent approved storage.")
else:
    print("Complete initialization before exporting.")
''',"The bundle contains the selected LoRA adapter, tokenizer, model card, provenance and available metrics. If the base model remains best, the export records that honestly.","export")

code("Try one generated business question",'''RUN_DEMO = False
if RUN_DEMO:
    DEMO_RESULT = demo(CFG, index=108)
else:
    print("Set RUN_DEMO=True to reload the selected model and try a synthetic question.")
''',"This optional cell reloads the model and consumes allocated GPU time. It uses a generated database only.","demo")

cli_source = sections[-1][1]
launcher = '''# The companion am_query_runner.py contains the same definition cells.
# If you only downloaded the notebook, generate that script from it here.
import shlex

NOTEBOOK_PATH = Path("AM_Query_Master.ipynb")
if NOTEBOOK_PATH.exists():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    definitions = ["".join(c["source"]) for c in notebook["cells"]
                   if c["cell_type"] == "code" and c.get("metadata", {}).get("am_query_role") == "definitions"]
    cli = notebook["metadata"]["am_query_cli_source"]
    Path("am_query_runner.py").write_text("\\n\\n".join(definitions)+"\\n\\n"+cli, encoding="utf-8")
    print("Wrote am_query_runner.py from this notebook's definitions.")
elif not Path("am_query_runner.py").exists():
    raise FileNotFoundError("Place this notebook here under its original filename, or download am_query_runner.py.")

cmd = [sys.executable, str(Path("am_query_runner.py").resolve()),
       "--profile", CFG.profile, "--root", str(Path(CFG.root).resolve()),
       "--hours", str(CFG.session_hours)]
print("Run this command INSIDE your allocated GPU job/session:")
print(shlex.join(cmd))
print("Only one process may use a run folder. Stop notebook training before launching the script.")
'''
code("Prepare a terminal/batch runner",launcher,
     "For long sessions, use the script within your institution's scheduler or supported persistent GPU service. This cell prints the exact command for your kernel and paths. It does not submit a job or reserve GPU time. Keep the output path on persistent storage, not an allocation's temporary filesystem.","launcher")

closing = """## Running until 28 September

Start the pilot only after the smoke test runs on your L40S. The actual training rate,
memory usage and generation latency determine how many useful experiments fit.
Use this schedule as a plan, not a promise:

| Date | Work |
|---|---|
| 16–17 September | Smoke test, inspect several gold questions/answers, record baseline speed and memory |
| 18–22 September | Pilot and resumable verified self-training; inspect development decisions |
| 23–25 September | If warranted, separate runs for seed or learning-rate ablations; keep all runs and count every trial |
| 26–27 September | Freeze the chosen experiment, run final comparisons, document failures and limits |
| By 28 September | Export adapter/provenance/metrics; back up before the allocation expires |

For multiple runs, do not repeatedly choose a winner by the same final holdout. Pick the
run using development results, then perform one final comparison. Customer-specific or
independent benchmarks must be secured with appropriate data/licence rights before any
accuracy claim for a product. This starter does not include or claim results on Spider,
BIRD or customer data. Real schemas, ambiguous requests and incorrect business assumptions
are substantially harder than this generator.

## Stop and resume

Create an empty file named `STOP` inside the experiment folder to request a cooperative stop.
Delete that file only when you want to resume. You can also interrupt the kernel; it resumes
from the last durable checkpoint. A scheduler kill may lose work since that checkpoint.
The server controls allocation length: do not run a GPU workload on its login node or
resubmit jobs outside your approved quota. The code never bypasses scheduling limits.

For Slurm, run `sbatch` with your institution's actual GPU partition/account/resource syntax.
The script supports SIGTERM and SIGUSR1 for cooperative checkpointing; an advance signal
such as `--signal=USR1@300` is useful when your cluster supports it. The exact partition,
account and GPU request cannot be inferred from `NVIDIA L40S-48Q`. The terminal command in
Cell 22 works in an already allocated GPU session. On a managed persistent Jupyter server,
use its supported long-running job method; browser closure does not guarantee persistence.

## Interpreting the results

- `state.json`: current best checkpoint, round decisions and stop reason.
- `supervised_seed/`: a supervised-only control kept even if rejected by selection.
- `rounds/`: candidate adapters, sampled answers, verification evidence and train checkpoints.
- `final_report.json`: only appears after all frozen final comparisons finish.
- `export/`: local adapter bundle and model card; no automatic publication.

Improvement over the base can come entirely from supervised seed training. Improvement
over the supervised-only control is needed to support a benefit from the added self-training
loop. Even that needs replication, ablations, stronger baselines and independent datasets for
a research paper. There is no guarantee that any candidate is better than the starting model.

## Troubleshooting

| Symptom | Next action |
|---|---|
| CUDA is False | Select the GPU server/kernel with an active allocation |
| Package-version mismatch | Run Cell 02 in the project kernel; restart if instructed |
| Need 35 GiB disk space | In Cell 16, set STORAGE_BASE to existing assigned storage with enough free space and quota; this moves the output and model-cache destinations together |
| CUDA out of memory | Start a new run with `CFG.micro_batch=1` and `CFG.inference_batch=1`; keep batch accumulation |
| torchvision or binary import error | Save the exact traceback; fix that kernel's Torch/vision pairing rather than replacing CUDA blindly |
| Hugging Face/PyPI blocked by the institution | Ask for approved model/package staging; do not disable certificate verification |
| Deadline or STOP reached | Inspect the reason; resume only while authorised GPU time remains |
| “settings changed” | Resume original settings or use a new run folder |
| “plateau” | Stop spending compute on the same recipe; diagnose data/task difficulty before a fresh experiment |
| No candidate passes the gate | Keep the current best model and report the negative result |

## Commercial scope

A future offering could be a private analytics assistant with customer-specific schemas,
read-only service credentials, permission-aware query generation, cost/timeout limits,
human review, audit logs and support. Revenue depends on demonstrable customer value and
demand. A synthetic benchmark or a number of training hours does not establish a valuation.
Before release, review compute/IP conditions and applicable Apache-2.0 notices/attribution.
No pricing, revenue or publication outcome is promised by this notebook.

## Primary references checked on 16 September 2026

- [Base model and licence](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [PEFT 0.14 quantized adapter training](https://huggingface.co/docs/peft/v0.14.0/en/developer_guides/quantization)
- [Transformers 4.48 bitsandbytes guidance](https://huggingface.co/docs/transformers/v4.48.2/quantization/bitsandbytes)
- [PyTorch build versions](https://pytorch.org/get-started/previous-versions/)
- [Python SQLite authorizer and progress handler](https://docs.python.org/3.12/library/sqlite3.html)
- [Slurm batch jobs](https://slurm.schedmd.com/sbatch.html)
- [Hugging Face 0.28.1 cache locations](https://huggingface.co/docs/huggingface_hub/v0.28.1/guides/manage-cache)

The pinned packages are a compatibility target for the supplied environment, not a claim
that they are the newest releases. Use safetensors and `trust_remote_code=False`; the only
optimizer files loaded are this experiment's own local checkpoints.
"""
md(closing)

notebook = {"nbformat":4,"nbformat_minor":5,"cells":cells,
            "metadata":{"kernelspec":{"display_name":"Python 3 (GPU project)",
                                       "language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.12"},
                        "am_query_protocol":"am-query-0.1.0",
                        "am_query_cli_source":cli_source}}
notebook_path = ROOT/"AM_Query_Master.ipynb"
notebook_path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
(ROOT/"AM_Query_Cells.md").write_text(intro+"\n\n"+"\n\n".join(transcript)+"\n\n"+closing,encoding="utf-8")
readme = """# AM-Query delivery

Open `AM_Query_Master.ipynb` in the Jupyter GPU environment. It contains every code
cell and explanation. If you prefer your existing blank notebook, copy the numbered
code blocks from `AM_Query_Cells.md` into separate cells in order.

The first run uses `PROFILE = "smoke"`. After it succeeds, select `pilot` in Cell 04
and use the separate pilot folder. Cell 17 starts or resumes training. Cell 19 opens
the final test only if you deliberately set its flag. Cell 22 prepares the terminal runner.

`am_query_runner.py` has the same implementation for allocated batch jobs:

```bash
python am_query_runner.py --action check
python am_query_runner.py --profile smoke --hours 2
python am_query_runner.py --profile pilot --hours 10
python am_query_runner.py --profile pilot --action status
python am_query_runner.py --profile pilot --hours 10 --action final
```

Use your GPU project's Python executable; it may not be the shell's default `python`.
Install dependencies via Cell 02 first. These are sequential alternatives: do not start
notebook and terminal training simultaneously. The final command freezes the experiment;
run it only after selection is finished. Model downloads require institution-approved
access to Hugging Face. The default stop date is 28 September 2026 in Melbourne.

No GPU job has been launched for you. No trained weights or performance result is included.
Sixteen CPU test cases passed; L40S QLoRA execution still requires the included smoke test.

## Fixing Cell 16: insufficient disk space

Set `STORAGE_BASE` in Cell 16 to an existing directory assigned to your account with
at least 35 GiB free and sufficient individual/project quota. Leave it blank to inspect
the allocation's `AM_QUERY_STORAGE`, `SCRATCH`, `WORK` and `PROJECT` directories.
The run and model-cache destinations move together; existing experiment files are
not moved or deleted. Scratch retention policies vary: back up before expiry.

For a notebook already running the original definition cells, upload the companion
`Cell_16_Storage_Fix.py` into the notebook's current directory and replace Cell 16 with:

```python
STORAGE_BASE = ""  # Or your existing assigned project/scratch directory.
%run -i Cell_16_Storage_Fix.py
```

If it finds no suitable storage, paste the printed diagnostics into the support
conversation. Code cannot manufacture free disk space or increase a server quota.
Do not lower the threshold or delete unknown files. Changing HF_HOME alone does not
override the explicit model-cache path in this project.

For the terminal runner, use an explicit assigned run directory with `--root`.
Use the same directory on every resume. The updated preflight checks both locations.
"""
(ROOT/"AM_Query_README.md").write_text(readme,encoding="utf-8")
print(json.dumps({"code_cells":cell_no,"total_cells":len(cells),
                  "notebook_bytes":notebook_path.stat().st_size,
                  "runner_bytes":(ROOT/"am_query_runner.py").stat().st_size},indent=2))
