# AM-Query delivery

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
