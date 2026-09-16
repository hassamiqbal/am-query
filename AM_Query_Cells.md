# AM-Query — one master GPU notebook

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


## Cell 01 — Confirm the GPU kernel

Expected: CUDA True and your L40S. This cell makes no installations or training changes.

```python
import sys
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
```


## Cell 02 — Install the project libraries

Use a project kernel rather than an institution-wide shared Python installation. These versions target the supplied Torch 2.5.1/CUDA 12.1 stack. The installer constrains Torch to its existing build. If it asks for a kernel restart, restart and continue at Cell 03. No paid API key is needed.

```python
import importlib.metadata as metadata
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
        constraint.write_text(f"torch=={torch_version}\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "--constraint",
                        str(constraint), *changes], check=True)
    if previously_loaded:
        raise RuntimeError("Packages installed. Restart this kernel, then continue from Cell 03.")
    print("Packages installed; your existing Torch/CUDA build was constrained and retained.")
else:
    print("Dependency versions already match. Continue.")
```


## Cell 03 — Configuration and local experiment records

Defines configuration, deadline checks, atomic records and a single-process lock. It does not start training.

```python
import argparse
import contextlib
import dataclasses
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

PROTOCOL_VERSION = "am-query-0.1.0"
DEPENDENCIES = {
    "transformers": "4.48.3", "peft": "0.14.0", "accelerate": "1.3.0",
    "bitsandbytes": "0.45.2", "huggingface-hub": "0.28.1",
    "safetensors": "0.5.2", "tokenizers": "0.21.0",
}

@dataclasses.dataclass
class Config:
    root: str = "am_query_runs/smoke"
    profile: str = "smoke"
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    requested_revision: str = "main"  # Resolved to one immutable commit on first use.
    seed: int = 20260916
    deadline: str = "2026-09-28T23:59:00+10:00"  # Melbourne; stop begins 5 min earlier.
    session_hours: float = 2.0  # Must be less than your allocation's wall time.
    checkpoint_reserve_seconds: int = 300
    seed_examples: int = 48
    dev_examples: int = 24
    test_examples: int = 144
    ood_examples: int = 48
    pool_examples: int = 24
    samples_per_task: int = 2
    warmup_steps: int = 8
    round_steps: int = 8
    max_rounds: int = 1
    patience: int = 3
    min_verified: int = 8
    max_length: int = 1536
    max_new_tokens: int = 256
    micro_batch: int = 1
    accumulation: int = 8
    inference_batch: int = 2
    rank: int = 16
    learning_rate: float = 5e-5
    checkpoint_every: int = 8
    min_gain: float = 0.02
    max_family_regression: float = 0.10
    bootstrap_replicates: int = 1000
    min_free_gb: float = 35.0

    @classmethod
    def for_profile(cls, profile="smoke", root=None, hours=None):
        if profile not in {"smoke", "pilot"}:
            raise ValueError("profile must be smoke or pilot")
        cfg = cls(profile=profile, root=root or f"am_query_runs/{profile}")
        if profile == "pilot":
            cfg.seed_examples = 2048
            cfg.dev_examples = 240
            cfg.test_examples = 360
            cfg.ood_examples = 120
            cfg.pool_examples = 512
            cfg.samples_per_task = 3
            cfg.warmup_steps = 256
            cfg.round_steps = 96
            cfg.max_rounds = 24
            cfg.micro_batch = 2
            cfg.accumulation = 8
            cfg.inference_batch = 4
            cfg.min_verified = 64
            cfg.checkpoint_every = 16
            cfg.session_hours = 10.0
        if hours is not None:
            cfg.session_hours = float(hours)
        return cfg

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def utc_now():
    return dt.datetime.now(dt.timezone.utc)

class BudgetExpired(RuntimeError):
    pass

class Budget:
    def __init__(self, cfg):
        end = dt.datetime.fromisoformat(cfg.deadline)
        if end.tzinfo is None:
            raise ValueError("deadline requires a timezone")
        if cfg.session_hours <= 0:
            raise ValueError("session_hours must be positive")
        self.stop_at = min(end.timestamp(), time.time() + cfg.session_hours * 3600)
        self.reserve = cfg.checkpoint_reserve_seconds
        self.requested_stop = False
        self.root = Path(cfg.root)

    def expired(self):
        return (self.requested_stop or (self.root / "STOP").exists()
                or time.time() >= self.stop_at - self.reserve)

    def check(self):
        if self.expired():
            raise BudgetExpired("Time limit, STOP file, or scheduler stop signal reached.")

@contextlib.contextmanager
def stop_signals(budget):
    previous = {}
    def request_stop(signum, frame):
        budget.requested_stop = True
    for name in ("SIGTERM", "SIGUSR1"):
        value = getattr(signal, name, None)
        if value is not None:
            try:
                previous[value] = signal.signal(value, request_stop)
            except ValueError:  # Only the main Python thread can install handlers.
                pass
    try:
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)

@contextlib.contextmanager
def run_lock(root):
    # Linux HPC/Jupyter: the OS releases the lock after a killed process.
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This experiment is already running in another process.")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def structural_config(cfg):
    values = dataclasses.asdict(cfg)
    # These may change between allocations without changing the experiment.
    for key in ("root", "session_hours", "checkpoint_reserve_seconds", "min_free_gb"):
        values.pop(key)
    return values
```


## Cell 04 — Select smoke test or pilot

The smoke profile uses 48 seed examples, 8 supervised update steps and one small self-training round. Pilot uses 2,048 seed examples, 256 supervised steps, up to 24 rounds and plateau stopping. It does not promise to use the whole period or improve every round. Once a run starts, use a new folder for changed hyperparameters.

```python
PROFILE = "smoke"  # First run: smoke. After it succeeds: pilot.
RUN_ROOT = f"am_query_runs/{PROFILE}"
SESSION_HOURS = 2.0 if PROFILE == "smoke" else 10.0
# SESSION_HOURS must fit inside your approved allocation, leaving time for startup.
CFG = Config.for_profile(PROFILE, root=RUN_ROOT, hours=SESSION_HOURS)

# Optional changes BEFORE a new experiment starts:
# CFG.micro_batch = 1  # Use this if pilot memory is insufficient.
# CFG.deadline = "2026-09-28T23:59:00+10:00"

print(json.dumps(dataclasses.asdict(CFG), indent=2))
print("Effective training batch:", CFG.micro_batch * CFG.accumulation)
```


## Cell 05 — Synthetic business schemas and independent task partitions

Creates fresh two-table business tasks. Training, development and final partitions use separate schema vocabularies. Two query compositions are reserved for the final test.

```python
FAMILIES = (
    "filtered_count", "total_value", "region_value", "category_mean",
    "top_entities", "no_activity", "monthly_value", "distinct_entities",
    "status_percentage", "count_range", "above_average", "latest_activity",
)
OOD_FAMILIES = ("running_total", "repeat_period")
STATES = ("paid", "pending", "cancelled", "refunded")
REGIONS = ("north", "south", "east", "west", "central")
CATEGORIES = ("hardware", "software", "service", "supplies")

def schema_for(split, variant):
    if split not in {"train", "dev", "test", "ood"}:
        raise ValueError("Unknown task partition")
    names = {
        "train": ("buyers", "purchases", "buyer_id", "region", "amount_cents", "status", "category", "ordered_on"),
        "dev": ("members", "payments", "member_key", "territory", "value_cents", "state", "segment", "booked_on"),
        "test": ("clients", "invoices", "client_key", "district", "total_cents", "stage", "product_class", "issued_on"),
        "ood": ("accounts", "transactions", "account_key", "zone", "net_cents", "payment_state", "service_class", "transacted_on"),
    }[split]
    person, event, key, region, amount, status, category, date = names
    # Schema identifiers differ across train/development/final partitions.
    return dict(person=f"{person}_{variant:02d}", event=f"{event}_{variant:02d}",
                key=key, region=region, amount=amount, status=status,
                category=category, date=date, event_id="record_id")

def schema_text(s):
    return (f"CREATE TABLE {s['person']} ({s['key']} INTEGER PRIMARY KEY, "
            f"{s['region']} TEXT NOT NULL);\n"
            f"CREATE TABLE {s['event']} ({s['event_id']} INTEGER PRIMARY KEY, "
            f"{s['key']} INTEGER NOT NULL REFERENCES {s['person']}({s['key']}), "
            f"{s['amount']} INTEGER NOT NULL, {s['status']} TEXT NOT NULL, "
            f"{s['category']} TEXT NOT NULL, {s['date']} TEXT NOT NULL);")

def make_task(split, index, seed=20260916):
    r = random.Random(f"{PROTOCOL_VERSION}:{seed}:{split}:{index}")
    families = OOD_FAMILIES if split == "ood" else FAMILIES
    family = families[index % len(families)]
    variant = (index // len(families)) % 12
    s = schema_for(split, variant)
    p, e, k, rg, a, st, cat, d, eid = (s[x] for x in (
        "person", "event", "key", "region", "amount", "status", "category", "date", "event_id"))
    status, region, category = r.choice(STATES), r.choice(REGIONS), r.choice(CATEGORIES)
    month = r.randint(1, 10)
    lo, mid, hi = (f"2025-{m:02d}-01" for m in (month, month + 1, month + 2))
    amount, limit, count = r.randrange(500, 20001, 100), r.randint(2, 8), r.randint(1, 5)
    join = f"FROM {p} p JOIN {e} e ON p.{k}=e.{k}"
    ordered = False
    if family == "filtered_count":
        q = f"Count records in {e} with {st} '{status}' and {a} at least {amount}."
        sql = f"SELECT COUNT(*) FROM {e} WHERE {st}='{status}' AND {a}>={amount};"
    elif family == "total_value":
        q = f"Return total {a} of '{status}' records from {lo} inclusive to {hi} exclusive. Return 0 if none."
        sql = f"SELECT COALESCE(SUM({a}),0) FROM {e} WHERE {st}='{status}' AND {d}>='{lo}' AND {d}<'{hi}';"
    elif family == "region_value":
        q = f"For each {rg}, return {rg} and sum of {a} for '{status}' records. Keep sums above {amount}. Sort by {rg}."
        sql = f"SELECT p.{rg}, SUM(e.{a}) {join} WHERE e.{st}='{status}' GROUP BY p.{rg} HAVING SUM(e.{a})>{amount} ORDER BY p.{rg};"
        ordered = True
    elif family == "category_mean":
        q = f"For each {cat}, return {cat} and mean {a} rounded to 2 decimals, using '{status}' records. Include groups with at least {count} records. Sort by {cat}."
        sql = f"SELECT {cat}, ROUND(AVG({a}),2) FROM {e} WHERE {st}='{status}' GROUP BY {cat} HAVING COUNT(*)>={count} ORDER BY {cat};"
        ordered = True
    elif family == "top_entities":
        q = f"Return the top {limit} {k} values and their sum of {a} for '{status}' records. Highest sum first; break ties by {k} ascending."
        sql = f"SELECT {k}, SUM({a}) AS total FROM {e} WHERE {st}='{status}' GROUP BY {k} ORDER BY total DESC, {k} ASC LIMIT {limit};"
        ordered = True
    elif family == "no_activity":
        q = f"Return {k} from {p} in {rg} '{region}' with no '{status}' records in {e}. Sort by {k}."
        sql = f"SELECT p.{k} FROM {p} p WHERE p.{rg}='{region}' AND NOT EXISTS (SELECT 1 FROM {e} e WHERE e.{k}=p.{k} AND e.{st}='{status}') ORDER BY p.{k};"
        ordered = True
    elif family == "monthly_value":
        q = f"Return year-month (YYYY-MM) and sum of {a} for '{status}' records in {e}, grouped by year-month of {d}. Sort earliest first."
        sql = f"SELECT strftime('%Y-%m',{d}) AS ym, SUM({a}) FROM {e} WHERE {st}='{status}' GROUP BY ym ORDER BY ym;"
        ordered = True
    elif family == "distinct_entities":
        q = f"Count distinct {k} with a '{status}' record in {cat} '{category}' and with {rg} '{region}'."
        sql = f"SELECT COUNT(DISTINCT e.{k}) {join} WHERE e.{st}='{status}' AND e.{cat}='{category}' AND p.{rg}='{region}';"
    elif family == "status_percentage":
        q = f"What percentage of all {e} records have {st} '{status}'? Return 0 if empty; otherwise round to 2 decimals."
        sql = f"SELECT COALESCE(ROUND(100.0*SUM(CASE WHEN {st}='{status}' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2),0) FROM {e};"
    elif family == "count_range":
        q = f"Return {k} and record count for '{status}' records from {lo} inclusive to {hi} exclusive. Keep counts at least {count}. Sort by {k}."
        sql = f"SELECT {k}, COUNT(*) FROM {e} WHERE {st}='{status}' AND {d}>='{lo}' AND {d}<'{hi}' GROUP BY {k} HAVING COUNT(*)>={count} ORDER BY {k};"
        ordered = True
    elif family == "above_average":
        q = f"Return {eid} and {a} for '{status}' records whose {a} exceeds the average {a} of all '{status}' records in {e}. Sort by {eid}."
        sql = f"SELECT {eid}, {a} FROM {e} WHERE {st}='{status}' AND {a}>(SELECT AVG({a}) FROM {e} WHERE {st}='{status}') ORDER BY {eid};"
        ordered = True
    elif family == "latest_activity":
        q = f"For each {k} with a '{status}' record, return {k}, {eid}, {a} of its latest such record by {d}. For equal dates take larger {eid}. Sort by {k}."
        sql = f"WITH ranked AS (SELECT {k}, {eid}, {a}, ROW_NUMBER() OVER (PARTITION BY {k} ORDER BY {d} DESC, {eid} DESC) AS rn FROM {e} WHERE {st}='{status}') SELECT {k}, {eid}, {a} FROM ranked WHERE rn=1 ORDER BY {k};"
        ordered = True
    elif family == "running_total":
        q = f"For '{status}' records in {e}, return {eid}, {k} and running sum of {a} per {k}, ordered within each account by {d} then {eid}. Sort output by {k}, {d}, {eid}."
        sql = f"SELECT {eid}, {k}, SUM({a}) OVER (PARTITION BY {k} ORDER BY {d}, {eid} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM {e} WHERE {st}='{status}' ORDER BY {k}, {d}, {eid};"
        ordered = True
    else:
        q = f"Return {k} from {p} with at least one '{status}' record in each interval [{lo}, {mid}) and [{mid}, {hi}). Dates are inclusive on the left only. Sort by {k}."
        sql = f"SELECT p.{k} FROM {p} p WHERE EXISTS (SELECT 1 FROM {e} e WHERE e.{k}=p.{k} AND e.{st}='{status}' AND e.{d}>='{lo}' AND e.{d}<'{mid}') AND EXISTS (SELECT 1 FROM {e} e WHERE e.{k}=p.{k} AND e.{st}='{status}' AND e.{d}>='{mid}' AND e.{d}<'{hi}') ORDER BY p.{k};"
        ordered = True
    introductions = {
        "train": ("Write a SQLite query. ", "Use the schema to answer: ", "Provide only SQL for this request: "),
        "dev": ("Produce a query for the analyst: ", "Translate this request into SQLite: "),
        "test": ("The reporting team asks: ", "Give the SQL needed for the following report: "),
        "ood": ("Create this business report in SQLite: ",),
    }
    task = dict(id=f"{split}-{index:08d}", split=split, index=index, family=family,
                schema=s, schema_variant=variant, question=r.choice(introductions[split])+q,
                gold=sql, ordered=ordered, database_seed=r.randrange(2**31),
                parameters=dict(status=status,region=region,category=category,amount=amount,
                                limit=limit,count=count,lo=lo,mid=mid,hi=hi))
    task["fingerprint"] = digest({"schema": s, "question": task["question"], "gold": sql,
                                  "database_seed":task["database_seed"]})
    return task

def make_tasks(split, n, seed, start=0):
    tasks, seen = [], set()
    for i in range(n*100):
        task = make_task(split, start+i, seed)
        key = digest({"schema":task["schema"],"gold":task["gold"]})
        if key not in seen:
            tasks.append(task)
            seen.add(key)
        if len(tasks) == n:
            return tasks
    raise RuntimeError("Requested more unique tasks than this generator can provide")

def build_database(task, variant=0):
    s = task["schema"]
    rng = random.Random(task["database_seed"] + 104729 * variant)
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_text(s))
    persons = [(i, REGIONS[(i+rng.randrange(5)) % 5]) for i in range(1, 49)]
    conn.executemany(f"INSERT INTO {s['person']} VALUES (?,?)", persons)
    events = []
    for i in range(1, 321):
        # Include zero amounts, ties, idle entities, all statuses and date boundaries.
        events.append((i, rng.randint(1, 42), rng.choice((0, 1000, rng.randint(1, 50000))),
                       STATES[(i+variant) % 4], rng.choice(CATEGORIES),
                       f"2025-{rng.randint(1,12):02d}-{rng.choice((1,2,15,28)):02d}"))
    # A final verifier variant is intentionally empty; check aggregate empty-set handling.
    if variant != 4:
        conn.executemany(f"INSERT INTO {s['event']} VALUES (?,?,?,?,?,?)", events)
    conn.commit()
    conn.enable_load_extension(False)
    conn.execute("PRAGMA query_only=ON")
    return conn
```


## Cell 06 — Read-only SQL verification with resource limits

Checks outputs across multiple database variants and validates gold queries against an independent Python reference. Generated SQL cannot write records, attach files or load extensions. This local verifier is not a production database security boundary.

```python
ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "coalesce", "ifnull", "nullif", "round",
    "strftime", "date", "julianday", "lower", "upper", "length", "abs",
    "row_number", "rank", "dense_rank", "lag", "lead", "total", "substr", "substring",
}

def extract_sql(text):
    text = text.strip()
    if text.startswith("```"):
        m = re.fullmatch(r"```(?:sql|sqlite)?\s*\n?(.*?)\n?```", text, re.I|re.S)
        if not m:
            raise ValueError("Return exactly one SQL code block or plain SQL.")
        text = m.group(1).strip()
    if len(text) > 12000 or not re.match(r"^(SELECT|WITH)\b", text, re.I):
        raise ValueError("Only a bounded SELECT/CTE query is accepted.")
    return text

def execute_readonly(conn, sql, schema, seconds=0.5, opcodes=200000, rows=1000):
    sql = extract_sql(sql)
    allowed_tables = {schema["person"], schema["event"]}
    def authorizer(action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in allowed_tables and database in {None, "main"}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and str(arg2 or arg1).lower() in ALLOWED_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    ticks, start = 0, time.monotonic()
    def progress():
        nonlocal ticks
        ticks += 1000
        return int(ticks > opcodes or time.monotonic()-start > seconds)
    conn.set_authorizer(authorizer)
    conn.set_progress_handler(progress, 1000)
    try:
        cur = conn.execute(sql)  # execute() rejects multiple statements.
        result = cur.fetchmany(rows+1)
        if len(result) > rows:
            raise ValueError("Query exceeds the row limit")
        return result
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)

def normalise_rows(rows, ordered):
    def value(v):
        if v is None:
            return ("null", "")
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                raise ValueError("Non-finite query result")
            return ("number", round(float(v), 6))
        return ("text", str(v))
    normalized = [tuple(value(v) for v in row) for row in rows]
    return normalized if ordered else sorted(normalized)

def verify_sql(task, candidate, variants=(0, 1, 2, 4)):
    try:
        candidate = extract_sql(candidate)
    except (ValueError, TypeError) as exc:
        return {"ok": False, "reason": str(exc)}
    for variant in variants:
        conn = build_database(task, variant)
        try:
            # Gold failures are generator defects, not candidate failures: fail loudly.
            expected = execute_readonly(conn, task["gold"], task["schema"])
            try:
                actual = execute_readonly(conn, candidate, task["schema"])
            except (sqlite3.Error, ValueError) as exc:
                return {"ok": False, "reason": f"execution: {type(exc).__name__}: {exc}"}
            expected_normal = normalise_rows(expected, task["ordered"])
            try:
                actual_normal = normalise_rows(actual, task["ordered"])
            except ValueError as exc:
                return {"ok":False,"reason":f"invalid result: {exc}"}
            if actual_normal != expected_normal:
                return {"ok": False, "reason": "result mismatch"}
        finally:
            conn.close()
    return {"ok": True, "reason": "matched all synthetic database variants"}

SYSTEM_PROMPT = (
    "You translate business questions into one read-only SQLite query. "
    "Return SQL only. Follow the provided schema exactly. Amounts are integer cents, "
    "dates use YYYY-MM-DD text. Do not write data, access files, or call tools."
)

def messages_for(task):
    return [{"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":schema_text(task["schema"])+"\n\n"+task["question"]}]

def data_self_check():
    for split in ("train", "dev", "test", "ood"):
        for i in range(24):
            task = make_task(split, i, seed=1729)  # Test fixtures differ from experiment data.
            assert verify_sql(task, task["gold"])["ok"], task["id"]
            for variant in (0,4):
                conn = build_database(task,variant)
                try:
                    actual = execute_readonly(conn,task["gold"],task["schema"])
                finally:
                    conn.close()
                expected = reference_answer(task,variant)
                assert normalise_rows(actual,task["ordered"]) == normalise_rows(expected,task["ordered"]), task["id"]
    task = make_task("train", 0)
    for sql in ("DELETE FROM buyers_00", "SELECT load_extension('x')",
                "SELECT 1; DROP TABLE buyers_00", "SELECT * FROM sqlite_master"):
        assert not verify_sql(task, sql)["ok"]
    assert not verify_sql(task, "SELECT 999999")["ok"]
    print("Data checks passed: 96 gold tasks, independent Python answers, database variants, read-only restrictions.")

def reference_answer(task,variant=0):
    """Independent Python oracle for checking the handwritten SQL gold templates."""
    conn = build_database(task,variant)
    try:
        people = dict(conn.execute(f"SELECT * FROM {task['schema']['person']}").fetchall())
        events = conn.execute(f"SELECT * FROM {task['schema']['event']}").fetchall()
    finally:
        conn.close()
    # Event tuple: record id, entity id, cents, status, category, date.
    p,f = task["parameters"],task["family"]
    matched = [e for e in events if e[3] == p["status"]]
    def rounded(num,den=1):
        return float((Decimal(num)/Decimal(den)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
    if f == "filtered_count":
        return [(sum(e[2]>=p["amount"] for e in matched),)]
    if f == "total_value":
        return [(sum(e[2] for e in matched if p["lo"]<=e[5]<p["hi"]),)]
    if f == "region_value":
        groups = defaultdict(int)
        for e in matched:
            groups[people[e[1]]] += e[2]
        return sorted((key,total) for key,total in groups.items() if total>p["amount"])
    if f == "category_mean":
        groups = defaultdict(list)
        for e in matched:
            groups[e[4]].append(e[2])
        return sorted((key,rounded(sum(v),len(v))) for key,v in groups.items() if len(v)>=p["count"])
    if f == "top_entities":
        groups = defaultdict(int)
        for e in matched:
            groups[e[1]] += e[2]
        return sorted(groups.items(),key=lambda x:(-x[1],x[0]))[:p["limit"]]
    if f == "no_activity":
        active = {e[1] for e in matched}
        return [(key,) for key,region in sorted(people.items()) if region == p["region"] and key not in active]
    if f == "monthly_value":
        groups = defaultdict(int)
        for e in matched:
            groups[e[5][:7]] += e[2]
        return sorted(groups.items())
    if f == "distinct_entities":
        return [(len({e[1] for e in matched if e[4]==p["category"] and people[e[1]]==p["region"]}),)]
    if f == "status_percentage":
        return [(rounded(100*len(matched),len(events)) if events else 0,)]
    if f == "count_range":
        groups = Counter(e[1] for e in matched if p["lo"]<=e[5]<p["hi"])
        return sorted((key,n) for key,n in groups.items() if n>=p["count"])
    if f == "above_average":
        total,n = sum(e[2] for e in matched),len(matched)
        return sorted((e[0],e[2]) for e in matched if n and e[2]*n>total)
    if f == "latest_activity":
        groups = defaultdict(list)
        for e in matched:
            groups[e[1]].append(e)
        result = []
        for key,entries in sorted(groups.items()):
            newest = max(entries,key=lambda x:(x[5],x[0]))
            result.append((key,newest[0],newest[2]))
        return result
    if f == "running_total":
        sums,result = defaultdict(int),[]
        for e in sorted(matched,key=lambda x:(x[1],x[5],x[0])):
            sums[e[1]] += e[2]
            result.append((e[0],e[1],sums[e[1]]))
        return result
    first = {e[1] for e in matched if p["lo"]<=e[5]<p["mid"]}
    second = {e[1] for e in matched if p["mid"]<=e[5]<p["hi"]}
    return [(key,) for key in sorted(first & second)]
```


## Cell 07 — GPU preflight and immutable model provenance

Checks packages, available VRAM and disk; records the exact model commit for reproducibility. Keep at least 35 GiB free; a longer experiment may need more for adapters and checkpoints.

```python
def storage_paths(cfg):
    root = Path(cfg.root).expanduser().resolve()
    return {"run": root, "model_cache": (root.parent / "model_cache").resolve()}

def existing_directory(path):
    path = Path(path).expanduser().resolve()
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise RuntimeError(f"Storage path is a file, not a directory: {path}")
    return path

def storage_report(cfg):
    report = {}
    for name, path in storage_paths(cfg).items():
        ancestor = existing_directory(path)
        report[name] = {"path": str(path), "checked_directory": str(ancestor),
                        "free_gib": shutil.disk_usage(ancestor).free / 2**30,
                        "writable": os.access(ancestor, os.W_OK | os.X_OK)}
    return report

def require_storage(cfg, create=False):
    report = storage_report(cfg)
    for name, entry in report.items():
        if not entry["writable"]:
            raise RuntimeError(f"{name} storage is not writable: {entry['path']}")
        if entry["free_gib"] < cfg.min_free_gb:
            raise RuntimeError(
                f"{name} storage has {entry['free_gib']:.2f} GiB free at {entry['path']}; "
                f"need at least {cfg.min_free_gb:.1f} GiB. Select assigned storage in "
                "Cell 16 (STORAGE_BASE), or set --root on the command line. "
                "Changing HF_HOME alone does not move this project's explicit model cache. "
                "Do not lower the threshold to bypass a full filesystem.")
    if create:
        import tempfile
        for entry in report.values():
            directory = Path(entry["path"])
            directory.mkdir(parents=True, exist_ok=True)
            # This detects write/permission failures, not remaining per-user quota.
            with tempfile.TemporaryFile(dir=directory) as probe:
                probe.write(b"AM-Query storage check\n")
                probe.flush()
                os.fsync(probe.fileno())
    return report

def configure_storage(cfg, storage_base=""):
    """Choose assigned storage for a NEW run; never move or delete existing data.

    An explicit base is a directory the operator is authorised to use. Automatic
    candidates are existing, user-owned directories from allocation variables.
    No temporary-directory variable or arbitrary shared mount is selected.
    """
    original = Path(cfg.root).expanduser().resolve()
    if not storage_base:
        try:
            require_storage(cfg)
            cfg.root = str(original)
            return cfg.root
        except (OSError, RuntimeError):
            pass
    candidates = []
    if storage_base:
        candidates.append(("STORAGE_BASE", Path(storage_base).expanduser().resolve()))
    else:
        for name in ("AM_QUERY_STORAGE", "SCRATCH", "WORK", "PROJECT"):
            value = os.environ.get(name)
            if value:
                path = Path(value).expanduser().resolve()
                try:
                    owned = hasattr(os, "getuid") and path.stat().st_uid == os.getuid()
                    if path.is_dir() and owned:
                        candidates.append((name, path))
                except OSError:
                    pass
    failures = []
    for name, base in candidates:
        try:
            if not base.is_dir():
                raise RuntimeError("Select an existing assigned directory")
            root = (base / "am_query_runs" / cfg.profile).resolve()
            candidate = dataclasses.replace(cfg, root=str(root))
            require_storage(candidate)
            if original != root and original.exists() and any(original.iterdir()):
                raise RuntimeError(
                    "Existing run files found. Preserve the original run: copy its complete "
                    "folder to assigned storage and explicitly set CFG.root to the copied "
                    "folder before continuing. Nothing has been moved or deleted.")
            cfg.root = str(root)
            print(f"Selected {name}: {base}")
            print("Confirm this directory's retention period and your individual storage quota.")
            return cfg.root
        except (OSError, RuntimeError) as exc:
            failures.append(f"{name}: {exc}")
    details = "\n".join(failures)
    raise RuntimeError(
        f"No suitable assigned storage found with {cfg.min_free_gb:.1f} GiB free. "
        "Run storage_diagnostics(CFG), then set STORAGE_BASE to your existing assigned "
        "project/scratch directory. If none has enough capacity, request more storage. "
        "No files were deleted and no download started.\n" + details)

def storage_diagnostics(cfg):
    import subprocess
    print(json.dumps(storage_report(cfg), indent=2))
    for name in ("AM_QUERY_STORAGE", "SCRATCH", "WORK", "PROJECT", "LOCAL_SCRATCH", "SLURM_TMPDIR"):
        value = os.environ.get(name)
        if value:
            print(f"{name}={value}")
    try:
        result = subprocess.run(["df", "-hP"], capture_output=True, text=True, timeout=10)
        print(result.stdout or result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Could not list mounted filesystems: {exc}")
    print("Filesystem free space is not a measurement of your personal/project quota.")
    print("Temporary allocation storage may disappear when the allocation ends.")

def preflight(cfg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select the allocated NVIDIA GPU Jupyter kernel first.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a GPU with BF16 support.")
    if sys.version_info < (3,10):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12.")
    mismatches = {}
    for package, expected in DEPENDENCIES.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches[package] = {"installed":actual, "expected":expected}
    if mismatches:
        raise RuntimeError("Run the notebook setup cell in a dedicated kernel, then restart: " + json.dumps(mismatches))
    root = Path(cfg.root).expanduser().resolve()
    cfg.root = str(root)
    storage = storage_report(cfg)
    disk_gb = storage["run"]["free_gib"]
    free, total = torch.cuda.mem_get_info(0)
    result = {"torch":torch.__version__, "cuda":torch.version.cuda,
              "gpu":torch.cuda.get_device_name(0), "gpu_free_gib":round(free/2**30,2),
              "gpu_total_gib":round(total/2**30,2), "disk_free_gib":round(disk_gb,2),
              "storage":storage, "python":sys.version.split()[0], "packages":DEPENDENCIES,
              "deadline":cfg.deadline, "session_hours":cfg.session_hours}
    print(json.dumps(result, indent=2))
    if free < 14*2**30:
        raise RuntimeError("Less than 14 GiB GPU memory is currently free; check the allocation.")
    require_storage(cfg, create=True)
    Budget(cfg).check()
    return result

def prepare_manifest(cfg, environment):
    path = Path(cfg.root)/"manifest.json"
    config = structural_config(cfg)
    if path.exists():
        existing = read_json(path)
        if existing["config"] != config or existing["protocol"] != PROTOCOL_VERSION:
            raise RuntimeError("Experiment settings changed. Use a NEW root; keep the old run for comparison.")
        return existing
    from huggingface_hub import HfApi
    revision = HfApi().model_info(cfg.model_id, revision=cfg.requested_revision, token=False).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Could not lock the model to an immutable commit")
    manifest = {"protocol":PROTOCOL_VERSION, "created_utc":utc_now().isoformat(),
                "config":config, "base_model":cfg.model_id, "base_revision":revision,
                "base_license":"Apache-2.0", "environment":environment,
                "data_source":"procedurally generated business records only",
                "final_test_policy":"single frozen comparison; no subsequent training"}
    atomic_json(path, manifest)
    return manifest
```


## Cell 08 — QLoRA engine: one allocated GPU, assistant-token loss only

Loads the 7B model in NF4 and trains LoRA weights in BF16. Prompts and padding are masked out of the loss. It uses only logical GPU 0 within your allocation.

```python
class Engine:
    def __init__(self, cfg, manifest):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        self.cfg = cfg
        self.torch = torch
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, revision=manifest["base_revision"], trust_remote_code=False,
            token=False,cache_dir=str(storage_paths(cfg)["model_cache"]))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=manifest["base_revision"], trust_remote_code=False,
            use_safetensors=True, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={"":0}, attn_implementation="sdpa", token=False,
            cache_dir=str(storage_paths(cfg)["model_cache"]))
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant":False})
        self.model = get_peft_model(base, LoraConfig(
            r=cfg.rank, lora_alpha=2*cfg.rank, lora_dropout=0.05,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
        self.model.print_trainable_parameters()
        print("Only the LoRA adapter is trainable. The 4-bit base weights remain fixed.")

    def restore_adapter(self, directory):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        weights = load_file(str(Path(directory)/"adapter_model.safetensors"), device="cpu")
        outcome = set_peft_model_state_dict(self.model, weights)
        if outcome.unexpected_keys:
            raise RuntimeError(f"Unexpected adapter keys: {outcome.unexpected_keys[:5]}")

    def save_adapter(self, directory):
        self.model.save_pretrained(str(directory), safe_serialization=True)

    def prompt(self, task):
        return self.tokenizer.apply_chat_template(
            messages_for(task), tokenize=False, add_generation_prompt=True)

    def generate(self, tasks, budget, samples=1, stochastic=False, seed=0):
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch, model, tok = self.torch, self.model, self.tokenizer
        budget.check()
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        tok.padding_side = "left"
        inputs = tok([self.prompt(t) for t in tasks], return_tensors="pt", padding=True,
                     add_special_tokens=False)
        if inputs.input_ids.shape[1] + self.cfg.max_new_tokens > self.cfg.max_length:
            raise RuntimeError("Prompt exceeds configured length; use a new experiment with a larger max_length.")
        inputs = {k:v.to("cuda:0") for k,v in inputs.items()}
        class StopForBudget(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return budget.expired()
        kwargs = dict(max_new_tokens=self.cfg.max_new_tokens,
                      do_sample=stochastic, num_return_sequences=samples,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                      use_cache=True, stopping_criteria=StoppingCriteriaList([StopForBudget()]))
        if stochastic:
            kwargs.update(temperature=0.7, top_p=0.9)
        elif samples != 1:
            raise ValueError("Greedy evaluation must generate exactly one answer")
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **kwargs)
        budget.check()  # A budget-truncated answer is never entered into an evaluation.
        text = tok.batch_decode(outputs[:,inputs["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)
        del outputs, inputs
        return [text[i*samples:(i+1)*samples] for i in range(len(tasks))]

    def encode_example(self, item):
        tok = self.tokenizer
        prompt_ids = tok(self.prompt(item["task"]), add_special_tokens=False)["input_ids"]
        answer_ids = tok(item["sql"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > self.cfg.max_length:
            raise RuntimeError("Training example is too long; it will not be silently truncated")
        return {"input_ids":prompt_ids+answer_ids,
                "labels":[-100]*len(prompt_ids)+answer_ids}

    def batch(self, items):
        torch = self.torch
        length = max(len(x["input_ids"]) for x in items)
        length = ((length+7)//8)*8
        ids, masks, labels = [], [], []
        for item in items:
            n = len(item["input_ids"])
            ids.append(item["input_ids"]+[self.tokenizer.pad_token_id]*(length-n))
            masks.append([1]*n+[0]*(length-n))
            labels.append(item["labels"]+[-100]*(length-n))
        return {k:torch.tensor(v, dtype=torch.long, device="cuda:0")
                for k,v in {"input_ids":ids,"attention_mask":masks,"labels":labels}.items()}
```


## Cell 09 — Durable training checkpoints and interruption recovery

Saves adapter, optimizer and RNG state. Recoverable interruptions resume from a complete optimizer checkpoint. SIGKILL/power loss can lose work since the latest checkpoint, not the stored champion.

```python
def adapter_digest(directory):
    path = Path(directory)/"adapter_model.safetensors"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def checkpoint_save(engine, optimizer, folder, step, trace):
    torch = engine.torch
    folder = Path(folder)
    target = folder/f"step_{step:06d}"
    if target.exists():
        # The existing checkpoint is already the last completed optimizer step.
        atomic_json(folder/"CURRENT.json", {"checkpoint":target.name,"step":step})
        return target
    temp = folder/f"step_{step:06d}.partial"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    engine.save_adapter(temp)
    torch.save({"step":step, "optimizer":optimizer.state_dict(),
                "cpu_rng":torch.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state(0)},
               temp/"optimizer.pt")
    atomic_json(temp/"trace.json", trace)
    os.replace(temp, target)
    atomic_json(folder/"CURRENT.json", {"checkpoint":target.name,"step":step})
    # Keep the latest two resumable checkpoints; parents and final adapters are separate.
    for old in sorted(folder.glob("step_[0-9]*"))[:-2]:
        if old.is_dir() and not old.name.endswith(".partial"):
            shutil.rmtree(old)
    return target

def train_stage(engine, examples, parent, folder, steps, budget, seed):
    cfg, torch = engine.cfg, engine.torch
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = [{"id":x["task"]["id"],"task_hash":x["task"]["fingerprint"],
               "sql":x["sql"],"source":x["source"]} for x in examples]
    stage_hash = digest({"parent":adapter_digest(parent), "examples":source, "steps":steps,
                         "config":structural_config(cfg), "seed":seed})
    manifest_path = folder/"training_manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["hash"] != stage_hash:
        raise RuntimeError("Refusing to resume a training stage with changed inputs")
    atomic_json(manifest_path, {"hash":stage_hash,"steps":steps,"examples":source})
    final = folder/"final"
    if (final/"COMPLETE.json").exists():
        if read_json(final/"COMPLETE.json")["hash"] != stage_hash:
            raise RuntimeError("Completed stage hash mismatch")
        engine.restore_adapter(final)
        return final
    budget.check()
    encoded = [engine.encode_example(x) for x in examples]
    if not encoded:
        raise RuntimeError("No verified training examples")
    engine.restore_adapter(parent)
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad],
                                 lr=cfg.learning_rate, weight_decay=0.01)
    completed, trace = 0, []
    current = folder/"CURRENT.json"
    if current.exists():
        ckpt = folder/read_json(current)["checkpoint"]
        engine.restore_adapter(ckpt)
        # Only load checkpoints generated locally by this experiment, never untrusted files.
        state = torch.load(ckpt/"optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device=0)
        completed = int(state["step"])
        trace = read_json(ckpt/"trace.json")
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        checkpoint_save(engine, optimizer, folder, 0, [])
    engine.model.train()
    engine.model.config.use_cache = False
    engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    started = time.monotonic()
    step_cpu_rng, step_cuda_rng = None, None
    try:
        for step in range(completed, steps):
            budget.check()
            step_cpu_rng = torch.get_rng_state()
            step_cuda_rng = torch.cuda.get_rng_state(0)
            warm = max(1, min(10, steps//10))
            factor = min(1.0, (step+1)/warm) * max(0.1, (steps-step)/max(1,steps-warm))
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate*factor
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(cfg.accumulation):
                budget.check()
                rng = random.Random(seed + step*cfg.accumulation + micro)
                selected = [encoded[rng.randrange(len(encoded))] for _ in range(cfg.micro_batch)]
                batch = engine.batch(selected)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = engine.model(**batch).loss
                if not torch.isfinite(raw_loss).item():
                    raise RuntimeError("Non-finite training loss; latest complete checkpoint remains available")
                step_loss += float(raw_loss.detach().cpu())/cfg.accumulation
                (raw_loss/cfg.accumulation).backward()
                del raw_loss, batch
            grad = torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0,
                                                 error_if_nonfinite=True)
            optimizer.step()
            completed = step+1
            step_cpu_rng, step_cuda_rng = None, None
            trace.append({"step":completed,"loss":step_loss,"grad_norm":float(grad),
                          "gpu_allocated_gib":torch.cuda.memory_allocated()/2**30})
            if completed % 4 == 0 or completed == steps:
                print(f"{folder.name}: step {completed}/{steps} loss={step_loss:.4f}", flush=True)
            if completed % cfg.checkpoint_every == 0:
                checkpoint_save(engine, optimizer, folder, completed, trace)
        checkpoint_save(engine, optimizer, folder, completed, trace)
        temp = folder/"final.partial"
        if temp.exists():
            shutil.rmtree(temp)
        engine.save_adapter(temp)
        atomic_json(temp/"COMPLETE.json", {"hash":stage_hash,"steps":completed,
                    "seconds_this_session":time.monotonic()-started})
        if final.exists():
            shutil.rmtree(final)
        os.replace(temp, final)
        atomic_json(folder/"training_trace.json",trace)
        # A completed stage resumes from final, so its optimizer snapshots can be
        # reclaimed. Keep final adapters and traces for ablation/reproducibility.
        for completed_checkpoint in folder.glob("step_[0-9]*"):
            if completed_checkpoint.is_dir():
                shutil.rmtree(completed_checkpoint)
        (folder/"CURRENT.json").unlink(missing_ok=True)
        return final
    except BudgetExpired:
        # Discard partial accumulated gradients; resume from a complete optimizer step.
        optimizer.zero_grad(set_to_none=True)
        if step_cpu_rng is not None:
            torch.set_rng_state(step_cpu_rng)
            torch.cuda.set_rng_state(step_cuda_rng, device=0)
        checkpoint_save(engine, optimizer, folder, completed, trace)
        raise BudgetExpired("Training paused at a recoverable optimizer checkpoint")
    except KeyboardInterrupt:
        # An interrupt could arrive during optimizer.step(): do not serialize partial weights.
        raise BudgetExpired("Interrupted. Resume from the latest durable checkpoint.")
    finally:
        del optimizer, encoded
        gc.collect()
        torch.cuda.empty_cache()
```


## Cell 10 — Evaluation caches, paired selection gates, and honest uncertainty

Greedy pass@1 is the selection metric. Promotion needs at least 2 percentage points improvement, a positive paired bootstrap lower bound and no large task-family regression. This gate is not a formal claim of generalization.

```python
def score_records(records):
    n = len(records)
    if not n:
        return {"n":0,"accuracy":0.0,"families":{},"wilson95":[0.0,1.0]}
    successes = sum(bool(x["ok"]) for x in records)
    p, z = successes/n, 1.959963984540054
    centre = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    groups = defaultdict(list)
    for x in records:
        groups[x["family"]].append(int(x["ok"]))
    return {"n":n,"accuracy":p,"successes":successes,"wilson95":[centre-radius,centre+radius],
            "families":{k:{"n":len(v),"accuracy":statistics.mean(v)} for k,v in sorted(groups.items())}}

def evaluate(engine, tasks, adapter, folder, budget):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest({"model":adapter_digest(adapter),
                       "tasks":[t["fingerprint"] for t in tasks],
                       "decoding":"greedy-pass-at-1", "protocol":PROTOCOL_VERSION})
    identity_path = folder/"identity.json"
    if identity_path.exists() and read_json(identity_path)["hash"] != identity:
        raise RuntimeError("Evaluation cache is for a different model or dataset")
    atomic_json(identity_path, {"hash":identity})
    records_path = folder/"records.json"
    records = read_json(records_path) if records_path.exists() else []
    by_id = {x["id"]:x for x in records}
    missing = [t for t in tasks if t["id"] not in by_id]
    if missing:
        engine.restore_adapter(adapter)
    for offset in range(0,len(missing),engine.cfg.inference_batch):
        budget.check()
        batch = missing[offset:offset+engine.cfg.inference_batch]
        outputs = engine.generate(batch,budget,seed=engine.cfg.seed)
        for task, variants in zip(batch, outputs):
            check = verify_sql(task,variants[0],variants=(0,1,2,3,4))
            records.append({"id":task["id"],"family":task["family"],
                            "schema_variant":task["schema_variant"],
                            "prediction":variants[0],**check})
        atomic_json(records_path,records)
        if (offset//engine.cfg.inference_batch) % 10 == 0:
            print(f"{folder.name}: evaluated {len(records)}/{len(tasks)}",flush=True)
    if len(records) != len(tasks):
        raise RuntimeError("Partial evaluation cannot be scored or promoted")
    records = sorted(records,key=lambda x:x["id"])
    result = {"identity":identity,"metrics":score_records(records),"records":records,
              "scope":"synthetic generator only; not a customer benchmark"}
    atomic_json(folder/"metrics.json",result["metrics"])
    return result

def promotion_decision(incumbent,candidate,cfg):
    a,b = incumbent["records"],candidate["records"]
    if [x["id"] for x in a] != [x["id"] for x in b]:
        raise ValueError("Promotion requires identical evaluation items")
    delta = candidate["metrics"]["accuracy"]-incumbent["metrics"]["accuracy"]
    # Cluster the paired differences by schema variant, rather than treating template
    # instances as independent. This is a selection gate, not a publication p-value.
    groups = defaultdict(list)
    for old,new in zip(a,b):
        groups[old["schema_variant"]].append(int(new["ok"])-int(old["ok"]))
    clusters = list(groups.values())
    rng = random.Random(cfg.seed)
    deltas = []
    for _ in range(cfg.bootstrap_replicates):
        resample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat = [x for group in resample for x in group]
        deltas.append(statistics.mean(flat))
    lower = sorted(deltas)[int(0.025*len(deltas))]
    regressions = []
    for name,old in incumbent["metrics"]["families"].items():
        new = candidate["metrics"]["families"][name]
        if old["accuracy"]-new["accuracy"] > cfg.max_family_regression+1e-12:
            regressions.append(name)
    promote = delta >= cfg.min_gain-1e-12 and lower > 0 and not regressions
    return {"promote":promote,"absolute_accuracy_gain":delta,
            "paired_cluster_bootstrap_lower95":lower,"regressed_families":regressions,
            "interpretation":"development selection only; repeated use can overfit"}
```


## Cell 11 — Model-generated examples: retain only execution-verified answers

Samples the current model, executes each answer only against generated databases, and retains verified candidates. The gold reference is used for scoring; it is not inserted as a self-generated training answer.

```python
def collect_verified(engine,tasks,parent,folder,budget,seed):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    identity = digest({"parent":adapter_digest(parent),"task_hashes":[t["fingerprint"] for t in tasks],
                       "samples":engine.cfg.samples_per_task,"seed":seed})
    identity_path = folder/"identity.json"
    if identity_path.exists() and read_json(identity_path)["hash"] != identity:
        raise RuntimeError("Rollout cache mismatch")
    atomic_json(identity_path,{"hash":identity})
    cache = folder/"attempts.json"
    attempts = read_json(cache) if cache.exists() else []
    done = {x["id"] for x in attempts}
    remaining = [t for t in tasks if t["id"] not in done]
    if remaining:
        engine.restore_adapter(parent)
    # One task at a time keeps sampled RNG deterministic after a checkpoint resume.
    for task in remaining:
        budget.check()
        candidates = engine.generate([task],budget,samples=engine.cfg.samples_per_task,
                                     stochastic=True,seed=seed+task["index"])[0]
        verdicts = []
        verified = []
        for sql in candidates:
            check = verify_sql(task,sql)
            verdicts.append({"sql":sql,**check})
            if check["ok"]:
                verified.append(extract_sql(sql))
        attempts.append({"id":task["id"],"candidates":verdicts,
                         "selected":min(verified,key=len) if verified else None})
        atomic_json(cache,attempts)
        if len(attempts)%16 == 0:
            print(f"Verified examples: {sum(x['selected'] is not None for x in attempts)}/{len(attempts)} tasks",flush=True)
    by_id = {t["id"]:t for t in tasks}
    accepted = [{"task":by_id[x["id"]],"sql":x["selected"],"source":"verified_model_generation"}
                for x in attempts if x["selected"] is not None]
    atomic_json(folder/"summary.json",{"tasks":len(tasks),"verified_examples":len(accepted),
                                      "self_generated":True,"gold_used_as_targets":False})
    return accepted
```


## Cell 12 — Experiment controller: baseline, supervised seed, then verified self-training

Runs the complete experiment, records the seed-data origin, replays some seed examples, retains the best model, stops on a plateau and exports its adapter.

```python
def load_or_make_seed(cfg,budget):
    path = Path(cfg.root)/"data"/"seed_examples.json"
    if path.exists():
        return read_json(path)
    tasks = make_tasks("train",cfg.seed_examples,cfg.seed)
    examples = []
    for i,task in enumerate(tasks):
        if i%16 == 0:
            budget.check()
        if not verify_sql(task,task["gold"])["ok"]:
            raise RuntimeError("Seed oracle failure")
        examples.append({"task":task,"sql":task["gold"],"source":"procedural_gold_seed"})
    atomic_json(path,examples)
    return examples

def save_state(root,state):
    state["updated_utc"] = utc_now().isoformat()
    atomic_json(Path(root)/"state.json",state)

def run_experiment(cfg):
    root = Path(cfg.root).expanduser().resolve()
    cfg.root = str(root)
    if (root/"final_freeze.json").exists():
        raise RuntimeError("This run is frozen for final testing. Start a new experiment for further training.")
    engine = None
    with run_lock(root):
        if (root/"final_freeze.json").exists():
            raise RuntimeError("This run is frozen for final testing")
        budget = Budget(cfg)
        with stop_signals(budget):
            try:
                budget.check()
                environment = preflight(cfg)
                manifest = prepare_manifest(cfg,environment)
                data_self_check()
                state_path = root/"state.json"
                state = read_json(state_path) if state_path.exists() else {
                    "phase":"baseline","round":0,"champion":"baseline", "champion_eval":"eval/baseline_dev",
                    "bad_rounds":0,"history":[],"status":"running"}
                if state["status"] in {"completed","plateau","insufficient_verified_examples"}:
                    print("Experiment already stopped:",state["status"])
                    return state
                engine = Engine(cfg,manifest)
                budget.check()
                (root/"tokenizer").mkdir(exist_ok=True)
                engine.tokenizer.save_pretrained(root/"tokenizer")
                baseline = root/"baseline"
                if not (baseline/"adapter_model.safetensors").exists():
                    # Fresh LoRA B matrices are zero: this is the quantized base-model baseline.
                    engine.save_adapter(baseline)
                dev = make_tasks("dev",cfg.dev_examples,cfg.seed)
                atomic_json(root/"data"/"development.json",dev)
                seed_examples = load_or_make_seed(cfg,budget)
                seed_keys = {digest({"schema":x["task"]["schema"],"gold":x["task"]["gold"]}) for x in seed_examples}
                if state["phase"] == "baseline":
                    base_eval = evaluate(engine,dev,baseline,root/"eval/baseline_dev",budget)
                    print("Frozen base development accuracy:",base_eval["metrics"]["accuracy"])
                    state["phase"] = "supervised_seed"
                    save_state(root,state)
                if state["phase"] == "supervised_seed":
                    trained = train_stage(engine,seed_examples,baseline,root/"supervised_seed",
                                          cfg.warmup_steps,budget,cfg.seed)
                    old = evaluate(engine,dev,baseline,root/"eval/baseline_dev",budget)
                    new = evaluate(engine,dev,trained,root/"eval/supervised_dev",budget)
                    decision = promotion_decision(old,new,cfg)
                    state["history"].append({"stage":"supervised_seed",**decision,
                        "candidate_accuracy":new["metrics"]["accuracy"]})
                    if decision["promote"]:
                        state["champion"] = "supervised_seed/final"
                        state["champion_eval"] = "eval/supervised_dev"
                    state["phase"] = "self_training"
                    save_state(root,state)
                    print("Supervised seed decision:",decision)
                while state["round"] < cfg.max_rounds:
                    budget.check()
                    round_no = state["round"] + 1
                    folder = root/f"rounds/round_{round_no:03d}"
                    task_path = folder/"tasks.json"
                    if task_path.exists():
                        tasks = read_json(task_path)
                    else:
                        tasks = make_tasks("train",cfg.pool_examples*2,cfg.seed,
                                           start=1000000*round_no)
                        tasks = [t for t in tasks if digest({"schema":t["schema"],"gold":t["gold"]}) not in seed_keys][:cfg.pool_examples]
                        if not tasks:
                            state["status"] = "insufficient_verified_examples"
                            save_state(root,state)
                            break
                        atomic_json(task_path,tasks)
                    parent = root/state["champion"]
                    accepted = collect_verified(engine,tasks,parent,folder/"rollouts",budget,
                                                seed=cfg.seed+round_no)
                    if len(accepted) < cfg.min_verified:
                        state["status"] = "insufficient_verified_examples"
                        state["history"].append({"round":round_no,"verified":len(accepted),
                                                 "reason":"Do not train on unverified self-generated answers"})
                        save_state(root,state)
                        break
                    rng = random.Random(cfg.seed+round_no)
                    # Replay gold seed data to reduce forgetting; label its source explicitly.
                    replay = rng.sample(seed_examples,min(len(seed_examples),max(1,len(accepted)//2)))
                    examples = accepted+replay
                    rng.shuffle(examples)
                    trained = train_stage(engine,examples,parent,folder/"training",
                                          cfg.round_steps,budget,cfg.seed+round_no)
                    incumbent = evaluate(engine,dev,parent,root/state["champion_eval"],budget)
                    candidate_eval = folder/"development"
                    candidate = evaluate(engine,dev,trained,candidate_eval,budget)
                    decision = promotion_decision(incumbent,candidate,cfg)
                    state["history"].append({"round":round_no,"verified":len(accepted),
                                             "candidate_accuracy":candidate["metrics"]["accuracy"],**decision})
                    if decision["promote"]:
                        state["champion"] = str(trained.relative_to(root))
                        state["champion_eval"] = str(candidate_eval.relative_to(root))
                        state["bad_rounds"] = 0
                    else:
                        state["bad_rounds"] += 1
                    state["round"] = round_no
                    save_state(root,state)
                    print(f"Round {round_no} decision:",decision,flush=True)
                    if state["bad_rounds"] >= cfg.patience:
                        state["status"] = "plateau"
                        save_state(root,state)
                        break
                if state["round"] >= cfg.max_rounds:
                    state["status"] = "completed"
                    save_state(root,state)
                export_bundle(cfg)
                print("Status:",state["status"],"| Best adapter:",state["champion"])
                return state
            except (BudgetExpired,KeyboardInterrupt) as exc:
                atomic_json(root/"last_pause.json",{"utc":utc_now().isoformat(),"reason":str(exc)})
                print("Paused. Rerun the SAME config/root to resume. Reason:",exc)
                return read_json(root/"state.json") if (root/"state.json").exists() else {"status":"paused_before_initialization"}
            finally:
                if engine is not None:
                    del engine
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
```


## Cell 13 — One frozen final test: baseline, supervised-only, and selected champion

Defines final evaluation without running it. Calling it later freezes this run before viewing test results. It compares the base, supervised-only and chosen model. No further training may use the same frozen run.

```python
def final_evaluation(cfg):
    root = Path(cfg.root).expanduser().resolve()
    cfg.root = str(root)
    engine = None
    with run_lock(root):
        budget = Budget(cfg)
        budget.check()
        env = preflight(cfg)
        manifest = prepare_manifest(cfg,env)
        state = read_json(root/"state.json")
        if state["phase"] != "self_training":
            raise RuntimeError("Complete supervised training and selection before final evaluation")
        if not (root/"supervised_seed/final/COMPLETE.json").exists():
            raise RuntimeError("Missing supervised baseline")
        models = {"base":root/"baseline", "supervised_only":root/"supervised_seed/final",
                  "selected_champion":root/state["champion"]}
        frozen = {"config":structural_config(cfg),
                  "adapters":{name:adapter_digest(path) for name,path in models.items()}}
        freeze = root/"final_freeze.json"
        if freeze.exists():
            if read_json(freeze)["identity"] != digest(frozen):
                raise RuntimeError("Final-test models/settings changed. Do not reuse this holdout for selection")
        else:
            atomic_json(freeze,{"identity":digest(frozen),"frozen":frozen,"utc":utc_now().isoformat()})
        with stop_signals(budget):
            try:
                engine = Engine(cfg,manifest)
                # Neither partition is generated by training or by model selection.
                final_sets = {"heldout_schema":make_tasks("test",cfg.test_examples,cfg.seed),
                              "heldout_composition":make_tasks("ood",cfg.ood_examples,cfg.seed)}
                report_path = root/"final_report.json"
                report = {"frozen_identity":digest(frozen),"profile":cfg.profile,"results":{},
                          "scope":"synthetic held-out schemas and query compositions; no real-customer accuracy claim",
                          "interval_note":"Wilson intervals are descriptive; generated examples are correlated",
                          "commercial_readiness":"not established by this experiment"}
                for name,adapter in models.items():
                    report["results"][name] = {}
                    for split,tasks in final_sets.items():
                        result = evaluate(engine,tasks,adapter,root/f"final_eval/{name}/{split}",budget)
                        report["results"][name][split] = result["metrics"]
                for split in final_sets:
                    champion = report["results"]["selected_champion"][split]["accuracy"]
                    supervised = report["results"]["supervised_only"][split]["accuracy"]
                    report.setdefault("champion_minus_supervised",{})[split] = champion-supervised
                atomic_json(report_path,report)
                export_bundle(cfg)
                print(json.dumps(report,indent=2))
                return report
            except (BudgetExpired,KeyboardInterrupt) as exc:
                print("Final test paused; rerun it with the identical frozen models:",exc)
                return {"status":"paused_final_evaluation"}
            finally:
                if engine is not None:
                    del engine
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
```


## Cell 14 — Export and a synthetic demonstration; nothing is automatically published

Creates an adapter/model-card bundle and an optional generated-data demo. It does not merge or redistribute the pretrained base model, create a public repository or deploy an endpoint.

```python
def export_bundle(cfg):
    root = Path(cfg.root).expanduser().resolve()
    state = read_json(root/"state.json")
    manifest = read_json(root/"manifest.json")
    source = root/state["champion"]
    export = root/"export"
    export.mkdir(parents=True,exist_ok=True)
    for name in ("adapter_model.safetensors","adapter_config.json"):
        shutil.copy2(source/name,export/name)
    tokenizer_dir = root/"tokenizer"
    if tokenizer_dir.exists():
        for file in tokenizer_dir.iterdir():
            if file.is_file():
                shutil.copy2(file,export/file.name)
    for name in ("manifest.json","state.json","final_report.json"):
        if (root/name).exists():
            shutil.copy2(root/name,export/name)
    final_text = "Not run. Development results must not be advertised as test performance."
    if (root/"final_report.json").exists():
        final_text = json.dumps(read_json(root/"final_report.json")["results"],indent=2)
    card = f"""# AM-Query experimental adapter

Owner: Hassam Iqbal. Status: research experiment, not production validated.
Base: {manifest['base_model']}
Immutable base revision: {manifest['base_revision']}
Base licence: Apache-2.0. Preserve applicable attribution/licence/NOTICE terms.
This is a LoRA derivative. It is not a new foundation model trained from scratch.
Selected checkpoint: {state['champion']}

## Training and intended use
Procedurally generated business questions and SQLite queries. A gold seed stage is
followed by rejection-sampled self-training with execution verification and seed
replay. The external verifier and held-out data supply the grounding. No clinical,
patient, customer, web-scraped or paid model API data is included.

## Limitations
Only English, SQLite and a small family of synthetic two-table business schemas
have been exercised. No claim of general SQL accuracy, superintelligence, novelty,
publication acceptance, monetary value or business readiness follows. Read-only
SQLite authorisation is a local prototype control, not a production sandbox.
The model may make incorrect, expensive or inappropriate queries.
Do not connect it to production databases without separate security and accuracy work.

## Evaluation
Development selection repeatedly uses one set and can overfit it. Final test
partitions are frozen before their first use. Compare baseline, supervised-only
and selected champion. A champion that does not beat supervised-only is not evidence
that self-training adds value. Generated-task intervals are descriptive only.

{final_text}

## Commercial release
Check institution GPU-use/IP conditions, base-model redistribution terms and any
future customer-data rights before release. This export publishes nothing.
Complete customer-specific evaluation, access control and operational support first.
"""
    (export/"README.md").write_text(card,encoding="utf-8")
    atomic_json(export/"integrity.json",{"adapter_sha256":adapter_digest(export),
                "base_revision":manifest["base_revision"],"protocol":PROTOCOL_VERSION})
    print("Export ready:",export)
    return export

def demo(cfg,index=108):
    root = Path(cfg.root).expanduser().resolve()
    cfg.root = str(root)
    with run_lock(root):
        preflight(cfg)
        engine = Engine(cfg,read_json(root/"manifest.json"))
        try:
            state = read_json(root/"state.json")
            engine.restore_adapter(root/state["champion"])
            task = make_task("train",index,cfg.seed)
            answer = engine.generate([task],Budget(cfg))[0][0]
            result = {"question":task["question"],"schema":schema_text(task["schema"]),
                      "sql":answer,"verification":verify_sql(task,answer)}
            if result["verification"]["ok"]:
                conn = build_database(task)
                try:
                    result["synthetic_result"] = execute_readonly(conn,answer,task["schema"])[:20]
                finally:
                    conn.close()
            print(json.dumps(result,indent=2))
            return result
        finally:
            del engine
            gc.collect()
            import torch
            torch.cuda.empty_cache()
```


## Cell 15 — Check the data and verifier

Run this before downloading model weights. It uses the CPU only.

```python
data_self_check()
```


## Cell 16 — Check the allocated GPU and disk

If your home folder has too little space, enter your assigned project/scratch directory in STORAGE_BASE. This puts run outputs and the explicit model cache on that storage. Automatic selection considers only existing user-owned directories in AM_QUERY_STORAGE, SCRATCH, WORK or PROJECT. It never selects SLURM_TMPDIR automatically. Check your storage quota and retention policy. A filesystem can report free space beyond your individual quota. Existing run files are never silently abandoned. If no suitable path exists, the cell prints storage diagnostics and stops before downloading. Do not reduce the 35 GiB check to bypass the problem.

```python
# Optional: paste an EXISTING storage directory assigned to your account here.
# Leave blank to keep a suitable current folder or inspect allocation variables.
STORAGE_BASE = ""

try:
    configure_storage(CFG, storage_base=STORAGE_BASE)
except RuntimeError:
    storage_diagnostics(CFG)
    raise
GPU_REPORT = preflight(CFG)
```


## Cell 17 — Run or resume the experiment

This is the training cell. It downloads public base-model files on first use, runs a baseline, fine-tunes, verifies sampled answers and evaluates candidates. There is no claimed runtime: the smoke test measures whether your actual environment works. On a new session, run the definition cells and this cell with the SAME config/root. A time-limit pause is expected and resumable.

```python
RUN_STATE = run_experiment(CFG)
```


## Cell 18 — Inspect progress and decisions

A rejected adapter is not a failure of the pipeline. Check whether improvements persist beyond supervised fine-tuning. Loss decreasing alone is not enough.

```python
root = Path(CFG.root).expanduser().resolve()
if (root/"state.json").exists():
    state = read_json(root/"state.json")
    print("Status:", state["status"])
    print("Best checkpoint:", state["champion"])
    print(json.dumps(state["history"], indent=2))
else:
    print("Initialization has not yet completed. Inspect the preceding cell output.")
```


## Cell 19 — Freeze and run the final holdout

This is a deliberate research freeze, not a publishing action. Afterwards the run refuses further training. Finish it before your GPU access ends. An interrupted final test resumes on the exact same models.

```python
RUN_FINAL_TEST = False  # Change to True only when this experiment is finished.
if RUN_FINAL_TEST:
    FINAL_REPORT = final_evaluation(CFG)
else:
    print("Final test remains unopened. Continue model selection first.")
```


## Cell 20 — Export the selected adapter

The bundle contains the selected LoRA adapter, tokenizer, model card, provenance and available metrics. If the base model remains best, the export records that honestly.

```python
root = Path(CFG.root).expanduser().resolve()
if (root/"state.json").exists():
    export_path = export_bundle(CFG)
    print("Keep the complete run folder and export folder on persistent approved storage.")
else:
    print("Complete initialization before exporting.")
```


## Cell 21 — Try one generated business question

This optional cell reloads the model and consumes allocated GPU time. It uses a generated database only.

```python
RUN_DEMO = False
if RUN_DEMO:
    DEMO_RESULT = demo(CFG, index=108)
else:
    print("Set RUN_DEMO=True to reload the selected model and try a synthetic question.")
```


## Cell 22 — Prepare a terminal/batch runner

For long sessions, use the script within your institution's scheduler or supported persistent GPU service. This cell prints the exact command for your kernel and paths. It does not submit a job or reserve GPU time. Keep the output path on persistent storage, not an allocation's temporary filesystem.

```python
# The companion am_query_runner.py contains the same definition cells.
# If you only downloaded the notebook, generate that script from it here.
import shlex

NOTEBOOK_PATH = Path("AM_Query_Master.ipynb")
if NOTEBOOK_PATH.exists():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    definitions = ["".join(c["source"]) for c in notebook["cells"]
                   if c["cell_type"] == "code" and c.get("metadata", {}).get("am_query_role") == "definitions"]
    cli = notebook["metadata"]["am_query_cli_source"]
    Path("am_query_runner.py").write_text("\n\n".join(definitions)+"\n\n"+cli, encoding="utf-8")
    print("Wrote am_query_runner.py from this notebook's definitions.")
elif not Path("am_query_runner.py").exists():
    raise FileNotFoundError("Place this notebook here under its original filename, or download am_query_runner.py.")

cmd = [sys.executable, str(Path("am_query_runner.py").resolve()),
       "--profile", CFG.profile, "--root", str(Path(CFG.root).resolve()),
       "--hours", str(CFG.session_hours)]
print("Run this command INSIDE your allocated GPU job/session:")
print(shlex.join(cmd))
print("Only one process may use a run folder. Stop notebook training before launching the script.")
```


## Running until 28 September

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
