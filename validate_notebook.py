"""CPU-only checks; does not execute installation, GPU or training cells."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
book = json.loads((ROOT / "AM_Query_Master.ipynb").read_text())
codes = [c for c in book["cells"] if c["cell_type"] == "code"]
assert len(codes) == 22, "Expected 22 numbered code cells"
for number, cell in enumerate(codes, 1):
    assert cell["metadata"]["am_query_cell"] == number
    assert not cell["outputs"] and cell["execution_count"] is None, "Clear notebook outputs before publishing"
    ast.parse("".join(cell["source"]))

definitions = ["".join(c["source"]) for c in codes
               if c["metadata"]["am_query_role"] == "definitions"]
generated = ast.parse("\n\n".join(definitions) + "\n\n" + book["metadata"]["am_query_cli_source"])
runner = ast.parse((ROOT / "am_query_runner.py").read_text())
if (runner.body and isinstance(runner.body[0], ast.Expr)
        and isinstance(runner.body[0].value, ast.Constant)
        and isinstance(runner.body[0].value.value, str)):
    runner.body.pop(0)
assert ast.dump(generated) == ast.dump(runner), "Notebook and runner implementations differ"
ast.parse((ROOT / "Cell_16_Storage_Fix.py").read_text())
print("PASS: 22 clean, parseable cells; definition/CLI AST matches runner; storage fix parses.")
