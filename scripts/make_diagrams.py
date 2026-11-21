"""Create assignment schema and method diagrams from the instructor's table specs."""

import argparse
import subprocess
from pathlib import Path

from preprocess import TABLES


def render(dot, path):
    source = path.with_suffix(".dot")
    source.write_text(dot)
    subprocess.run(["dot", "-Tpng", str(source), "-o", str(path)], check=True)
    subprocess.run(["dot", "-Tsvg", str(source), "-o", str(path.with_suffix(".svg"))], check=True)


def schema_dot():
    lines = [
        "digraph Schema {",
        '  graph [rankdir=TB, bgcolor="white", splines=true, nodesep=0.65, ranksep=0.9];',
        '  node [shape=box, style="rounded,filled", fillcolor="#e9f2fb", color="#3a6080", fontname="DejaVu Sans", fontsize=11];',
        '  edge [color="#53697d", fontname="DejaVu Sans", fontsize=8];',
    ]
    for name, spec in TABLES.items():
        fields = [f"PK: {spec['pk']}"]
        fields += [f"FK: {field}" for field in spec["fks"]]
        label = "\\n".join([name] + fields)
        lines.append(f'  {name} [label="{label}"];')
    for name, spec in TABLES.items():
        for field, target in spec["fks"].items():
            target_pk = TABLES[target]["pk"]
            lines.append(f'  {name} -> {target} [label="{field} → {target_pk}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


FRAMEWORK = """digraph Pipeline {
  graph [rankdir=LR, bgcolor="white", splines=ortho];
  node [shape=box, style="rounded,filled", fillcolor="#e9f2fb", color="#3a6080", fontname="DejaVu Sans"];
  edge [color="#53697d"];
  raw [label="GH Archive\\nhour files"];
  prep [label="Instructor preprocess.py\\nanonymization"];
  tables [label="Eight Parquet\\nrelational tables"];
  windows [label="UTC weekly cutoffs\\nvisible rows only"];
  features [label="User / Temporal /\\nRelational features"];
  models [label="Six LightGBM\\nbinary models"];
  tuning [label="Separate threshold week\\nearly stopping + F1 cutoffs"];
  eval [label="Later holdout week\\nteacher evaluate.py"];
  output [label="Provisional / final\\n4465-row CSV"];
  raw -> prep -> tables -> windows -> features -> models -> tuning -> eval -> output;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render(schema_dot(), args.out / "schema_graph.png")
    render(FRAMEWORK, args.out / "framework.png")
    print(args.out / "schema_graph.png")
    print(args.out / "framework.png")


if __name__ == "__main__":
    main()
