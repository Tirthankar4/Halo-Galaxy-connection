"""
Build a Colab/Kaggle-ready bundle for running NF_train.py.

Usage:
    python CAMELS_codes/build_nf_train_cloud_bundle.py
    python CAMELS_codes/build_nf_train_cloud_bundle.py --entry-script CAMELS_codes/NF_train.py
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


THIRD_PARTY_PACKAGE_MAP = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "pyro": "pyro-ppl",
}


def parse_imports(py_file: Path) -> List[tuple[str, int]]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    imports: List[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imports.append((module_name, node.level))
    return imports


def resolve_local_module(
    importer: Path,
    module_name: str,
    level: int,
    search_roots: Iterable[Path],
) -> Optional[Path]:
    search_roots = [root.resolve() for root in search_roots]

    if level > 0:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        module_parts = [part for part in module_name.split(".") if part]
        candidate = base.joinpath(*module_parts)
        py_candidate = candidate.with_suffix(".py")
        if py_candidate.exists():
            return py_candidate.resolve()
        init_candidate = candidate / "__init__.py"
        if init_candidate.exists():
            return init_candidate.resolve()
        return None

    if not module_name:
        return None

    module_parts = module_name.split(".")
    for root in search_roots:
        candidate = root.joinpath(*module_parts)
        py_candidate = candidate.with_suffix(".py")
        if py_candidate.exists():
            return py_candidate.resolve()

        init_candidate = candidate / "__init__.py"
        if init_candidate.exists():
            return init_candidate.resolve()

    return None


def collect_local_dependencies(entry_script: Path, source_root: Path) -> Set[Path]:
    entry_script = entry_script.resolve()
    source_root = source_root.resolve()

    included: Set[Path] = set()
    pending: List[Path] = [entry_script]
    search_roots = [source_root, source_root.parent]

    while pending:
        current = pending.pop()
        if current in included:
            continue

        included.add(current)
        imports = parse_imports(current)
        for module_name, level in imports:
            local_file = resolve_local_module(
                importer=current,
                module_name=module_name,
                level=level,
                search_roots=search_roots,
            )
            if local_file is None:
                continue
            if local_file.suffix != ".py":
                continue
            if local_file not in included:
                pending.append(local_file)

    return included


def _collect_external_import_roots(files: Iterable[Path]) -> Set[str]:
    stdlib_names = set(getattr(sys, "stdlib_module_names", set()))
    import_roots: Set[str] = set()
    local_module_roots = {f.stem for f in files}

    for py_file in files:
        for module_name, level in parse_imports(py_file):
            if level > 0 or not module_name:
                continue
            root_name = module_name.split(".")[0]
            if root_name in local_module_roots:
                continue
            if root_name in stdlib_names:
                continue
            import_roots.add(root_name)

    # pandas.read_parquet typically needs pyarrow present.
    if "pandas" in import_roots:
        import_roots.add("pyarrow")
    return import_roots


def build_requirements(files: Iterable[Path]) -> List[str]:
    import_roots = _collect_external_import_roots(files)
    requirements = sorted(THIRD_PARTY_PACKAGE_MAP.get(name, name) for name in import_roots)
    return requirements


def write_runner_script(bundle_dir: Path, entry_script_name: str) -> None:
    runner_path = bundle_dir / "run_training.py"
    runner_code = f'''\
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NF_train.py inside this cloud bundle and save model metadata."
    )
    parser.add_argument("--data-path", required=True, help="Path to training parquet file.")
    parser.add_argument(
        "--output-dir",
        default="model_data",
        help="Directory where trained model outputs will be saved."
    )
    parser.add_argument(
        "--n-galaxies",
        type=int,
        default=None,
        help="Optional subsampling per simulation (same as NF_train.py)."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_root = Path(__file__).resolve().parent
    train_script = bundle_root / "src" / "{entry_script_name}"
    cmd = [
        sys.executable,
        str(train_script),
        "--data-path", str(Path(args.data_path).resolve()),
        "--output-dir", str(output_dir),
    ]
    if args.n_galaxies is not None:
        cmd.extend(["--n-galaxies", str(args.n_galaxies)])

    subprocess.run(cmd, check=True)

    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts.append(
                {{
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256sum(path),
                }}
            )

    train_config_path = output_dir / "train_config.json"
    train_config = None
    if train_config_path.exists():
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))

    info = {{
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "command": cmd,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "train_config": train_config,
    }}
    (output_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved model info to {{output_dir / 'model_info.json'}}")


if __name__ == "__main__":
    main()
'''
    runner_path.write_text(runner_code, encoding="utf-8")


def write_readme(bundle_dir: Path) -> None:
    readme = textwrap.dedent(
        """\
        # NF_train cloud bundle (Colab / Kaggle)

        This folder is generated automatically from your latest local scripts.
        It contains:
        - `src/` -> current training code and local dependencies
        - `requirements.txt` -> inferred pip dependencies
        - `run_training.py` -> launches training and writes `model_info.json`

        ## 1) Install dependencies
        ```bash
        pip install -r requirements.txt
        ```

        ## 2) Run training
        ```bash
        python run_training.py --data-path /path/to/camels_astrid_sb7_090.parquet --output-dir model_data
        ```

        Optional subsampling:
        ```bash
        python run_training.py --data-path /path/to/camels_astrid_sb7_090.parquet --output-dir model_data --n-galaxies 5000
        ```

        ## 3) Download outputs
        Download the full `model_data/` folder. It will contain:
        - `flow_state.pt`
        - `scalers.npz`
        - `loss_history.npy`
        - `train_config.json`
        - `model_info.json`
        """
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")


def zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(bundle_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone Colab/Kaggle bundle for NF_train.py "
            "from the latest local scripts."
        )
    )
    parser.add_argument(
        "--entry-script",
        default="CAMELS_codes/NF_train.py",
        help="Path to the training entry script.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="cloud_runs/nf_train_bundle",
        help="Output directory for generated bundle.",
    )
    parser.add_argument(
        "--zip-path",
        default="cloud_runs/nf_train_bundle.zip",
        help="Output zip path for easy upload to Colab/Kaggle.",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent
    entry_arg = Path(args.entry_script)
    entry_script = (entry_arg if entry_arg.is_absolute() else workspace_root / entry_arg).resolve()
    if not entry_script.exists():
        raise FileNotFoundError(f"Entry script not found: {entry_script}")

    source_root = entry_script.parent.resolve()
    bundle_arg = Path(args.bundle_dir)
    zip_arg = Path(args.zip_path)
    bundle_dir = (bundle_arg if bundle_arg.is_absolute() else workspace_root / bundle_arg).resolve()
    zip_path = (zip_arg if zip_arg.is_absolute() else workspace_root / zip_arg).resolve()

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "src").mkdir(parents=True, exist_ok=True)

    local_files = collect_local_dependencies(entry_script=entry_script, source_root=source_root)

    # Keep relative structure from source_root so imports keep working.
    copied_files: Dict[str, str] = {}
    for src_file in sorted(local_files):
        rel = src_file.relative_to(source_root)
        dst = bundle_dir / "src" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
        copied_files[str(rel)] = str(src_file)

    requirements = build_requirements(local_files)
    (bundle_dir / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")

    write_runner_script(bundle_dir=bundle_dir, entry_script_name=entry_script.name)
    write_readme(bundle_dir=bundle_dir)

    manifest = {
        "entry_script": str(entry_script),
        "source_root": str(source_root),
        "bundle_dir": str(bundle_dir),
        "files_included": copied_files,
        "requirements": requirements,
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    zip_bundle(bundle_dir=bundle_dir, zip_path=zip_path)

    print(f"Bundle created: {bundle_dir}")
    print(f"Zip created: {zip_path}")
    print(f"Files included: {len(copied_files)}")
    for rel in sorted(copied_files):
        print(f"  - {rel}")


if __name__ == "__main__":
    main()
