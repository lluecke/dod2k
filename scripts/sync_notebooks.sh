#!/usr/bin/env python3
import os
from pathlib import Path

# Directories
docs_notebooks = Path("docs/notebooks")
source_notebooks = Path("notebooks")

# Create docs/notebooks directory
docs_notebooks.mkdir(parents=True, exist_ok=True)

# Remove old symlinks
for old_link in docs_notebooks.glob("*.ipynb"):
    if old_link.is_symlink():
        old_link.unlink()
        print(f"Removed symlink: {old_link.name}")
    else:
        print(f"Skipped real file: {old_link.name}")


# Create new symlinks
for notebook in source_notebooks.glob("*.ipynb"):
    target = docs_notebooks / notebook.name
    relative_source = Path("../../notebooks") / notebook.name
    target.symlink_to(relative_source)
    print(f"Linked: {notebook.name}")

print("Notebook symlinks updated!")