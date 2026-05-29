# save as fix_notebooks.py in project root
import json
import os

notebooks = [
    "notebooks/03_evaluate.ipynb",
    "notebooks/04_optuna_hpo.ipynb",
]

for path in notebooks:
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Remove widgets from metadata
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        print(f"✅ Removed metadata.widgets from {path}")

    # Remove widgets from each cell
    for cell in nb.get("cells", []):
        if "widgets" in cell.get("metadata", {}):
            del cell["metadata"]["widgets"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ Fixed: {path}")

print("\nDone! Now run:")
print("git add notebooks/03_evaluate.ipynb notebooks/04_optuna_hpo.ipynb")
print("git commit -m '🔧 Fix widget metadata for GitHub rendering'")
print("git push origin main")