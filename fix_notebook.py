import json

path = "notebooks/03_evaluate.ipynb"

# ── Read with explicit UTF-8 encoding ─────────────────────
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Remove widgets metadata ────────────────────────────────
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]
    print("✅ Removed metadata.widgets")

for cell in nb.get("cells", []):
    if "widgets" in cell.get("metadata", {}):
        del cell["metadata"]["widgets"]

# ── Write with explicit UTF-8 encoding ────────────────────
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook fixed!")
print("   Run: git add notebooks/03_evaluate.ipynb")
print("   Run: git commit -m '🔧 Fix widget metadata'")
print("   Run: git push origin main")