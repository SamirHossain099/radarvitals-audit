"""Packaging guard: the staged repository must hold no working notes, no dataset-derived arrays and no manuscript.

Checks the directory, not only git (standing rule 16): a gitignored file still sits in a hand-made zip.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {"CLAUDE.md", "PLAN.md", "FINDINGS.md", "CORRECTIONS.md", "PAPER-DRAFT.md", "PAPER-OUTLINE.md",
                   "MANUSCRIPT.md", "MANUSCRIPT.docx", "MANUSCRIPT.pdf", "references-notes.md", "render_manuscript.py",
                   "test_manuscript.py", "zotero_keys.json", "references.bib", "references.ris", "references_missing.ris",
                   "stage_public_repo.py", "refs_resolved.json"}
FORBIDDEN_DIRS = {"submission", "public-repo", "src"}               # src holds the manuscript tooling
FORBIDDEN_SUFFIX = (".npz", ".npy", ".pt", ".ckpt", ".mat", ".docx", ".pdf", ".xlsx")
FORBIDDEN_PREFIX = ("make_docx", "make_refs", "build_refs", "sync_zotero", "update_zotero", "prose_scan", "register_scan",
                    "preview_refs")
EM, EN = chr(0x2014), chr(0x2013)


def walk():
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".pytest_cache", ".ruff_cache")]
        for f in files:
            yield Path(d) / f


def test_no_working_notes_or_manuscript():
    bad = [p for p in walk() if p.name in FORBIDDEN_NAMES or p.name.startswith(FORBIDDEN_PREFIX)]
    assert not bad, bad


def test_no_private_directories():
    bad = [p for p in walk() if set(p.relative_to(ROOT).parts[:-1]) & FORBIDDEN_DIRS]
    assert not bad, bad


def test_no_dataset_derived_arrays_or_documents():
    bad = [p for p in walk() if p.suffix in FORBIDDEN_SUFFIX and p.parent.name != "figures"]
    assert not bad, bad


def test_no_local_paths_in_results():
    bad = []
    for p in list((ROOT / "results").glob("*.json")) + list((ROOT / "splits").glob("*.json")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "N:\\\\" in txt or "N:/" in txt or "C:\\\\Users" in txt:
            bad.append(p.name)
    assert not bad, bad


def test_no_em_or_en_dashes_in_released_text():           # standing rule 1
    bad = []
    for p in walk():
        if p.suffix in (".md", ".py", ".cff", ".txt", ".yml", ".toml"):
            s = p.read_text(encoding="utf-8", errors="replace")
            bad += [(str(p.relative_to(ROOT)), hex(ord(c))) for c in (EM, EN) if c in s]
    assert not bad, bad
