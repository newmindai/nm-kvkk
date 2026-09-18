"""[pkg] Prompt files. Each file holds exactly the string the e08 code carried as a Python constant; the loader
strips the single trailing newline an editor adds, so `load(name)` equals the original constant byte for byte.
Writer prompts (*.j2) are Jinja templates rendered by gen_direct.py; judge and repair prompts (*.txt) use str.format."""

import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")


def load(name):
    with open(os.path.join(DIR, name), encoding="utf-8") as f:
        s = f.read()
    return s[:-1] if s.endswith("\n") else s
