"""
Optional helper: merges main.py + modules/*.py + the sample data into ONE file
(resume_screener_single_file.py) that can be pasted into a single Google Colab cell.
Run:  python build_single_file.py
"""
import glob
import os
import re

ORDER = ["preprocessing", "extraction", "representation", "matching",
         "summarization", "clustering", "topic_modeling"]
MARK = "# ==== END BOOTSTRAP ===="


def strip_internal_imports(src):
    return "\n".join(l for l in src.splitlines() if not l.startswith("from modules."))


main_src = open("main.py", encoding="utf-8").read()
head, tail = main_src.split(MARK, 1)
parts = [head + MARK + "\n"]

for name in ORDER:
    src = strip_internal_imports(open(f"modules/{name}.py", encoding="utf-8").read())
    parts.append(f"\n\n# {'=' * 70}\n# MODULE FILE: modules/{name}.py\n# {'=' * 70}\n{src}\n")

# embedded sample data (used only when the data folder is missing, e.g. in a fresh Colab runtime)
sample = {}
for f in sorted(glob.glob("data/resumes/*.txt"), key=lambda x: int(re.findall(r"\d+", x)[-1])):
    sample[os.path.basename(f)] = open(f, encoding="utf-8").read()
jd = open("data/job_description.txt", encoding="utf-8").read()
parts.append(f"\n\n# {'=' * 70}\n# EMBEDDED SAMPLE DATA\n# {'=' * 70}\n")
parts.append(f"SAMPLE_RESUMES = {sample!r}\nSAMPLE_JD = {jd!r}\n")
parts.append('''

def ensure_sample_data(resume_dir, jd_path):
    """Create the sample resumes / job description if they do not exist (fresh Colab runtime)."""
    if not glob.glob(os.path.join(resume_dir, "*.txt")):
        os.makedirs(resume_dir, exist_ok=True)
        for fname, text in SAMPLE_RESUMES.items():
            open(os.path.join(resume_dir, fname), "w", encoding="utf-8").write(text)
        print(f"[info] sample resumes written to {resume_dir}/ - replace them with your own TXT files if needed")
    if not os.path.exists(jd_path):
        os.makedirs(os.path.dirname(jd_path) or ".", exist_ok=True)
        open(jd_path, "w", encoding="utf-8").write(SAMPLE_JD)
''')

body = strip_internal_imports(tail)
body = body.replace("    resumes = load_resumes(args.resumes)",
                    "    ensure_sample_data(args.resumes, args.jd)\n    resumes = load_resumes(args.resumes)")
parts.append(f"\n\n# {'=' * 70}\n# MAIN PROGRAM\n# {'=' * 70}\n{body}\n")

open("resume_screener_single_file.py", "w", encoding="utf-8").write("".join(parts))
print("Created resume_screener_single_file.py")
