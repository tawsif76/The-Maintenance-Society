"""Compile the PRIMEarxiv LaTeX preprint with pdfLaTeX and BibTeX."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "results/preprint/latex"


def run_command(command, directory):
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("\n".join(result.stdout.splitlines()[-30:]))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "main-primearxiv.pdf")
    args = parser.parse_args()
    pdflatex, bibtex = shutil.which("pdflatex"), shutil.which("bibtex")
    if not pdflatex or not bibtex:
        parser.exit(1, "Install a LaTeX distribution with pdflatex and bibtex to build the preprint.\n")
    for name in ("main.tex", "ref.bib", "PRIMEarxiv.sty"):
        if not (PROJECT / name).is_file():
            parser.exit(1, f"Missing LaTeX source: {PROJECT / name}\n")
    try:
        with tempfile.TemporaryDirectory(prefix="maintenance-latex-") as folder:
            build = Path(folder)
            shutil.copy2(PROJECT / "ref.bib", build / "ref.bib")
            command = [pdflatex, "-no-shell-escape", "-interaction=nonstopmode",
                       "-halt-on-error", "-file-line-error", f"-output-directory={build}", "main.tex"]
            if "miktex" in pdflatex.lower():
                command.insert(1, "--disable-installer")
            run_command(command, PROJECT)
            run_command([bibtex, "main"], build)
            run_command(command, PROJECT)
            run_command(command, PROJECT)
            log = (build / "main.log").read_text(encoding="utf-8", errors="replace")
            undefined = len(re.findall(r"(?:Citation|Reference) .*? undefined", log))
            overfull = log.count("Overfull \\hbox")
            if undefined or overfull or "There were undefined references" in log:
                raise RuntimeError(f"Resolve layout/citation warnings before delivery: "
                                   f"{undefined} undefined references or citations; {overfull} overfull boxes.")
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(build / "main.pdf", output)
            shutil.copy2(build / "main.bbl", PROJECT / "main.bbl")
            pages = re.search(r"Output written on[\s\S]*?\((\d+)\s+pages?", log)
            print(f"PDF: {output}")
            if pages:
                print(f"Pages: {pages[1]}")
            print("Undefined references/citations: 0; overfull boxes: 0.")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Cannot build preprint:\n{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
