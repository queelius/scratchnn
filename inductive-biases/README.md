# Inductive Biases in Neural Networks

A book built with the `bookwright` workflow, scaffolded inside the
`scratchnn` repository (the pure-Python neural network library the book
teaches from).

## Layout

```
inductive-biases/
  book/
    book.tex          root document (frontmatter, parts, backmatter)
    preamble.tex      shared preamble (pdflatex + biber)
    alex.sty          project notation macros
    references.bib    bibliography (biblatex)
    Makefile          build pipeline
    parts/            \part files, each \input-ing its chapters
    chapters/         chapter .tex (authored via /bookwright:write)
    frontmatter/      preface, notation, etc.
    appendices/
  notebooks/          paired Jupyter notebooks (Python + uv stack)
  papers/             source papers (git-subtree-add later, optional)
  pyproject.toml      uv environment for the notebooks
  docs/superpowers/
    bookwright.config.yaml   project settings (notebook stack, paths)
    specs/            per-part content specs (/bookwright:design)
    plans/            per-chapter plans
```

## Build

```bash
cd book && make          # pdflatex + biber + pdflatex x2 -> book.pdf
make cleanall            # remove build artifacts and the PDF
```

## Notebooks

```bash
uv sync                  # create the environment from pyproject.toml
uv pip install -e ../..  # the scratchnn library, editable
```

## Source material

The library is in the parent repo at `../src/scratchnn`. The bite-sized
blog posts the chapters grow from are in `../docs/series`, and an earlier
single-file LaTeX assembly is in `../docs/booklet`.

## Next steps

- `/bookwright:design part1` to plan Part I's chapters.
- `/bookwright:write` to draft a chapter once its plan exists.

The `soul` plugin's voice hook applies to `.tex` writes here (no em-dashes;
use commas, colons, periods, or parentheses).
