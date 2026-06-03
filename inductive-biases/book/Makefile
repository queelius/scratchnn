# Makefile for the book. Run `make` to build book.pdf (pdflatex + biber).
BOOK  = book
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: $(BOOK).pdf

# Depend on every source subfile so edits to chapters/frontmatter/figures
# trigger a rebuild. The old list named only parts/*.tex, so chapter edits
# left book.pdf stale. Use $(wildcard) so empty globs vanish instead of
# becoming literal (missing) prerequisites.
SOURCES = $(BOOK).tex preamble.tex alex.sty references.bib \
          $(wildcard parts/*.tex) $(wildcard chapters/*.tex) \
          $(wildcard chapters/*/*.tex) $(wildcard frontmatter/*.tex) \
          $(wildcard appendices/*.tex) $(wildcard figures/*)

$(BOOK).pdf: $(SOURCES)
	$(LATEX) $(BOOK).tex
	biber $(BOOK)
	$(LATEX) $(BOOK).tex
	$(LATEX) $(BOOK).tex

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.toc *.run.xml *.lof *.lot
	rm -f parts/*.aux chapters/*.aux frontmatter/*.aux appendices/*.aux

cleanall: clean
	rm -f $(BOOK).pdf

.PHONY: all clean cleanall
