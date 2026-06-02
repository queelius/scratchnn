# Makefile for the book. Run `make` to build book.pdf (pdflatex + biber).
BOOK  = book
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: $(BOOK).pdf

$(BOOK).pdf: $(BOOK).tex preamble.tex alex.sty references.bib parts/*.tex
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
