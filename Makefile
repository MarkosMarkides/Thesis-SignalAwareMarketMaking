TEX = paper

.PHONY: pdf view clean

pdf:
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex
	xelatex -interaction=nonstopmode -halt-on-error $(TEX).tex

view: pdf
	open $(TEX).pdf

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).fdb_latexmk $(TEX).fls \
		$(TEX).log $(TEX).out $(TEX).synctex.gz $(TEX).toc
