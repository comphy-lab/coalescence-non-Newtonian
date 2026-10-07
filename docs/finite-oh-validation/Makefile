NAME ?= finite-oh-validation-v5
OUT ?= build

.PHONY: report
report:
	test ! -e "$(OUT)/$(NAME).pdf"
	mkdir -p "$(OUT)"
	latexmk -pdf -halt-on-error -interaction=nonstopmode -jobname="$(NAME)" -outdir="$(OUT)" main.tex
