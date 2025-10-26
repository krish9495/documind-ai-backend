@echo off
echo Compiling research paper...

REM First pass
pdflatex -interaction=nonstopmode research_paper.tex

REM Run BibTeX for bibliography
bibtex research_paper

REM Second pass for cross-references
pdflatex -interaction=nonstopmode research_paper.tex

REM Third pass to finalize
pdflatex -interaction=nonstopmode research_paper.tex

REM Clean up auxiliary files
del research_paper.aux research_paper.bbl research_paper.blg research_paper.log research_paper.out

echo Compilation complete! Output file: research_paper.pdf
pause