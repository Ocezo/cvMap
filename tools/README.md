# Tools: images generation

## LaTeX compilation

## To generate the empty grid PDF file

Install the PDFLaTex building tool and do:

```
pdflatex grid.tex
pdflatex grid.tex
rm grid.aux grid.log
```

## ImageMagick generation

## To resize the PDF to a JPEG file

Convert the PDF to a JPEG file.
Resize it to A4 ratio: 1226 x 1734 px:

```
convert grid.pdf grid.jpg
convert grid.jpg -resize 1226x1734! grid.jpg
```
