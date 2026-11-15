Unofficial NeurIPS 2026 poster templates
========================================

Made by the Quantitative Research Society for its NeurIPS 2026 attendee guide:
https://qrsntu.org/service/neurips26/posters/

These are not official NeurIPS templates and QRS is not affiliated with NeurIPS. The sizes follow the official
poster instructions, read on 5 November 2026:
https://neurips.cc/Conferences/2026/PosterInstructions

  main.tex       main conference poster, 140 cm wide x 90 cm high, landscape
  workshop.tex   workshop poster, 61 cm wide x 85 cm high, portrait
  neurips26-poster.sty   the shared style

How to use
----------
1. Upload this zip to Overleaf (New project > Upload project), or compile locally with pdflatex.
2. Open main.tex or workshop.tex and set the main document to it.
3. Fill in \title, \author, \affiliation, \session, \posterid and \paperurl. The QR code links to \paperurl.
   Your poster ID is the last six digits of your poster's page on neurips.cc/virtual/2026.
4. Replace each \placeholder with \includegraphics[width=\linewidth]{your-figure}.
5. To add logos, upload the image files and uncomment the \logos line.
6. Change the colours at the top of neurips26-poster.sty if you like.

The page is the printed size. Print at 100%, and do not print larger than the size above.
Check the official instructions before you print: sizes and rules can change.

Licence: MIT. Use and change these files freely.
