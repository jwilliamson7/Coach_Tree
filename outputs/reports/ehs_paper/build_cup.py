import pathlib
src = pathlib.Path("outputs/reports/ehs_paper/coaching_heritability_selection.tex").read_text(encoding="utf-8")

# --- extract abstract text ---
ab = src[src.index(r"\begin{abstract}")+len(r"\begin{abstract}"):src.index(r"\end{abstract}")]
ab = ab.replace(r"\noindent", "").strip()

# --- body: from \section{Introduction} up to \section*{Acknowledgments} ---
b0 = src.index(r"\section{Introduction}")
b1 = src.index(r"\section*{Acknowledgments}")
body = src[b0:b1].rstrip()
body = body.replace(r"\citep{", r"\parencite{").replace(r"\citet{", r"\textcite{")

preamble = r"""%%% EHS submission version (Cambridge cup-journal class). Compile with pdflatex + biber.
%%% cup-journal.cls + cup-logo-new.pdf must be present (from the Cambridge Medium template).
%%% Social media summary (enter in submission portal, <=120 chars):
%%%   NFL coaching lineages inherit a mentor's system, not his success: what
%%%   transmits is largely not what wins.
\documentclass[
  journal=ehs,
  manuscript=article-type,
  year=2026,
  volume=8,
]{cup-journal}

\usepackage{amsmath}
\usepackage[nopatch]{microtype}
\usepackage{booktabs}
\usepackage{graphicx}

\title{Heritability and selection of strategic traits in the NFL coaching tree}

\author{Jon Williamson}
\affiliation{Williamson Consulting Group, USA}
\email[Jon Williamson]{jonwilliamson@live.com}

\addbibresource{refs.bib}

\keywords{cultural evolution, social learning, vertical transmission, heritability, cultural fitness, sport}

\begin{document}

\begin{abstract}
%(ABSTRACT)
\end{abstract}

"""
preamble = preamble.replace("%(ABSTRACT)", ab)

backmatter = r"""
\paragraph{Acknowledgments}
This work uses play-by-play data from \texttt{nfl\_data\_py} and coaching histories
from Pro Football Reference; the WAR phenotype is from the companion Coach\_WAR
project.

\paragraph{Funding Statement}
This research received no specific grant from any funding agency, commercial or
not-for-profit sectors.

\paragraph{Competing Interests}
The author declares none.

\paragraph{Data Availability Statement}
The confirmatory analyses (C1--C3, hypotheses H1--H3) were preregistered before any
confirmatory quantity was computed on the final measurement pipeline:
\url{https://doi.org/10.17605/OSF.IO/Y2KR5} (osf.io/y2kr5). All data and analysis
code are public and seed-fixed, and the full pipeline runs from a single command:
\url{https://github.com/jwilliamson7/Coach_Tree}. The WAR phenotype is drawn from
the sibling repository \url{https://github.com/jwilliamson7/Coach_WAR}, pinned at
commit \texttt{dffe2f1}.

\paragraph{Ethical Standards}
The research uses only publicly available data on professional-sport outcomes and
involves no human or animal subjects; it meets all applicable ethical guidelines.

\paragraph{Author Contributions}
J.W. conceived the study, developed the methods and software, conducted the
analysis, produced the visualizations, and wrote the manuscript.

\printbibliography

\end{document}
"""

out = preamble + body + "\n" + backmatter
pathlib.Path("outputs/reports/ehs_paper/coaching_heritability_selection_cup.tex").write_text(out, encoding="utf-8")
print("abstract chars:", len(ab))
print("parencite:", out.count(r"\parencite{"), "textcite:", out.count(r"\textcite{"),
      "| leftover citep/citet:", out.count(r"\citep{"), out.count(r"\citet{"))
print("figures:", out.count(r"\begin{figure}"), "tables:", out.count(r"\begin{table}"))
print("wrote coaching_heritability_selection_cup.tex, chars:", len(out))
