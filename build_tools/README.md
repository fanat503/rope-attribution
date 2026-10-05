# The exact tectonic version the paper is built and verified with.
#
# Build:  tectonic -X compile --outdir paper/build paper/main.tex
#
# Tectonic is a single static binary with no TeX distribution to install; it
# fetches only the packages the document actually uses and caches them.
# Windows release (51 MB, deliberately not committed):
#   https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip
#
# Verified with 0.15.0 on this document: 15 pages, 0 errors, all nine figures
# resolved from figures/, every \ref and \cite resolved. The only warnings are
# Underfull hbox/vbox, which are typographic and expected in two-column layout.
version = "0.15.0"