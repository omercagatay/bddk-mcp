"""Maintainer tool: repair one PDF's text layer and recognise its formula and table regions.

Run ``python -m scripts.document_repair --help``; the workflow is in docs/DOCUMENT_QUALITY.md
("Repairing a document from its PDF"). The tool writes only into its output directory; it never opens the
corpus file or the seed. ``recognize.py`` beside this package runs on the GPU host and is never imported here.
"""
