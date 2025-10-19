#!/usr/bin/env python

import argparse
import pypdf
import sys
import re

description = (
    "Pypdf-based script to count relevant characters in 2AMU30 reports. "
    "It returns the number of mainmatter characters detected in the pdf. "
    "The pdf is assumed to be LaTeX-generated using the 2AMU30 report "
    "template, so that the first page contains only frontmatter "
    "(abstract, etc.) and the last page contains only the reference list."
)
epilog = (
    "Pages with an atypically low number of characters may correspond to those "
    "with a relatively large fraction of graphical content. Pages with "
    "atypically high number of characters may indicate text extraction errors "
    "(pypdf does a decent job, but is not perfect)."
)

parser = argparse.ArgumentParser(
    prog="charcount", description=description, epilog=epilog
)

def pages(pages):
    if type(pages) is str:
        pages = pages.split('-')
    if len(pages) != 2:
        raise ValueError("Wrong number of hyphens in page range")
    first = int(pages[0])
    end = int(pages[1])
    if first < 1:
        raise ValueError("Lower page range bound smaller than 1")
    return [first-1,  # pypdf counts pages from 0
            end]      # below, last page of the range is excluded

parser.add_argument('filename')
parser.add_argument('-e', '--exclude_poster', action='store_true',
                    help="exclude 1-page poster added at the end of the report")
parser.add_argument('-v', '--verbose', action='store_true',
                    help="also print extracted text for debugging purposes")
parser.add_argument('-p', '--pages', type=pages, metavar='first-last',
                    help="give an explicit page range of the form first-last")

args = parser.parse_args()

pdf = pypdf.PdfReader(args.filename)
total_pages = len(pdf.pages)

if args.pages is None:
    args.pages = pages([2,               # start on second page
                        total_pages-1])  # until (excluded) last
if args.pages[1] > total_pages:
    raise ValueError("Page range bound larger than number of pages")
if args.exclude_poster:
    args.pages[1] -= 1
    if args.pages[1] <= args.pages[0]:
        raise ValueError("Poster page cannot be only page")

total_characters = 0
for pagenumber in range(*args.pages):
    text = pdf.pages[pagenumber].extract_text()
    if args.verbose:
        print("\n", text, "\n")  # useful for investigating text extraction quality
    characters = len(
        # remove white space from string of characters
        re.sub(r"\s+", "", text, flags=re.UNICODE)
    )
    print(f"page {pagenumber+1}: {characters}")
    total_characters += characters
print(f"all counted pages: {total_characters}")
