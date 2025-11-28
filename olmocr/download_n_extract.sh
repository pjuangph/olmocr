#!/bin/bash

# downloading dataset 
# python -m data.prepare_olmocrmix \
#   --dataset-path allenai/olmOCR-mix-1025 \
#   --subset 00_documents \
#   --split train \
#   --destination ../data/olmOCR-mix-1025

# Extracting tar

# 00_documents
python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 00_documents \
  --split train \
  --destination ../data/olmOCR-mix-1025

python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 00_documents \
  --split eval \
  --destination ../data/olmOCR-mix-1025

# 01_books
python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 01_books \
  --split train \
  --destination ../data/olmOCR-mix-1025

python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 01_books \
  --split eval \
  --destination ../data/olmOCR-mix-1025

# 02_loc_transcripts
python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 02_loc_transcripts \
  --split train \
  --destination ../data/olmOCR-mix-1025

python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 02_loc_transcripts \
  --split eval \
  --destination ../data/olmOCR-mix-1025

# 03_national_archives
python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 03_national_archives \
  --split train \
  --destination ../data/olmOCR-mix-1025

python -m data.prepare_olmocrmix \
  --dataset-path allenai/olmOCR-mix-1025 \
  --subset 03_national_archives \
  --split eval \
  --destination ../data/olmOCR-mix-1025
