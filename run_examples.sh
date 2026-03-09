#!/usr/bin/env bash
set -e

echo "Available main scripts:"
ls -1 *_final.py extract_all_features.py competition_beater.py 2>/dev/null || true
