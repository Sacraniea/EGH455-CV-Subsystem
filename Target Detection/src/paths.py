# src/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # adjust parents[n] to your layout

if __name__ == "__main__":
    print(f"Root directory: {ROOT}")