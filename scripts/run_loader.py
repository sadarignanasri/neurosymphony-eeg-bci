import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import DataLoader


def main():
    ds_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "ds004514"
    loader = DataLoader(ds_root)
    loader.print_dataset_info('sub-01')

    try:
        raw = loader.load_raw_bdf('sub-01')
        print('Loaded raw data successfully:')
        print(raw)
    except Exception as e:
        print('Error loading raw:', e)


if __name__ == '__main__':
    main()
