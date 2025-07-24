import os
import time
import pandas as pd
import numpy as np
from gwpy.timeseries import TimeSeries
import requests
import io

# --- config ---
BASE_PATH = 'data'
DETECTORS = ['H1', 'L1']

# Signal Config
SIGNAL_CATALOGS = ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident']
SIGNAL_DURATION = 32
SIGNAL_SAMPLE_RATE = 4096

# Glitch Config
GLITCH_METADATA_FILES = [
    'H1_O3a.csv', 'L1_O3a.csv', 'H1_O3b.csv', 'L1_O3b.csv'
]
GLITCH_CLASSES = [
    "Blip", "Koi_Fish", "Tomte", "Whistle", "Scratching",
    "Power_Line", "No_Glitch", "Light_Modulation"
]

# reduced number of glitches per class for testing
NUM_GLITCHES_PER_CLASS = 150
GLITCH_DURATION = 4
GLITCH_SAMPLE_RATE = 4096


def setup_directories():
    """Creates the directory structure for storing the datasets."""
    print("--- Setting up directories ---")
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(BASE_PATH, 'signals', split), exist_ok=True)
        os.makedirs(os.path.join(BASE_PATH, 'glitches', split), exist_ok=True)
    print("Directory setup complete.\n")


def download_real_signals():
    """Downloads and classifies real signals from multiple GWOSC catalogs."""
    print("--- Downloading Real Signal Data (BBH and BNS) ---")
    all_events = []
    for catalog in SIGNAL_CATALOGS:
        try:
            catalog_url = f'https://gwosc.org/eventapi/csv/{catalog}/'
            print(f"Fetching catalog: {catalog}")
            response = requests.get(catalog_url).content
            df = pd.read_csv(io.StringIO(response.decode('utf-8')))
            df.rename(columns={'gps': 'GPS', 'commonName': 'name'}, inplace=True)
            all_events.append(df)
        except Exception as e:
            print(f"  - Could not fetch or process {catalog}. Error: {e}")

    if not all_events:
        print("FATAL: No event catalogs could be downloaded. Exiting signal download.")
        return

    events_df = pd.concat(all_events, ignore_index=True).drop_duplicates(subset=['name']).set_index('name')
    events_df = events_df.dropna(subset=['mass_1_source', 'mass_2_source', 'GPS'])
    events_df['type'] = np.where(
        (events_df['mass_1_source'] < 3) & (events_df['mass_2_source'] < 3), 'BNS', 'BBH'
    )
    print(f"\nFound {len(events_df)} total unique events across all catalogs.")
    print(events_df['type'].value_counts())

    shuffled_df = events_df.sample(frac=1)
    train_df, val_df, test_df = np.split(shuffled_df, [int(.8*len(shuffled_df)), int(.9*len(shuffled_df))])
    splits = {'train': train_df, 'validation': val_df, 'test': test_df}

    for split_name, df in splits.items():
        print(f"\nProcessing {split_name} set ({len(df)} events)...")
        for name, row in df.iterrows():
            for detector in DETECTORS:
                try:
                    time.sleep(1)
                    
                    gps_int = int(row['GPS'])
                    t0, t1 = gps_int - (SIGNAL_DURATION // 2), gps_int + (SIGNAL_DURATION // 2)
                    filename = f"{row['type']}_{detector}_{gps_int}.gwf"
                    filepath = os.path.join(BASE_PATH, 'signals', split_name, filename)
                    if os.path.exists(filepath): continue
                    
                    print(f"  Downloading {row['type']} {name} for {detector}...")
                    data = TimeSeries.fetch_open_data(detector, t0, t1, sample_rate=SIGNAL_SAMPLE_RATE, cache=True)
                    data.write(filepath)
                except Exception as e:
                    print(f"    - Could not download {row['type']} {name} for {detector}: {e}")


def download_glitch_data():
    """Downloads glitch data using the public Gravity Spy metadata from Zenodo."""
    print("\n--- Downloading Glitch Data from Public Gravity Spy Dataset ---")
    
    zenodo_base_url = "https://zenodo.org/records/5649212/files/"
    
    all_glitches = []
    for filename in GLITCH_METADATA_FILES:
        try:
            metadata_url = f"{zenodo_base_url}{filename}"
            print(f"Fetching glitch metadata from: {metadata_url}")
            df = pd.read_csv(metadata_url)
            
            detector = filename.split('_')[0]
            df['detector'] = detector
            
            df.rename(columns={'peak_time': 'GPS', 'ml_label': 'type'}, inplace=True)
            
            df = df[df['type'].isin(GLITCH_CLASSES)]
            
            all_glitches.append(df)
            print(f"  - Successfully loaded {len(df)} glitches from {filename}")
            
        except Exception as e:
            print(f"  - Could not read metadata from {filename}. Error: {e}")
            
    if not all_glitches:
        print("FATAL: No glitch metadata could be downloaded. Exiting glitch download.")
        return
        
    glitches_df = pd.concat(all_glitches, ignore_index=True)
    print(f"\nFound metadata for {len(glitches_df)} total glitches.")
    print("Glitch class distribution:")
    print(glitches_df['type'].value_counts())
    
    shuffled_df = glitches_df.sample(frac=1)
    train_df, val_df, test_df = np.split(shuffled_df, [int(.8*len(shuffled_df)), int(.9*len(shuffled_df))])
    splits = {'train': train_df, 'validation': val_df, 'test': test_df}

    for split_name, df in splits.items():
        print(f"\nProcessing {split_name} set ({len(df)} glitches)...")
        for g_class in GLITCH_CLASSES:
            class_df = df[df['type'] == g_class]
            if len(class_df) == 0:
                print(f"  No '{g_class}' glitches found in {split_name} set")
                continue
                
            split_multiplier = {'train': 1.0, 'validation': 0.125, 'test': 0.125}
            max_samples = int(NUM_GLITCHES_PER_CLASS * split_multiplier[split_name])
            sample_df = class_df.sample(n=min(len(class_df), max_samples))
            
            print(f"  Downloading {len(sample_df)} '{g_class}' glitches...")
            for _, row in sample_df.iterrows():
                try:
                    time.sleep(1)
                    
                    detector = row['detector']
                    gps_time = float(row['GPS'])
                    gps_int = int(gps_time)
                    
                    t0, t1 = gps_time - (GLITCH_DURATION // 2), gps_time + (GLITCH_DURATION // 2)
                    out_filename = f"{g_class}_{detector}_{gps_int}.gwf"
                    filepath = os.path.join(BASE_PATH, 'glitches', split_name, out_filename)
                    
                    if os.path.exists(filepath): 
                        continue
                    
                    data = TimeSeries.fetch_open_data(detector, t0, t1, sample_rate=GLITCH_SAMPLE_RATE, cache=True)
                    data.write(filepath)
                    
                except Exception as e:
                    print(f"    - Could not download glitch at {row['GPS']}: {e}")


def verify_downloads():
    """Verify that data was downloaded successfully."""
    print("\n--- Verifying Downloads ---")
    
    for data_type in ['signals', 'glitches']:
        print(f"\n{data_type.upper()} Summary:")
        for split in ['train', 'validation', 'test']:
            path = os.path.join(BASE_PATH, data_type, split)
            if os.path.exists(path):
                files = os.listdir(path)
                print(f"  {split}: {len(files)} files")
                
                if data_type == 'glitches' and files:
                    class_counts = {}
                    for file in files:
                        class_name = file.split('_')[0]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    for class_name, count in sorted(class_counts.items()):
                        print(f"    - {class_name}: {count}")
                        
                elif data_type == 'signals' and files:
                    type_counts = {'BBH': 0, 'BNS': 0}
                    for file in files:
                        if file.startswith('BBH'):
                            type_counts['BBH'] += 1
                        elif file.startswith('BNS'):
                            type_counts['BNS'] += 1
                    
                    for signal_type, count in type_counts.items():
                        print(f"    - {signal_type}: {count}")
            else:
                print(f"  {split}: Directory not found")


if __name__ == '__main__':
    setup_directories()
    download_real_signals()
    download_glitch_data()
    verify_downloads()
    
    print("\n==========================")
    print("Data download complete.")
    print(f"All data has been saved to the '{BASE_PATH}' directory.")
    print("==========================")