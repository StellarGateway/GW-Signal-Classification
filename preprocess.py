import os
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class GWDataPreprocessor:
    
    def __init__(self, 
                 bandpass_freq=(35, 350),
                 notch_freqs=[60, 120, 180],
                 whitening_duration=4,
                 output_sample_rate=2048):
        
        self.bandpass_freq = bandpass_freq
        self.notch_freqs = notch_freqs
        self.whitening_duration = whitening_duration
        self.output_sample_rate = output_sample_rate
        self.scalers = {}
        
    def get_channel_names(self, filepath):
        try:
            from gwpy.io.gwf import get_channel_names
            channels = get_channel_names(filepath)
            return channels
        except Exception as e:
            print(f"Error getting channel names: {e}")
            return []
    
    def load_strain_data(self, filepath, channel='Strain'):
        
        try:
            data = TimeSeries.read(filepath, channel)
            return data
        except:
            try:
                channels = self.get_channel_names(filepath)
                print(f"Available channels in {os.path.basename(filepath)}: {channels}")
                
                strain_patterns = ['Strain', 'STRAIN', 'DCS-CALIB_STRAIN_C01', 'GDS-CALIB_STRAIN']
                
                for pattern in strain_patterns:
                    matching_channels = [ch for ch in channels if pattern in ch]
                    if matching_channels:
                        print(f"Using channel: {matching_channels[0]}")
                        return TimeSeries.read(filepath, matching_channels[0])
                
                if channels:
                    print(f"No strain channel found, using: {channels[0]}")
                    return TimeSeries.read(filepath, channels[0])
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        
        return None
    
    def apply_bandpass_filter(self, data, low_freq=None, high_freq=None):
        low_freq = low_freq or self.bandpass_freq[0]
        high_freq = high_freq or self.bandpass_freq[1]
        
        try:
            filtered_data = data.bandpass(low_freq, high_freq)
            return filtered_data
        except Exception as e:
            print(f"Bandpass filtering failed: {e}")
            return data
    
    def apply_notch_filter(self, data, freqs=None):
        freqs = freqs or self.notch_freqs
        
        try:
            filtered_data = data.copy()
            for freq in freqs:
                filtered_data = filtered_data.notch(freq)
            return filtered_data
        except Exception as e:
            print(f"Notch filtering failed: {e}")
            return data
    
    def whiten_data(self, data, duration=None, method='median'):
        
        duration = duration or self.whitening_duration
        
        try:
            asd = data.asd(fftlength=duration, method=method)
            
            whitened = data.whiten(asd=asd)
            return whitened
        except Exception as e:
            print(f"Whitening failed: {e}")
            return data
    
    def resample_data(self, data, target_rate=None):
        target_rate = target_rate or self.output_sample_rate
        
        if data.sample_rate != target_rate:
            try:
                resampled = data.resample(target_rate)
                return resampled
            except Exception as e:
                print(f"Resampling failed: {e}")
                return data
        return data
    
    def remove_outliers(self, data, threshold=5):
        try:
            z_scores = np.abs((data.value - np.mean(data.value)) / np.std(data.value))
            mask = z_scores < threshold
            
            clean_data = data.copy()
            outlier_indices = np.where(~mask)[0]
            
            if len(outlier_indices) > 0:
                for idx in outlier_indices:
                    if idx > 0 and idx < len(data.value) - 1:
                        clean_data.value[idx] = (clean_data.value[idx-1] + clean_data.value[idx+1]) / 2
                        
            return clean_data
        except Exception as e:
            print(f"Outlier removal failed: {e}")
            return data
    
    def extract_features(self, data):
        
        features = {}
        
        try:
            signal_data = np.array(data.value)
            sample_rate = float(data.sample_rate.value)
            
            features['mean'] = float(np.mean(signal_data))
            features['std'] = float(np.std(signal_data))
            features['var'] = float(np.var(signal_data))
            features['skewness'] = float(pd.Series(signal_data).skew())
            features['kurtosis'] = float(pd.Series(signal_data).kurtosis())
            features['rms'] = float(np.sqrt(np.mean(signal_data**2)))
            features['peak_to_peak'] = float(np.ptp(signal_data))
            features['zero_crossings'] = int(len(np.where(np.diff(np.sign(signal_data)))[0]))
            
            freqs, psd = welch(signal_data, fs=sample_rate, nperseg=1024)
            features['dominant_freq'] = float(freqs[np.argmax(psd)])
            features['spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd))
            features['spectral_spread'] = float(np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd)))
            features['spectral_rolloff'] = float(freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]])
            
            low_band = (freqs >= 35) & (freqs <= 100)
            mid_band = (freqs >= 100) & (freqs <= 250)
            high_band = (freqs >= 250) & (freqs <= 350)
            
            features['energy_low_band'] = float(np.sum(psd[low_band]))
            features['energy_mid_band'] = float(np.sum(psd[mid_band]))
            features['energy_high_band'] = float(np.sum(psd[high_band]))
            features['energy_ratio_low_high'] = float(features['energy_low_band'] / (features['energy_high_band'] + 1e-10))
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            
        return features
    
    def create_spectrogram(self, data, nperseg=256, overlap=0.5):
        try:
            data_length = len(data)
            data_duration = float(data.duration.value)
            sample_rate = float(data.sample_rate.value)
            
            if data_duration < 0.1:
                print(f"  Data too short for spectrogram: {data_duration:.4f}s")
                return None
            
            max_nperseg = data_length // 4
            nperseg = min(nperseg, max_nperseg)
            
            if nperseg < 64:
                nperseg = max(32, data_length // 8)
            
            if nperseg < 16:
                print(f"  Data too short for meaningful spectrogram: nperseg={nperseg}")
                return None
            
            fftlength = nperseg / sample_rate
            
            fftlength = min(fftlength, data_duration / 4)
            
            stride_samples = int(nperseg * (1 - overlap))
            stride_time = stride_samples / sample_rate
            
            max_stride_time = data_duration / 10
            stride_time = min(stride_time, max_stride_time)
            
            if stride_time < fftlength:
                stride_time = fftlength
                print(f"  Adjusted stride to match fftlength: {stride_time:.4f}s")
            
            min_stride_time = 1.0 / sample_rate
            stride_time = max(stride_time, min_stride_time)
            
            if stride_time > data_duration / 3:
                fftlength = min(fftlength, data_duration / 3)
                stride_time = fftlength
                print(f"  Using no overlap: stride=fftlength={stride_time:.4f}s")
            
            print(f"  Creating spectrogram: stride={stride_time:.4f}s, fftlength={fftlength:.4f}s, data_duration={data_duration:.4f}s")
            
            spectrogram = data.spectrogram(stride_time, fftlength=fftlength)
            return spectrogram
        except Exception as e:
            print(f"Spectrogram creation failed: {e}")
            print(f"  Data duration: {data_duration:.4f}s, length: {data_length} samples")
            return None
    
    def normalize_data(self, data, method='standard', fit_scaler=True):
        
        if method not in self.scalers and fit_scaler:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
            
        if method in self.scalers:
            if fit_scaler:
                normalized = self.scalers[method].fit_transform(data.reshape(-1, 1)).flatten()
            else:
                normalized = self.scalers[method].transform(data.reshape(-1, 1)).flatten()
        else:
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
            
        return normalized
    
    def diagnose_data_quality(self, filepath):
        print(f"\nData Quality Diagnosis: {os.path.basename(filepath)}")
        
        try:
            data = self.load_strain_data(filepath)
            if data is None:
                print("Could not load data")
                return
            
            signal_data = np.array(data.value)
            sample_rate = float(data.sample_rate.value)
            duration = float(data.duration.value)
            
            print(f"Basic Properties:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Data points: {len(signal_data)}")
            print(f"   Data range: [{np.min(signal_data):.2e}, {np.max(signal_data):.2e}]")
            
            print(f"\nQuality Checks:")
            
            zero_count = np.sum(signal_data == 0)
            if zero_count > len(signal_data) * 0.01:
                print(f"   {zero_count} zeros found ({zero_count/len(signal_data)*100:.1f}%)")
            else:
                print(f"   Zero values: {zero_count} ({zero_count/len(signal_data)*100:.3f}%)")
            
            z_scores = np.abs((signal_data - np.mean(signal_data)) / np.std(signal_data))
            outliers = np.sum(z_scores > 5)
            if outliers > len(signal_data) * 0.001:
                print(f"   {outliers} outliers found ({outliers/len(signal_data)*100:.3f}%)")
            else:
                print(f"   Outliers (>5σ): {outliers} ({outliers/len(signal_data)*100:.3f}%)")
            
            dynamic_range = np.ptp(signal_data) / np.std(signal_data)
            if dynamic_range < 6:
                print(f"   Low dynamic range: {dynamic_range:.2f} (might appear flat in spectrogram)")
            else:
                print(f"   Dynamic range: {dynamic_range:.2f}")
            
            try:
                freqs, psd = welch(signal_data, fs=sample_rate, nperseg=min(1024, len(signal_data)//4))
                
                dominant_freq = freqs[np.argmax(psd)]
                print(f"   Dominant frequency: {dominant_freq:.1f} Hz")
                
                detection_band = (freqs >= 35) & (freqs <= 350)
                power_in_band = np.sum(psd[detection_band]) / np.sum(psd)
                print(f"   Power in detection band (35-350 Hz): {power_in_band*100:.1f}%")
                
                if power_in_band < 0.1:
                    print(f"   Very little power in detection band - data might be mostly noise")
                
            except Exception as e:
                print(f"   Frequency analysis failed: {e}")
            
            filename = os.path.basename(filepath)
            if filename.startswith('BBH') or filename.startswith('BNS'):
                expected_duration = 32
                if abs(duration - expected_duration) > 1:
                    print(f"   Unexpected duration for signal: {duration:.2f}s (expected ~{expected_duration}s)")
                else:
                    print(f"   Signal duration appropriate: {duration:.2f}s")
            else:
                expected_duration = 4
                if abs(duration - expected_duration) > 1:
                    print(f"   Unexpected duration for glitch: {duration:.2f}s (expected ~{expected_duration}s)")
                else:
                    print(f"   Glitch duration appropriate: {duration:.2f}s")
            
            print(f"\nRecommendations:")
            if dynamic_range < 6:
                print(f"   • Consider adjusting spectrogram color scale (use percentile-based vmin/vmax)")
                print(f"   • Check if this is actually a 'quiet' period with little signal")
            
            if power_in_band < 0.1:
                print(f"   • Verify this is the correct data - most power should be 35-350 Hz for GW data")
                print(f"   • Consider whether bandpass filtering is too aggressive")
            
            if outliers > len(signal_data) * 0.001:
                print(f"   • Consider more aggressive outlier removal")
                print(f"   • Check for instrumental glitches or data quality flags")
            
            print(f"   • Use percentile-based color scaling for spectrograms")
            print(f"   • Consider smoothing ASD for whitening to avoid over-whitening")
            
        except Exception as e:
            print(f"Diagnosis failed: {e}")

    def preprocess_single_file(self, filepath, apply_whitening=True, extract_features_flag=True, create_spectrogram_flag=True):
        
        print(f"Processing: {os.path.basename(filepath)}")
        
        result = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'success': False,
            'error': None
        }
        
        try:
            data = self.load_strain_data(filepath)
            if data is None:
                result['error'] = "Failed to load data"
                return result
            
            print(f"  Original: {len(data)} samples at {data.sample_rate} Hz")
            
            result['original_duration'] = float(data.duration.value)
            result['original_sample_rate'] = float(data.sample_rate.value)
            result['original_length'] = len(data)
            
            data = self.remove_outliers(data)
            
            data = self.apply_bandpass_filter(data)
            
            data = self.apply_notch_filter(data)
            
            data = self.resample_data(data)
            
            if apply_whitening:
                data = self.whiten_data(data)
            
            print(f"  Processed: {len(data)} samples at {data.sample_rate} Hz")
            
            result['processed_data'] = data.value
            result['processed_times'] = data.times.value
            result['sample_rate'] = float(data.sample_rate.value)
            result['duration'] = float(data.duration.value)
            
            if extract_features_flag:
                result['features'] = self.extract_features(data)
            
            if create_spectrogram_flag:
                spec = self.create_spectrogram(data)
                if spec is not None:
                    result['spectrogram'] = {
                        'frequencies': spec.frequencies.value,
                        'times': spec.times.value,
                        'power': spec.value
                    }
            
            result['success'] = True
            print(f"  Successfully processed")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  Processing failed: {e}")
        
        return result
    
    def preprocess_dataset(self, data_dir, output_dir=None, max_files=None):
        
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist")
            return None
        
        gwf_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.gwf'):
                    gwf_files.append(os.path.join(root, file))
        
        if max_files:
            gwf_files = gwf_files[:max_files]
        
        print(f"Found {len(gwf_files)} GWF files to process")
        
        results = []
        successful = 0
        failed = 0
        
        for i, filepath in enumerate(gwf_files):
            print(f"\nProgress: {i+1}/{len(gwf_files)}")
            
            result = self.preprocess_single_file(filepath)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        print(f"\nProcessing Complete")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            metadata = []
            features_list = []
            
            for result in results:
                if result['success']:
                    meta = {
                        'filename': result['filename'],
                        'original_duration': result['original_duration'],
                        'original_sample_rate': result['original_sample_rate'],
                        'processed_sample_rate': result['sample_rate'],
                        'duration': result['duration']
                    }
                    
                    filename = result['filename']
                    if filename.startswith('BBH'):
                        meta['signal_type'] = 'BBH'
                        meta['class'] = 'signal'
                    elif filename.startswith('BNS'):
                        meta['signal_type'] = 'BNS'
                        meta['class'] = 'signal'
                    else:
                        meta['signal_type'] = filename.split('_')[0]
                        meta['class'] = 'glitch'
                    
                    metadata.append(meta)
                    
                    if 'features' in result:
                        features = result['features'].copy()
                        features['filename'] = result['filename']
                        features['class'] = meta['class']
                        features['signal_type'] = meta['signal_type']
                        features_list.append(features)
                    
                    data_filename = result['filename'].replace('.gwf', '_processed.npy')
                    np.save(os.path.join(output_dir, data_filename), result['processed_data'])
                    
                    if 'spectrogram' in result:
                        spec_filename = result['filename'].replace('.gwf', '_spectrogram.npz')
                        np.savez(os.path.join(output_dir, spec_filename), **result['spectrogram'])
            
            if metadata:
                pd.DataFrame(metadata).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
            
            if features_list:
                pd.DataFrame(features_list).to_csv(os.path.join(output_dir, 'features.csv'), index=False)
            
            print(f"Results saved to: {output_dir}")
        
        return results
    
    def plot_preprocessing_comparison(self, filepath, save_path=None, show_plot=False):
        original_data = self.load_strain_data(filepath)
        if original_data is None:
            print("Could not load data for comparison")
            return
        
        processed_result = self.preprocess_single_file(filepath, create_spectrogram_flag=False)
        if not processed_result['success']:
            print("Processing failed for comparison")
            return
        
        processed_data = processed_result['processed_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Comparison: {os.path.basename(filepath)}', fontsize=16)
        
        axes[0,0].plot(original_data.times.value, original_data.value)
        axes[0,0].set_title('Original Time Series')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Strain')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(processed_result['processed_times'], processed_data)
        axes[0,1].set_title('Processed Time Series')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Strain (whitened)')
        axes[0,1].grid(True, alpha=0.3)
        
        freqs_orig, psd_orig = welch(original_data.value, fs=float(original_data.sample_rate.value), nperseg=1024)
        axes[1,0].loglog(freqs_orig, psd_orig)
        axes[1,0].set_title('Original Power Spectral Density')
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('PSD [strain²/Hz]')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_xlim([10, 1000])
        
        freqs_proc, psd_proc = welch(processed_data, fs=processed_result['sample_rate'], nperseg=1024)
        axes[1,1].loglog(freqs_proc, psd_proc)
        axes[1,1].set_title('Processed Power Spectral Density')
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('PSD [whitened strain²/Hz]')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim([10, 1000])
        
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(filepath).replace('.gwf', '')
            save_path = f"test_plots/{filename}_preprocessing_comparison"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Comparison plots saved: {save_path}.png/.pdf")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def test_preprocessor():
    print("Testing GW Data Preprocessor")
    
    preprocessor = GWDataPreprocessor(
        bandpass_freq=(35, 350),
        notch_freqs=[60, 120, 180],
        whitening_duration=4,
        output_sample_rate=2048
    )
    
    os.makedirs("test_plots", exist_ok=True)
    
    test_files = [
        "data/signals/train/BBH_H1_1238303737.gwf",
        "data/signals/train/BNS_H1_1187008882.gwf",
        "data/glitches/train/Blip_H1_1238475101.gwf",
        "data/glitches/train/Whistle_H1_1244702784.gwf",
    ]
    
    successful_tests = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting file: {os.path.basename(test_file)}")
            
            result = preprocessor.preprocess_single_file(test_file)
            
            if result['success']:
                print(f"Successfully processed {result['filename']}")
                print(f"  Duration: {result['duration']:.2f} seconds")
                print(f"  Sample rate: {result['sample_rate']} Hz")
                print(f"  Data shape: {result['processed_data'].shape}")
                
                if 'features' in result:
                    print(f"  Extracted {len(result['features'])} features")
                    
                preprocessor.plot_preprocessing_comparison(test_file, show_plot=False)
                successful_tests += 1
                
            else:
                print(f"Processing failed: {result['error']}")
        else:
            print(f"Test file not found: {test_file}")
    
    print(f"\nTest Summary: {successful_tests}/{len(test_files)} files processed successfully")
    if successful_tests > 0:
        print(f"Comparison plots saved in test_plots/ directory")
        print(f"Plots are saved in both PNG and PDF formats for paper use")


def preprocess_full_dataset():
    print("Processing Full Dataset")
    
    preprocessor = GWDataPreprocessor()
    
    for split in ['train', 'validation', 'test']:
        signal_dir = f"data/signals/{split}"
        glitch_dir = f"data/glitches/{split}"
        
        if os.path.exists(signal_dir):
            print(f"\nProcessing signals - {split} set")
            preprocessor.preprocess_dataset(
                signal_dir, 
                output_dir=f"processed_data/signals/{split}",
                max_files=None
            )
        
        if os.path.exists(glitch_dir):
            print(f"\nProcessing glitches - {split} set")
            preprocessor.preprocess_dataset(
                glitch_dir, 
                output_dir=f"processed_data/glitches/{split}",
                max_files=None
            )
    
    print("\nFull dataset processing complete!")
    print("Processed data saved in processed_data/ directory")
    print("Features and metadata saved as CSV files")
    print("Spectrograms saved as NPZ files")
    print("Time series data saved as NPY files")


if __name__ == "__main__":
    test_preprocessor()
    
    print("\n" + "="*60)
    print("FULL DATASET PROCESSING")
    print("="*60)
    print("This will process the entire dataset. This may take a long time.")
    print("Estimated time: 10-30 minutes depending on dataset size")
    
    preprocess_full_dataset()