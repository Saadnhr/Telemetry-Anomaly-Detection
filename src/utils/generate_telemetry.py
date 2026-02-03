"""
Synthetic Satellite Telemetry Generator
Generates realistic satellite telemetry time-series with injected anomalies
for anomaly detection algorithm development and testing.

Author: Saad Nhari
Date: January 2026
Project: Satellite Telemetry Anomaly Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SatelliteTelemetryGenerator:
    """
    Generates synthetic satellite telemetry data with realistic patterns
    and controlled anomaly injection.
    """
    
    def __init__(self, 
                 start_time: datetime,
                 duration_hours: float,
                 sampling_rate_hz: float = 1.0,
                 random_seed: int = 42):
        """
        Initialize telemetry generator.
        
        Args:
            start_time: Start timestamp for telemetry
            duration_hours: Duration of telemetry in hours
            sampling_rate_hz: Sampling frequency in Hz
            random_seed: Random seed for reproducibility
        """
        self.start_time = start_time
        self.duration_hours = duration_hours
        self.sampling_rate_hz = sampling_rate_hz
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Calculate number of samples
        self.n_samples = int(duration_hours * 3600 * sampling_rate_hz)
        
        # Generate timestamps
        self.timestamps = pd.date_range(
            start=start_time,
            periods=self.n_samples,
            freq=f'{int(1000/sampling_rate_hz)}ms'
        )
        
        # Anomaly labels (0 = normal, 1 = anomaly)
        self.anomaly_labels = np.zeros(self.n_samples, dtype=int)
        
    def _orbital_pattern(self, period_minutes: float = 90) -> np.ndarray:
        """Generate orbital sinusoidal pattern."""
        t = np.arange(self.n_samples) / (self.sampling_rate_hz * 60)
        return np.sin(2 * np.pi * t / period_minutes)
    
    def _eclipse_pattern(self, period_minutes: float = 90, eclipse_fraction: float = 0.35) -> np.ndarray:
        """Generate eclipse/sunlight binary pattern (1=sun, 0=eclipse)."""
        t = np.arange(self.n_samples) / (self.sampling_rate_hz * 60)
        phase = (t % period_minutes) / period_minutes
        return (phase > eclipse_fraction).astype(float)
    
    def generate_battery_voltage(self, nominal: float = 28.0, noise_std: float = 0.2) -> np.ndarray:
        """
        Generate battery voltage telemetry.
        Voltage varies with charge/discharge cycles during eclipse/sun.
        """
        eclipse = self._eclipse_pattern()
        orbital = self._orbital_pattern()
        
        # Base voltage with eclipse effects
        voltage = nominal + 2 * eclipse - 1  # Higher in sun, lower in eclipse
        voltage += 0.5 * orbital  # Small orbital variations
        voltage += np.random.normal(0, noise_std, self.n_samples)  # Sensor noise
        
        return voltage
    
    def generate_solar_current(self, max_current: float = 8.0, noise_std: float = 0.15) -> np.ndarray:
        """
        Generate solar array current.
        Current is high during sunlight, zero during eclipse.
        """
        eclipse = self._eclipse_pattern()
        orbital = self._orbital_pattern()
        
        # Current proportional to sun exposure
        current = max_current * eclipse
        current += 0.3 * eclipse * orbital  # Angle variations
        current += np.random.normal(0, noise_std, self.n_samples)
        current = np.clip(current, 0, max_current * 1.2)
        
        return current
    
    def generate_temperature(self, 
                           component: str,
                           nominal: float = 20.0,
                           noise_std: float = 1.0) -> np.ndarray:
        """
        Generate temperature telemetry for various components.
        Temperature varies with orbital thermal cycles.
        """
        eclipse = self._eclipse_pattern()
        orbital = self._orbital_pattern()
        
        if component == 'cpu':
            # CPU temperature varies with processing load + thermal cycle
            load_variation = np.random.uniform(0.8, 1.2, self.n_samples)
            temp = nominal + 5 * load_variation
            temp += 3 * orbital  # Orbital thermal variation
            temp -= 2 * (1 - eclipse)  # Cooler in eclipse
            
        elif component == 'battery':
            # Battery temperature lags thermal cycle
            temp = nominal + 5 * eclipse - 2.5
            # Add thermal inertia (moving average)
            temp = pd.Series(temp).rolling(window=60, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            temp += 2 * orbital
            
        else:
            temp = nominal + 5 * orbital
        
        temp += np.random.normal(0, noise_std, self.n_samples)
        
        return temp
    
    def generate_attitude(self, axis: str, noise_std: float = 0.5) -> np.ndarray:
        """
        Generate attitude angles (roll, pitch, yaw).
        Stable with small perturbations.
        """
        # Target attitude (mostly stable with small drift)
        if axis == 'roll':
            target = 0
        elif axis == 'pitch':
            target = 10
        else:  # yaw
            target = 0
        
        # Small drift over time
        drift = np.cumsum(np.random.normal(0, 0.001, self.n_samples))
        
        # Control corrections (periodic)
        corrections = 2 * np.sin(2 * np.pi * np.arange(self.n_samples) / (300 * self.sampling_rate_hz))
        
        attitude = target + drift + corrections
        attitude += np.random.normal(0, noise_std, self.n_samples)
        
        # Wrap angles
        if axis in ['roll', 'yaw']:
            attitude = np.mod(attitude + 180, 360) - 180
        else:  # pitch
            attitude = np.clip(attitude, -90, 90)
        
        return attitude
    
    def generate_communication_snr(self, nominal: float = 25.0, noise_std: float = 2.0) -> np.ndarray:
        """
        Generate communication signal-to-noise ratio.
        Varies with orbital geometry.
        """
        orbital = self._orbital_pattern()
        
        # SNR varies with distance to ground station
        snr = nominal + 8 * orbital
        
        # Occasional dropouts when out of view
        ground_station_view = (orbital > -0.3).astype(float)
        snr *= ground_station_view
        
        snr += np.random.normal(0, noise_std, self.n_samples)
        snr = np.clip(snr, 0, 40)
        
        return snr
    
    def inject_anomaly(self,
                      parameter: str,
                      data: np.ndarray,
                      anomaly_type: str,
                      start_idx: int,
                      duration: int = None) -> np.ndarray:
        """
        Inject an anomaly into telemetry data.
        
        Args:
            parameter: Parameter name
            data: Original data array
            anomaly_type: Type of anomaly ('spike', 'drop', 'drift', 'oscillation', 'stuck')
            start_idx: Start index of anomaly
            duration: Duration in samples (default varies by type)
        
        Returns:
            Modified data with anomaly
        """
        if duration is None:
            duration = int(30 * self.sampling_rate_hz)  # 30 seconds default
        
        end_idx = min(start_idx + duration, len(data))
        
        if anomaly_type == 'spike':
            # Sudden spike
            magnitude = np.random.uniform(3, 5) * np.std(data)
            data[start_idx:end_idx] += magnitude
            
        elif anomaly_type == 'drop':
            # Sudden drop
            magnitude = np.random.uniform(3, 5) * np.std(data)
            data[start_idx:end_idx] -= magnitude
            
        elif anomaly_type == 'drift':
            # Gradual drift
            drift_slope = np.linspace(0, np.random.uniform(2, 4) * np.std(data), end_idx - start_idx)
            data[start_idx:end_idx] += drift_slope
            
        elif anomaly_type == 'oscillation':
            # High-frequency oscillation
            freq = np.random.uniform(0.5, 2.0)  # Hz
            amplitude = np.random.uniform(2, 4) * np.std(data)
            t = np.arange(end_idx - start_idx) / self.sampling_rate_hz
            oscillation = amplitude * np.sin(2 * np.pi * freq * t)
            data[start_idx:end_idx] += oscillation
            
        elif anomaly_type == 'stuck':
            # Sensor stuck at value
            stuck_value = data[start_idx]
            data[start_idx:end_idx] = stuck_value
        
        # Mark as anomaly
        self.anomaly_labels[start_idx:end_idx] = 1
        
        return data
    
    def generate_dataset(self, n_anomalies: int = 10) -> pd.DataFrame:
        """
        Generate complete telemetry dataset with anomalies.
        
        Args:
            n_anomalies: Number of anomalies to inject
        
        Returns:
            DataFrame with telemetry time-series
        """
        # Generate normal telemetry
        data = {
            'timestamp': self.timestamps,
            'battery_voltage': self.generate_battery_voltage(),
            'solar_current': self.generate_solar_current(),
            'temp_cpu': self.generate_temperature('cpu', nominal=25),
            'temp_battery': self.generate_temperature('battery', nominal=15),
            'attitude_roll': self.generate_attitude('roll'),
            'attitude_pitch': self.generate_attitude('pitch'),
            'attitude_yaw': self.generate_attitude('yaw'),
            'comm_snr': self.generate_communication_snr()
        }
        
        df = pd.DataFrame(data)
        
        # Inject anomalies
        parameters = ['battery_voltage', 'solar_current', 'temp_cpu', 'temp_battery', 
                     'attitude_roll', 'attitude_pitch', 'comm_snr']
        anomaly_types = ['spike', 'drop', 'drift', 'oscillation', 'stuck']
        
        anomaly_log = []
        
        for i in range(n_anomalies):
            param = np.random.choice(parameters)
            anom_type = np.random.choice(anomaly_types)
            
            # Random start time (avoid first and last 10% of data)
            start_idx = np.random.randint(int(0.1 * self.n_samples), int(0.9 * self.n_samples))
            
            # Duration varies by type
            if anom_type in ['spike', 'drop']:
                duration = int(np.random.uniform(5, 30) * self.sampling_rate_hz)  # 5-30 seconds
            elif anom_type == 'stuck':
                duration = int(np.random.uniform(60, 300) * self.sampling_rate_hz)  # 1-5 minutes
            else:
                duration = int(np.random.uniform(30, 180) * self.sampling_rate_hz)  # 30s-3min
            
            # Inject anomaly
            df[param] = self.inject_anomaly(param, df[param].values, anom_type, start_idx, duration)
            
            anomaly_log.append({
                'anomaly_id': i,
                'parameter': param,
                'type': anom_type,
                'start_time': df.loc[start_idx, 'timestamp'],
                'duration_seconds': duration / self.sampling_rate_hz,
                'start_idx': start_idx,
                'end_idx': start_idx + duration
            })
        
        # Add anomaly label column
        df['anomaly'] = self.anomaly_labels
        
        # Save anomaly log
        self.anomaly_log = pd.DataFrame(anomaly_log)
        
        return df


# Example usage
if __name__ == "__main__":
    # Generate 24 hours of telemetry at 1 Hz sampling
    generator = SatelliteTelemetryGenerator(
        start_time=datetime(2026, 1, 12, 0, 0, 0),
        duration_hours=24,
        sampling_rate_hz=1.0,
        random_seed=42
    )
    
    # Generate dataset with 15 anomalies
    df = generator.generate_dataset(n_anomalies=15)
    
    # Display info
    print("=" * 70)
    print("SATELLITE TELEMETRY DATASET GENERATED")
    print("=" * 70)
    print(f"\nTotal samples: {len(df):,}")
    print(f"Duration: {generator.duration_hours} hours")
    print(f"Sampling rate: {generator.sampling_rate_hz} Hz")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print(f"\nParameters: {list(df.columns[1:-1])}")
    print(f"\nAnomaly statistics:")
    print(f"  Total anomalies injected: {len(generator.anomaly_log)}")
    print(f"  Anomalous samples: {df['anomaly'].sum():,} ({100*df['anomaly'].sum()/len(df):.2f}%)")
    print(f"  Normal samples: {(1-df['anomaly']).sum():,} ({100*(1-df['anomaly']).sum()/len(df):.2f}%)")
    
    print("\nAnomaly breakdown:")
    print(generator.anomaly_log[['parameter', 'type', 'start_time', 'duration_seconds']].to_string(index=False))
    
    print("\nDataset preview:")
    print(df.head(10))
    
    print("\nStatistical summary:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv('../../data/raw/telemetry_24h.csv', index=False)
    generator.anomaly_log.to_csv('../../data/raw/anomaly_log.csv', index=False)
    print("\n✅ Data saved to data/raw/")