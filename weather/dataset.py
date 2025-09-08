"""
Scripts to download raw weather data from Open Meteo API
"""
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_START_DATE = "1940-01-02"
DEFAULT_END_DATE = "2024-12-31"

class WeatherDataCollector:
    """
    Collects historical weather data from Open Meteo API for Sydney
    """
    
    def __init__(self, 
                 latitude: float = -33.8678, 
                 longitude: float = 151.2073,
                 timezone: str = "Australia/Sydney"):
        """
        Initialize the weather data collector
        
        Args:
            latitude: Latitude for Sydney
            longitude: Longitude for Sydney  
            timezone: Timezone for data
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Include all parameters that the API returns (including undocumented ones like mean values)
        self.daily_params = [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",  # API provides this even though not documented
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",  # API provides this even though not documented
            "sunrise",
            "sunset",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "sunshine_duration",
            "daylight_duration",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration"
        ]
        
    def fetch_weather_data(self, 
                          start_date: str, 
                          end_date: str) -> Dict:
        """
        Fetch weather data from Open Meteo API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing the API response
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(self.daily_params),
            "timezone": self.timezone
        }
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
            
    def download_all_data(self, 
                         start_date: str = DEFAULT_START_DATE,
                         end_date: str = DEFAULT_END_DATE) -> Dict:
        """
        Download all data in a single request
        
        Args:
            start_date: Starting date for data collection (YYYY-MM-DD format)
            end_date: Ending date for data collection (YYYY-MM-DD format)
            
        Returns:
            API response dictionary
        """
        
        # Fetch all data in one request
        data = self.fetch_weather_data(start_date, end_date)
        
        return data
    
    def process_api_response(self, data: Dict) -> pd.DataFrame:
        """
        Process API response into DataFrame
        
        Args:
            data: API response dictionary
            
        Returns:
            DataFrame with daily weather data
        """
        # Extract daily data
        daily_data = data.get('daily', {})
        
        if daily_data:
            daily_df = pd.DataFrame(daily_data)
            daily_df['time'] = pd.to_datetime(daily_df['time'])
            return daily_df.sort_values('time').reset_index(drop=True)
        else:
            return pd.DataFrame()
    
    def save_processed_data(self, 
                           daily_df: pd.DataFrame,
                           output_dir: Path) -> None:
        """
        Save processed dataframe to CSV file
        
        Args:
            daily_df: Daily weather data
            output_dir: Directory to save files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save daily data  
        daily_file = output_dir / "sydney_weather_daily.csv"
        daily_df.to_csv(daily_file, index=False)
        logger.info(f"Saved daily data to {daily_file}")
        
        # Save data info
        info_file = output_dir / "data_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Sydney Weather Data\n")
            f.write(f"==================\n\n")
            f.write(f"Location: Latitude {self.latitude}, Longitude {self.longitude}\n")
            f.write(f"Timezone: {self.timezone}\n")
            f.write(f"Date Range: {daily_df['time'].min()} to {daily_df['time'].max()}\n")
            f.write(f"Daily Data Shape: {daily_df.shape}\n")
            f.write(f"\nDaily Columns:\n{list(daily_df.columns)}\n")


def download_sydney_weather_data():
    """
    Main function to download Sydney weather data (daily only)
    """
    # Initialize collector
    collector = WeatherDataCollector()
    
    # Define paths
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all data in one request
    logger.info("Starting data download...")
    logger.info(f"Downloading daily weather data from {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    
    data = collector.download_all_data()
    
    # Process API response
    logger.info("Processing data...")
    daily_df = collector.process_api_response(data)
    
    # Save processed data
    logger.info("Saving processed data...")
    collector.save_processed_data(daily_df, raw_data_dir)
    
    logger.info("Data download complete!")
    
    return daily_df


if __name__ == "__main__":
    download_sydney_weather_data()