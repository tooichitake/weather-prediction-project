# 🌦️ Sydney Weather Prediction Project

> *Can we predict Sydney's rain before it falls? How much precipitation will drench the city in the coming days?*

This project tackles the challenge of weather prediction using machine learning, focusing specifically on Sydney, Australia's largest city. By leveraging historical weather patterns, we build models that help answer two critical questions about Sydney's future weather.

## 🎯 Project Goals

### 🌧️ Rain Prediction Model
**Question**: Will it rain in Sydney exactly 7 days from now?
- **Type**: Binary Classification
- **Output**: Yes/No prediction with confidence score
- **Use Case**: Perfect for planning outdoor events, construction schedules, or weekend activities

### 💧 Precipitation Volume Model  
**Question**: How much rain will Sydney receive over the next 3 days?
- **Type**: Regression
- **Output**: Total precipitation in millimeters
- **Use Case**: Essential for water management, agriculture planning, and flood preparedness

## 📊 The Data Story

Our models learn from Sydney's rich weather history, drawing insights from years of meteorological data:

- **Location**: Sydney CBD (Latitude: -33.8678°, Longitude: 151.2073°)
- **Data Source**: Open Meteo Historical Weather API
- **Training Period**: All available data up to December 31, 2024
- **Test Period**: 2025 onwards (kept separate to ensure fair model evaluation)

### Key Weather Variables
- Temperature (min, max, mean)
- Precipitation amounts
- Wind speed and direction
- Humidity levels
- Atmospheric pressure
- Solar radiation
- Cloud cover

## 🏗️ Project Architecture

```
at2/
├── 📁 data/                     # The heart of our data pipeline
│   ├── raw/                     # Fresh from the API
│   ├── processed/               # Clean and ready for modeling
│   └── interim/                 # Work in progress
│
├── 🤖 models/                   # Our trained prediction engines
│   ├── rain_or_not/            # Binary rain predictors
│   └── precipitation_fall/      # Volume estimators
│
├── 📓 notebooks/                # Experimentation playground
│   ├── rain_or_not/            # Rain prediction experiments
│   └── precipitation_fall/      # Volume prediction trials
│
├── 📦 weather/                  # The brain - our custom Python package
│   ├── dataset.py              # Data collection wizardry
│   ├── features.py             # Feature engineering magic
│   ├── modeling/               # Model training & prediction
│   ├── plots.py                # Beautiful visualizations
│   └── config.py               # Central configuration
│
└── 📊 reports/                  # Insights and discoveries
    └── figures/                 # Visual storytelling
```

## 🔬 The Weather Package Deep Dive

### `dataset.py` - Data Collection Module
The gateway to weather data, featuring:
- **WeatherDataCollector**: Seamlessly fetches historical weather data
- Automatic data validation and quality checks
- Smart target variable generation for both models
- Efficient caching to minimize API calls

### `features.py` - Feature Engineering Laboratory
Where raw weather data transforms into predictive signals:
- **Temporal Features**: Capture seasonal patterns and cycles
- **Lag Features**: Learn from recent weather history (1, 3, 7, 14, 30 days)
- **Rolling Statistics**: Moving averages and variability measures
- **Weather Interactions**: Combine multiple variables for deeper insights

### `modeling/` - The Prediction Factory
- **train.py**: 
  - Automated hyperparameter tuning with GridSearchCV
  - Time-series aware cross-validation
  - Model comparison and selection
  - Performance metrics tracking
  
- **predict.py**:
  - Efficient batch predictions
  - Uncertainty quantification
  - Model versioning and loading
  - Real-time prediction capabilities

### `plots.py` - Visual Intelligence
Transform numbers into insights:
- Time series decomposition plots
- Feature importance rankings
- Model performance dashboards
- Prediction confidence intervals
- Weather pattern visualizations

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd at2

# Install dependencies using pip
pip install -r requirements.txt

# Or using Poetry (recommended)
poetry install
```

### Quick Example: Fetch Weather Data
```python
from weather.dataset import WeatherDataCollector

# Initialize collector
collector = WeatherDataCollector()

# Fetch Sydney's weather history
data = collector.fetch_historical_data(
    lat=-33.8678,
    lon=151.2073,
    start_date="2020-01-01",
    end_date="2024-12-31"
)

print(f"Collected {len(data)} days of weather data!")
```

### Train a Rain Prediction Model
```python
from weather.modeling.train import train_rain_model
from weather.features import engineer_features

# Prepare features
X, y = engineer_features(data, target='rain_in_7_days')

# Train model with automatic hyperparameter tuning
model, metrics = train_rain_model(X, y)

print(f"Model accuracy: {metrics['accuracy']:.2%}")
```

## 📈 Model Performance

Our models are evaluated using rigorous time-series cross-validation to ensure they generalize well to future predictions:

- **Rain Prediction**: Evaluated using accuracy, precision, recall, and F1-score
- **Volume Prediction**: Assessed with RMSE, MAE, and R² score

All models respect temporal ordering - we never train on future data to predict the past!

## 🔮 Future Enhancements

- Ensemble methods combining multiple weather models
- Integration of satellite imagery data
- Real-time prediction updates
- Extended forecast horizons
- Climate change impact analysis

## 📝 License

This project is part of an academic assessment for Advanced Machine Learning.