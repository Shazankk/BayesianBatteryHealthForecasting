# Install tensorflow for Mac M-Series

1. Clean up the environment:

```shell
# Deactivate and remove existing environment
deactivate
rm -rf .venv

# Create fresh environment
python3.10 -m venv .venv
source .venv/bin/activate
```

2. Install packages in the correct order:

```shell
# Update pip
pip install --upgrade pip setuptools wheel

# Install TensorFlow packages with specific versions
pip install tensorflow-macos==2.16.2
pip install tensorflow-metal==1.1.0

# Install TF-Keras and TensorFlow Probability
pip install tf-keras==2.16.0
pip install "tensorflow-probability[tf]==0.24.0"
```

3. Update your test notebook:

```python
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras

# Print version information
print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorFlow Probability Version: {tfp.__version__}")
print(f"TF-Keras Version: {tf_keras.__version__}")

# Basic TF operation to verify setup
x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(f"\nTensorFlow operation test:\n{x}")

# Test TFP distribution
dist = tfp.distributions.Normal(loc=0., scale=1.)
sample = dist.sample()
print(f"\nTFP distribution test:\n{sample}")

# Verify GPU availability
print(f"\nMetal device: {tf.config.list_physical_devices('GPU')}")
```

## Import data from huggingface/other sources

1. Import statement

```shell
pip install pandas numpy scipy scikit-learn xgboost
pip install pickle5
pip install seaborn matplotlib
pip install memory-profiler
# For hugging face
pip install datasets
pip install transformers
pip install kagglehub

pip install optuna
pip install shap
pip install imblearn
pip install python-dotenv
```

Alternatively, you can install all dependencies from the requirements.txt file:

```shell
pip install -r requirements.txt
```

## Project Overview

This project implements Bayesian statistical methods for battery health prediction and remaining useful life (RUL) forecasting. By leveraging probabilistic models, we can quantify the uncertainty in predictions, which is critical for reliability engineering and maintenance scheduling.
ments over time

## Datasetta

The model is trained on time-series battery degradation data, which includes:- Internal resistance measurements

- Charge/discharge cycles
- Capacity measurements over time## Methodology
- Temperature data
- Voltage curves### Bayesian Framework
- Internal resistance measurements
  Our approach employs a Bayesian hierarchical model to capture both the natural variability in battery degradation and the uncertainty in our predictions. The framework consists of:

## Methodology

degradation mechanisms

### Bayesian Framework

3. **Posterior Distribution**: Updated beliefs about battery health parameters
   Our approach employs a Bayesian hierarchical model to capture both the natural variability in battery degradation and the uncertainty in our predictions. The framework consists of:

### Key Models Implemented

1. **Prior Distribution**: Encoding domain knowledge about battery degradation mechanisms
2. **Likelihood Function**: Based on observed degradation patternsrediction
3. **Posterior Distribution**: Updated beliefs about battery health parameters

### Key Models Implemented- Monte Carlo methods for sampling from posterior distributions

- Bayesian Linear Regression for capacity fade prediction## Installation and Setup
- Hierarchical models for fleet-level inferences
- Gaussian Process models for uncertainty quantification
- Monte Carlo methods for sampling from posterior distributions
  Prediction_And_Lifetime_Forecasting.git

## Installation and Setupcd Bayesian_Battery_Health_Prediction_And_Lifetime_Forecasting

````bashvironment (optional but recommended)
# Clone the repository
git clone https://github.com/username/Bayesian_Battery_Health_Prediction_And_Lifetime_Forecasting.gitsource venv/bin/activate  # On Windows use: venv\Scripts\activate
cd Bayesian_Battery_Health_Prediction_And_Lifetime_Forecasting

# Create virtual environment (optional but recommended) install -r requirements.txt
python -m venv venv```
source venv/bin/activate  # On Windows use: venv\Scripts\activate
## Usage
# Install dependencies
pip install -r requirements.txt
``` and making predictions
from battery_health import BayesianHealthModel
## Usage

```pythonmodel = BayesianHealthModel.load('models/trained_model.pkl')
# Example code for loading a pre-trained model and making predictions
from battery_health import BayesianHealthModel
diction, confidence_interval = model.predict(new_battery_data)
# Load model```
model = BayesianHealthModel.load('models/trained_model.pkl')
## Results
# Make predictions with uncertainty estimates
prediction, confidence_interval = model.predict(new_battery_data)Our Bayesian approach demonstrates several advantages over deterministic models:
````

## Results Available

Our Bayesian approach demonstrates several advantages over deterministic models:4. **Interpretability**: Offers insights into the degradation physics

1. **Uncertainty Quantification**: Provides confidence intervals for all predictions### Performance Metrics
2. **Adaptability**: Model updates beliefs as new data becomes available
3. **Robustness**: Performs well even with limited training datatational Cost |
4. **Interpretability**: Offers insights into the degradation physics--- | ------------------ |
   Low |

### Performance Metrics | Medium |

| Gaussian Process | 2.5% | Well-calibrated | High |
| Model | RMSE | Uncertainty Calibration | Computational Cost |
|-------|------|------------------------|-------------------|## Conclusions and Future Work
| Bayesian Linear | 3.2% | Well-calibrated | Low |
| Hierarchical Bayes | 2.8% | Well-calibrated | Medium |The Bayesian framework successfully captures the uncertainty in battery health prediction, providing more reliable forecasts for battery end-of-life. Future work will focus on:
| Gaussian Process | 2.5% | Well-calibrated | High |

## Kaggle dataset and notebook

Kaggle dataset and notebook link can be accessed from:

https://www.kaggle.com/code/shagodg/ev-car-charging-eda-and-questionaire

## Conclusions and Future Worke updates

The Bayesian framework successfully captures the uncertainty in battery health prediction, providing more reliable forecasts for battery end-of-life. Future work will focus on:4. Optimizing computational efficiency for edge deployment

1. Incorporating more complex degradation mechanisms## References
2. Developing online learning capabilities for real-time updates
3. Extending the model to different battery chemistriesModeling
4. Optimizing computational efficiency for edge deployment

- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning

- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective- Plett, G. L. (2015). Battery Management Systems, Volume I: Battery Modeling## References
