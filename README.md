# FRAUDSHIELD AI

FRAUDSHIELD AI is a comprehensive solution for detecting credit card fraud using advanced machine learning models. This project includes a streamlined pipeline for data preprocessing, model training, and deployment.

## Features

- **Dataset Overview**: Analyzes the class imbalance and key statistics of the dataset.
- **Exploratory Data Analysis (EDA)**: Provides insights into the data distribution and feature importance.
- **Model Training**: Compares multiple models, including Logistic Regression, Decision Tree, Random Forest, XGBoost, and Artificial Neural Networks (ANN).
- **Explainability**: Uses SHAP values to explain model predictions.
- **Deployment Ready**: Organized structure for easy deployment and containerization.

## Project Structure

- `app.py`: Streamlit app for fraud detection.
- `credit_card_fraud_detection.py`: Training pipeline for fraud detection models.
- `models/`: Directory containing trained models and scalers.
- `creditcard.csv`: Dataset used for training and evaluation.
- `requirements.txt`: Python dependencies for the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install dependencies using:

  ```bash
  pip install -r requirements.txt
  ```

### Running the Application

1. Train the models (if not already trained):

   ```bash
   python credit_card_fraud_detection.py
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

### Folder Organization

- All models are stored in the `models/` directory for better organization.
- The code is updated to ensure backward compatibility with previous versions.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.