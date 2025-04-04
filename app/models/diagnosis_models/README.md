# Diagnostic AI Models

This directory contains the trained machine learning models used for medical diagnosis.

## Model Types

The system uses multiple AI approaches for diagnosis:

1. **Neural Network Models** - Deep learning models for pattern recognition in symptom combinations
2. **Decision Tree Models** - Interpretable models for rule-based diagnosis
3. **Bayesian Network Models** - Probabilistic models for diagnosis under uncertainty

## Files

- `neural_net.pt` - PyTorch neural network model
- `decision_tree.pkl` - Scikit-learn decision tree classifier
- `vectorizer.pkl` - Pre-fitted MultiLabelBinarizer for symptom vectorization
- `label_encoder.pkl` - Mapping between model outputs and disease names

## Training

Models are automatically trained when the system first runs if pre-trained models are not available.
Training occurs in the following steps:

1. Data collection from the database (diseases and their associated symptoms)
2. Feature engineering (vectorization of symptoms)
3. Model training with appropriate hyperparameters
4. Model evaluation and validation
5. Saving the trained models to disk

## Usage

The AI service automatically loads these models when needed. The models are used to provide
diagnostic suggestions based on reported symptoms.

## Performance Considerations

- The neural network requires more computational resources but can capture complex patterns
- The decision tree is faster and more interpretable
- The Bayesian network handles uncertain information better

## Model Updates

Models are retrained periodically as new medical data is collected via the scraper service.
This ensures that the diagnostic capabilities improve over time.