# NewsLens: News Category Classification Using a Neural Network

## Project Report

**Submitted by:** ____________________  
**Course / Organization:** ____________________  
**Submission date:** 10 September 2026  

---

## Abstract

NewsLens is a machine-learning application that automatically classifies news headlines and article excerpts into four categories: World, Sports, Business, and Sci/Tech. The project combines automated text preprocessing, TF-IDF feature extraction, a feed-forward neural network, and a professional web interface. A FastAPI service exposes the trained model through a prediction endpoint, while a React frontend provides an interactive neon-style classification dashboard. The project also includes a Streamlit interface and a Render deployment configuration.

The local demonstration dataset contains 32 balanced news articles, with eight examples in each category. The preprocessing pipeline cleans text, removes duplicates, validates labels, and produces a model-ready CSV file automatically.

## 1. Introduction

The rapid growth of digital news makes manual organization difficult. A news category classifier can help users organize articles, support content search, and create structured news feeds. This project develops an end-to-end text-classification system that accepts a headline or short article excerpt and predicts the most likely news category.

The four target categories are:

- **World:** international affairs, diplomacy, disasters, and global events
- **Sports:** competitions, athletes, teams, and tournaments
- **Business:** markets, companies, finance, trade, and economic activity
- **Sci/Tech:** science, software, devices, research, and technology

## 2. Objectives

1. Build an automated news text preprocessing pipeline.
2. Convert cleaned text into numerical TF-IDF features.
3. Train a multiclass neural-network classifier.
4. Display a predicted category and ranked confidence signals.
5. Provide a usable web interface for non-technical users.
6. Package the application for local execution and Render deployment.

## 3. System Architecture

The application has four main layers:

1. **Data layer:** AG News-style CSV data with `Class Index`, `Title`, and `Description` columns.
2. **ML layer:** text cleaning, TF-IDF vectorization, and an MLP neural network.
3. **API layer:** FastAPI endpoints for health checks and predictions.
4. **Interface layer:** React frontend with a neon dashboard and a Streamlit alternative.

### Request flow

```text
User enters headline
        |
        v
React frontend sends POST /api/predict
        |
        v
FastAPI cleans text and applies TF-IDF vectorizer
        |
        v
MLP neural network predicts four category probabilities
        |
        v
React displays category, confidence, and alternative signals
```

## 4. Dataset

The project uses an AG News-compatible format. The local training file is `data/train.csv` and contains 32 articles:

| Category | Number of articles |
|---|---:|
| World | 8 |
| Sports | 8 |
| Business | 8 |
| Sci/Tech | 8 |
| **Total** | **32** |

The dataset is intentionally balanced for the demonstration. A larger AG News dataset can be uploaded through the Streamlit application or substituted in the `data` directory for stronger generalization.

## 5. Automated Preprocessing

The preprocessing pipeline runs automatically whenever data is loaded. It performs the following steps:

- Combines `Title` and `Description` into one text field.
- Decodes HTML entities.
- Removes HTML tags.
- Converts text to lowercase.
- Removes accents and non-alphabetic characters.
- Collapses repeated whitespace.
- Removes missing and very short records.
- Removes duplicate articles.
- Validates the four supported labels.
- Converts labels to the internal indexes 0 through 3.
- Saves the result as `data/preprocessed_news.csv`.

This makes the training process repeatable and reduces manual preparation.

## 6. Feature Engineering

The cleaned articles are transformed using TF-IDF, or Term Frequency-Inverse Document Frequency. TF-IDF gives higher importance to terms that are frequent in one document but less common across the full collection. The implementation uses unigram and bigram features so that both individual words and short phrases contribute to classification.

The current local training run produces **958 TF-IDF features** from the 32-article dataset.

## 7. Neural Network Model

The classifier is a feed-forward multilayer perceptron implemented with scikit-learn's `MLPClassifier`. Its structure is:

- Input: TF-IDF feature vector
- Hidden layer 1: 64 neurons with ReLU-like activation behavior
- Hidden layer 2: 32 neurons
- Output: four category probabilities
- Random state: 42 for reproducibility
- Maximum iterations: 250

The model returns a probability distribution for all four categories. The application shows the highest-probability category as the prediction and presents the remaining categories as alternative signals.

## 8. Application Features

### React interface

The React interface provides:

- Neon control-room visual design
- Four-category navigation
- Model online status
- Headline and article input
- One-click example headlines
- Predicted category display
- Confidence progress bar
- Alternative category signals
- Training-data and feature-count metrics
- Responsive desktop and mobile layout

### FastAPI service

The API provides:

- `GET /api/health` for service status and model statistics
- `POST /api/predict` for news classification
- Same-origin static hosting of the built React application in production

### Streamlit interface

The Streamlit version remains available for quick experimentation, CSV upload, automated preprocessing, and model inspection.

## 9. Example Prediction

Input headline:

> Scientists develop a battery that charges electric cars in five minutes

Expected category:

> **Sci/Tech**

The interface returns the predicted category, a model confidence value, and the other category probabilities. Confidence is model output, not a guarantee of factual correctness.

## 10. Deployment

The project includes `render.yaml` for deployment on Render. The deployment process:

1. Installs Python dependencies.
2. Installs React dependencies.
3. Builds the React frontend with Vite.
4. Starts FastAPI with Uvicorn on Render's assigned port.
5. Serves the React build through FastAPI.
6. Uses `/api/health` as the Render health check.

The included `start_newslens.bat` file starts the API and React development server locally and opens the application in a browser.

## 11. Limitations

- The local dataset is small and intended for demonstration.
- A small dataset can produce overconfident predictions and may not represent real-world news diversity.
- The classifier does not verify facts or detect misinformation.
- Headlines containing vocabulary not present in training data may be difficult to classify.
- A production system should use a much larger train/validation/test split and report accuracy, precision, recall, F1-score, and a confusion matrix.

## 12. Future Enhancements

1. Train on the complete AG News dataset.
2. Add a validation and test pipeline with formal evaluation metrics.
3. Compare the neural network with Logistic Regression, SVM, and transformer models.
4. Add confidence calibration and low-confidence warnings.
5. Store prediction history and allow users to export results.
6. Add authentication and rate limiting for public deployment.
7. Add automated tests for preprocessing, API responses, and label mapping.

## 13. Conclusion

NewsLens demonstrates a complete news-category classification workflow from raw CSV data to a deployed web application. Automated preprocessing improves repeatability, TF-IDF provides an effective text representation, and the neural network converts the representation into four category predictions. The React and FastAPI architecture makes the model accessible through a professional interface, while the Render configuration supports straightforward deployment.

The project provides a strong foundation for expanding from a small educational dataset to a larger, evaluated, and production-ready news classification service.

---

## References

1. AG News dataset format and four-category classification task.
2. Scikit-learn documentation for `TfidfVectorizer` and `MLPClassifier`.
3. FastAPI documentation for API application development.
4. React and Vite documentation for frontend development.
5. Render documentation for web-service deployment.
