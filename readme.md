# MODERN DISEASE PREDICTOR WITH CHATGPT

## Overview
The Machine Learning Disease Prediction Web App is a comprehensive health platform developed using React and Vite for the front end, and Flask for the backend. The primary objective is to provide users with an intelligent and user-friendly interface for predicting potential diseases based on input symptoms.

[View Website](https://moderndiseasedetector.onrender.com/)


## Key Features

 **Technology Stack:**
- **Frontend:** Developed using React and Vite, ensuring a responsive and modern user interface.
- **Backend:** Powered by Flask, a lightweight yet robust web framework for Python.

**Machine Learning Integration:**
- The heart of the application lies in its machine learning algorithms. By leveraging p

**ChatGPT Integration:**
- To enhance user interaction and understanding, the app integrates ChatGPT. Users can engage in a chat-like conversation to learn more about the predicted disease and gain a brief overview of its characteristics.


## How It Works:

**User Input:**
   - Users provide their symptoms through an intuitive interface.

**Machine Learning Prediction:**
   - The app processes the symptoms using machine learning algorithms, predicting potential diseases based on historical data and patterns.

**ChatGPT Interaction:**
   - For a more insightful experience, users can engage in a conversation with ChatGPT to gather additional information and insights about the predicted disease.


## Benefits:

**User-Friendly Interface:**
   - The web app offers a seamless and user-friendly experience, ensuring accessibility for individuals with varying technical expertise.

**Educational Interaction:**
   - By integrating ChatGPT, the app not only predicts diseases but also educates users about them, fostering a better understanding of their health.

**Responsive Design:**
   - The use of React and Vite ensures a responsive design, making the app accessible across various devices and screen sizes.

## Methodology

**Data Collection:**
   - Gather a diverse dataset containing symptoms and corresponding disease labels.
   - Ensure data quality and annotation accuracy.

**Data Preprocessing:**
   - Handle missing data and outliers.
   - Remove duplicate rows.

**Feature Selection/Dimensionality Reduction:**
   - Use techniques like feature selection or dimensionality reduction (e.g., Principal Component Analysis) to manage the high-dimensional symptom data.

**Model Selection:**
   - Choose appropriate machine learning models for disease detection.
   - Options may include Classical Machine Learning, Random Forest, Support Vector Machine, Gradient Boosting, etc.

**Splitting Dataset:**
   - Split the dataset into training and test datasets.

**Model Training:**
   - Train different models on the training dataset.

**Model Evaluation:**
   - Evaluate different model performance using cross-validation score and other metrics.
   - Choose the best-performing model.

**Hyperparameter Tuning:**
   - Use grid search cv and random search cv to tune the hyperparameters of the model and improve its performance.

**Model Testing:**
   - Use the test dataset to evaluate the final model's generalization performance.

**Deployment and Interface:**
   - Deploy the trained model in a production environment, considering scalability and performance.

## Architecture Diagram
![Architecture](./images/image4.png)

## Result
![Result](./images/image5.png)
   - It is evident from the table that SVC is performing best when it comes to accuracy but
     we still selected Bagging(SVC) because Bagging is an ensemble technique and offers
     less variance in comparison to SIngle SVC model and makes it more robust.

## In Conclusion:

The Machine Learning Disease Prediction Web App is not just a diagnostic tool; it's a holistic health companion that empowers users with insights, educational content, and a user-friendly interface for better health management.

## Images
![HOMEPAGE](./images/image1.png)
![SEARCH](./images/image2.png)
![RESULT](./images/image3.png)