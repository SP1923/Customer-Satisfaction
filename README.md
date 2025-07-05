
Customer Satisfaction Prediction

A machine learning project to predict customer satisfaction ratings from customer support ticket data. The goal is to use ticket features (like product, issue type, etc.) to classify satisfaction scores on a scale (e.g. 1–5). The notebook loads a cleaned dataset of support tickets, engineers features, and trains classifiers to predict the Customer Satisfaction Rating.

Dataset Overview

Source: customer_support_tickets_cleaned.csv (509 records, 17 original columns)

Features: Customer demographics and ticket details, including age, gender, product purchased, ticket type, subject, description, status, priority, channel, etc. The target variable is Customer Satisfaction Rating (ratings 1.0–5.0).

Task: Multi-class classification (predict satisfaction rating). Many target values are missing (NaN) and were presumably filtered or ignored during modeling.


Features Used


All relevant columns were encoded and used as input features (after dropping identifiers and unused columns). Specifically:

Numeric features: Ticket ID (identifier), Customer Age, Days Since First Purchase (newly computed from the purchase date)

Categorical features (label-encoded): Customer Gender, Product Purchased, Ticket Type, Ticket Subject, Ticket Description (encoded text categories), Ticket Status, Ticket Priority, Ticket Channel
(Note: The original text fields and timing fields like “Resolution”, “First Response Time”, and “Time to Resolution” were dropped or converted. For example, “Days Since First Purchase” was derived from the purchase date and included; a feature “Time to Resolution_seconds” was computed but largely NaN.)


Data Preprocessing


Feature Engineering: Computed new features such as Days Since First Purchase (days from earliest purchase date) and Time to Resolution (seconds) from existing fields.

Dropping irrelevant columns: Removed customer name/email, raw date/time fields, resolution text, and other fields not used as predictors (including the target when preparing features).

Encoding: Converted all categorical/text features into numeric form using label encoding. This includes gender, product, ticket type, subject, description, status, priority, and channel.

Train/Test Split: Data was split 60% train, 20% validation, and 20% test via successive train_test_split calls (80/20 then splitting 25% of the train).

Scaling: Feature values were standardized (zero mean, unit variance) using StandardScaler on the training set and applied to test data.


Exploratory Data Analysis (EDA)


The notebook performs various data visualizations to understand feature distributions and relationships:

Distribution of Satisfaction Ratings: A histogram (with KDE) shows how customer satisfaction scores are distributed across the dataset.

Product and Ticket Type Distribution: Bar charts and pie charts illustrate the most common products purchased and types of tickets. For example, a bar plot of the top products and a pie chart of ticket types (e.g. billing, technical, refund).

Ticket Channel and Priority: A bar plot of ticket channels (e.g. Chat, Email, Social media, Phone) and ticket priority levels (Low/Medium/High/Critical) visualizes how tickets are distributed.

Customer Demographics: The analysis includes plots like average satisfaction by customer gender (bar chart), and separate distributions of products for different genders.

Pairplot/Correlations: A Seaborn pairplot was generated on the dataset to inspect pairwise relationships between numerical features (though results are largely not discussed).

Overall, these EDA steps give insight into class imbalances and feature behavior but do not show strong obvious correlations with satisfaction rating.


Machine Learning Model


Model Used: Random Forest Classifier was the primary model built for predicting the satisfaction rating. (No other classifiers were implemented in this notebook.)

The Random Forest was instantiated with random_state=0 initially and later re-instantiated with tuned parameters from grid search.


Hyperparameter Tuning


Grid Search: GridSearchCV was used to find optimal Random Forest hyperparameters, optimizing for the F1-score.

Parameter grid: Included variations of

n_estimators: [50, 100] (number of trees)

max_depth: [10, 50] (tree depth)

min_samples_leaf: [0.5, 1]

min_samples_split: [0.001, 0.01]

max_features: ["sqrt"]

max_samples: [0.5, 0.9] (subset of samples per tree)

Results: The best parameters found were: max_depth=10, max_features='sqrt', max_samples=0.5, min_samples_leaf=0.5, min_samples_split=0.001, n_estimators=50. A final Random Forest was trained using these settings.


Evaluation Metrics & Results


Evaluation on Test Set: The optimized Random Forest was evaluated on the held-out test set.

The notebook computes classification metrics (accuracy, precision, recall, F1) on test predictions. In practice, the notebook prints the weighted precision (∼0.087) and issues warnings about undefined metrics for classes with no predictions. This very low precision (and by implication low F1 and accuracy) indicates poor performance, likely due to class imbalance or insufficient signal.
Note: The notebook only shows the precision score output (0.087). No confusion matrix or full classification report is provided, but such evaluation metrics would typically be reported here. In a complete report one would also include the overall accuracy and per-class F1-scores.


Visualizations (Generated)


The following plots were created during analysis (refer to the notebook for images):

Customer Satisfaction Rating Histogram: Distribution of the target rating.

Product Purchased Bar Chart: Top products by count (horizontal bar chart).

Ticket Type Pie Chart: Share of each ticket type (billing, technical issue, refund, etc.).

Satisfaction by Gender Bar Chart: Average satisfaction rating for Male, Female, Other customers.

Ticket Channel Bar Plot: Counts of tickets by communication channel (Chat, Email, Phone, Social media).
(No feature importance or confusion matrix was generated in the notebook.)


Suggestions for Future Work


Address Class Imbalance: Many ratings may be missing or imbalanced. Consider data resampling, synthetic data, or focusing on binary sentiment (satisfied vs. not).

Text Feature Processing: The notebook naively label-encodes ticket descriptions/subjects. Future work could apply NLP (TF-IDF, word embeddings) on text fields for richer features.

Model Experimentation: Try other algorithms (e.g. Gradient Boosting, SVM, neural networks) and ensemble techniques to improve accuracy.


Feature Engineering: Incorporate additional features (e.g. ticket resolution time, customer purchase history) or remove irrelevant ones (Ticket ID) to see impact.

Hyperparameter Optimization: Use more extensive search or Bayesian optimization; evaluate with cross-validation on all metrics (accuracy, recall, F1).

Validation of Results: Implement confusion matrices and more thorough reporting to identify where the model is making errors.
