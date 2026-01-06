# Stock-Sentiment-ML-Capstone
Machine learning model to predict stock prices using Twitter sentiment analysis

Stock-Sentiment-ML-Capstone
Machine learning model to predict stock prices using Twitter sentiment analysis


Stock-Sentiment-ML-Capstone

Machine Learning project to predict stock price movements using Twitter sentiment analysis.

This project investigates the relationship between **social media sentiment** and **stock market movements**. By analyzing over **80,000 tweets** alongside historical stock data for major tickers (like `$TSLA`, `$MSFT`, `$AAPL`), I developed a **Machine Learning pipeline** to predict future price trends. The goal is to combine **Natural Language Processing (NLP)** on financial tweets with **technical stock indicators** to forecast the direction of stock prices (Up/Down).

🎯 Key Objectives
- Perform **NLP on financial tweets** to calculate daily sentiment polarity using TextBlob and VADER.
- Integrate **technical indicators** such as Moving Averages (SMA/EMA), RSI, and Volatility with sentiment data.
- Build and compare **Machine Learning models** (XGBoost, LSTM, and classical ML methods like k-NN and Logistic Regression) for predicting binary price movement.
- Evaluate model performance using metrics like **accuracy, precision, recall**, and visualize results with **charts and plots**.

🛠️ Project Highlights
- Over **80,000 tweets** analyzed for sentiment trends.
- Integration of **time-series stock data** with daily sentiment scores.
- Multiple models implemented:
  - **XGBoost:** Handles tabular features combining technical and sentiment data.
  - **LSTM (RNN):** Captures sequential dependencies in stock prices.
  - **Classical ML (k-NN, Logistic Regression):** Provides baseline for comparison (~51% accuracy).
- Visualizations include sentiment trends versus stock price movements, model prediction plots, and confusion matrices.

 📂 What’s in the Project
- **Notebooks:** Full workflow from data exploration, preprocessing, and model training.
- **Scripts:** Python scripts to run models and preprocess data.
- **Sample Data:** CSVs for tweets and stock prices (small sample included for testing).
- **Saved Models:** Trained XGBoost and LSTM models ready for prediction.
- **Requirements:** Python libraries used include Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, Matplotlib, Seaborn, TextBlob, VADER.

🔮 Future Improvements
- Incorporate more social media platforms (Reddit, StockTwits) for richer sentiment data.
- Use **real-time streaming data** for live stock prediction.
- Deploy an **interactive dashboard** (Streamlit or Gradio) for users to explore predictions and sentiment trends in real-time.
- Experiment with advanced models like **Transformers or hybrid LSTM+XGBoost pipelines** for improved prediction accuracy.

