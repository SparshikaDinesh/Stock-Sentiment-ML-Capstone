# Stock-Sentiment-ML-Capstone
Machine learning model to predict stock prices using Twitter sentiment analysis





This project investigates the relationship between **social media sentiment** and **stock market movements**. By analyzing tweets alongside historical stock data for major tickers (like `$TSLA`, `$MSFT`, `$AAPL`), I developed a **Machine Learning pipeline** to predict future price trends.

---

## 🎯 Key Objectives
- Perform **NLP** on financial tweets to calculate daily sentiment polarity.
- Integrate **technical indicators** (SMA/EMA, RSI, Volatility) with sentiment data.
- Compare performance of **ML models** for binary price movement classification (Up/Down).

---

## 🛠️ Technologies Used
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, XGBoost, Matplotlib, Seaborn, TextBlob, VADER  
- **Methods:** Sentiment Analysis, TF-IDF Vectorization, Time-Series Forecasting (LSTM)  

---

## 📊 Models & Results
| Model       | Use Case                  | Note |
|------------|---------------------------|------|
| XGBoost    | Tabular classification     | Handles technical + sentiment features |
| LSTM (RNN) | Sequential time-series     | Captures long-term dependencies |
| Classical ML (k-NN, LogReg) | Baseline | ~51% accuracy for price direction |

---

## 📂 Repository Structure
