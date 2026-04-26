# 🎬 Movie Recommendation & Review System

## 📌 Introduction

This project is a **Flask-based web application** that provides personalized movie recommendations, user authentication, sentiment analysis on reviews, and watchlist/favorites management. It combines **machine learning models** with a user-friendly interface to deliver an interactive movie discovery experience.

---

## 📂 Table of Contents

* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [Configuration](#-configuration)
* [Machine Learning Models](#-machine-learning-models)
* [Datasets](#-datasets)
* [API Integration](#-api-integration)
* [Troubleshooting](#-troubleshooting)
* [Contributors](#-contributors)
* [License](#-license)

---

## 🚀 Features

* 🔐 User authentication (Email + Google OAuth)
* 🎥 Movie recommendation system (content-based filtering)
* 💬 Sentiment analysis on user reviews
* ⭐ Favorites & Watchlist management
* 🖼️ Movie poster fetching & caching
* 📊 Multiple streaming platform datasets (Netflix, Amazon Prime, etc.)
* ⚡ Optimized performance with pre-trained models

---

## 🧰 Tech Stack

* **Backend:** Flask (Python)
* **Frontend:** HTML, CSS (templates assumed)
* **Machine Learning:** Scikit-learn
* **Data Handling:** Pandas
* **Authentication:** Flask-Login + OAuth (Google)
* **Other Tools:** dotenv, requests

---

## 📁 Project Structure

```
Mini_project_movie_/
│── app.py                     # Main Flask application
│── train_recommender.py       # Script to train recommendation model
│── train_sentiment.py         # Script to train sentiment model
│── fetch_posters.py           # Poster fetching utility
│── recommender_model.pkl      # Trained recommendation model
│── sentiment_model.pkl        # Trained sentiment model
│── vectorizer.pkl             # TF-IDF vectorizer
│── sentiment_vectorizer.pkl   # Sentiment vectorizer
│── movie.csv                  # Main movie dataset
│── users.csv                  # User data
│── reviews.csv                # User reviews
│── watchlist.csv              # Watchlist storage
│── favorites.csv              # Favorites storage
│── posters.csv                # Cached posters
│── requirements.txt           # Dependencies
│── Procfile                   # Deployment config
│── meg.env                    # Environment variables
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Mini_project_movie_
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Application

```bash
python app.py
```

Or (Windows):

```bash
run_server.bat
```

Then open:

```
http://127.0.0.1:5000/
```

---

## 🔧 Configuration

Create a `.env` file (or use `meg.env`) with:

```
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

Also update:

* `API_KEY` (movie poster API)
* `SECRET_KEY` (Flask security)

---

## 🤖 Machine Learning Models

### 1. Recommendation System

* Uses **TF-IDF Vectorization**
* Computes **cosine similarity** between movies
* Stored in:

  * `recommender_model.pkl`
  * `vectorizer.pkl`

### 2. Sentiment Analysis

* Classifies reviews as positive/negative
* Stored in:

  * `sentiment_model.pkl`
  * `sentiment_vectorizer.pkl`

---

## 📊 Datasets

The project uses multiple datasets:

* `movie.csv` (primary dataset)
* Streaming platform datasets:

  * Netflix
  * Amazon Prime
  * Disney+
  * Hulu
  * Peacock
  * Paramount+

---

## 🌐 API Integration

* Movie poster fetching via external API
* Cached locally in `posters.csv` for performance

---

## 🛠️ Troubleshooting

### Common Issues

* **Module not found**
  → Run `pip install -r requirements.txt`

* **OAuth not working**
  → Check `.env` credentials

* **CSV errors**
  → Run:

  ```bash
  python fix_csv.py
  ```

* **Model issues**
  → Retrain models:

  ```bash
  python train_recommender.py
  python train_sentiment.py
  ```

---

## 👥 Contributors

* Project developed as a **Mini Project (Movie Recommendation System)**
  *(Add your team members here)*

---

## 📄 License

This project is licensed under the MIT License (or specify your license).

---

## 💡 Future Improvements

* Deploy to cloud (Heroku/AWS)
* Add collaborative filtering
* Improve UI/UX
* Add real-time recommendations
* Integrate more APIs (IMDb, TMDb)

---

## 📬 Contact

For questions or contributions, feel free to reach out.

---

✨ *Enjoy discovering your next favorite movie!* 🎥
