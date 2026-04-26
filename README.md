# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python and Machine Learning that suggests similar movies based on a user's selection.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 📖 Overview

This mini project implements a **Content-Based Movie Recommendation System** that recommends movies similar to a user-selected movie. The system analyzes movie metadata (genres, cast, crew, keywords, overview) and uses **cosine similarity** on vectorized features to find and rank the most similar movies.

---

## ✨ Features

- 🔍 Search for any movie by title
- 🎯 Get top N similar movie recommendations
- 📊 Uses TF-IDF / CountVectorizer for feature extraction
- 📐 Cosine Similarity for measuring movie likeness
- 🖥️ Simple and interactive user interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Vectorization & similarity computation |
| NLTK | Text preprocessing / stemming |
| Streamlit / Flask | Web application interface |
| Jupyter Notebook | Exploratory data analysis & model building |

---

## 📂 Dataset

The project uses the **TMDB 5000 Movie Dataset** available on [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), which includes:

- `tmdb_5000_movies.csv` — Movie metadata (title, genres, keywords, overview, etc.)
- `tmdb_5000_credits.csv` — Cast and crew information

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Merge movies and credits datasets on movie title
   - Extract relevant features: genres, keywords, cast, crew (director), and overview
   - Convert list-based JSON columns to plain text tags

2. **Feature Engineering**
   - Combine all features into a single `tags` column
   - Apply stemming to normalize words
   - Vectorize the tags using `CountVectorizer` (Bag of Words)

3. **Similarity Computation**
   - Compute **Cosine Similarity** between all movie vectors
   - Store the similarity matrix for fast lookup

4. **Recommendation**
   - Given an input movie, retrieve its index
   - Sort all movies by cosine similarity score
   - Return the top N most similar movies

---

## 🗂️ Project Structure

```
mini_project_movie_recommendation_system/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── model/
│   ├── movie_list.pkl          # Processed movie data
│   └── similarity.pkl          # Precomputed similarity matrix
│
├── app.py                      # Main application (Streamlit/Flask)
├── movie_recommendation.ipynb  # Jupyter Notebook (EDA + model building)
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/GannamaneniMeghana/mini_project_moive_recommendation_system.git
   cd mini_project_moive_recommendation_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
   - Place both files inside the `data/` folder

4. **Run the Jupyter Notebook** *(to generate model files)*
   ```bash
   jupyter notebook movie_recommendation.ipynb
   ```
   Run all cells to generate `movie_list.pkl` and `similarity.pkl` inside the `model/` directory.

5. **Launch the app**
   ```bash
   streamlit run app.py
   ```
   Then open your browser at `http://localhost:8501`

---

## 🎮 Usage

1. Open the web app in your browser
2. Select or type a movie name from the dropdown
3. Click **"Recommend"**
4. View the list of top recommended movies similar to your selection

---

## 🔮 Future Enhancements

- [ ] Add **Collaborative Filtering** for personalized recommendations
- [ ] Integrate **TMDB API** to fetch real-time movie posters and ratings
- [ ] Implement a **Hybrid Recommendation System** (content + collaborative)
- [ ] Add user login and watch history tracking
- [ ] Deploy on **Heroku** / **Streamlit Cloud**

---

## 👩‍💻 Author

**Gannamaneni Meghana**

- GitHub: [@GannamaneniMeghana](https://github.com/GannamaneniMeghana)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
