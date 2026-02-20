from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import random
import re
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv('meg.env')
app.secret_key = "supersecretkey" # Needed for flash messages
# Allow OAuth to work over HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# OAuth Configuration
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), 
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

movies = pd.read_csv("movie.csv")
movies.rename(columns={
    'Movie Name': 'title',
    'Rating(10)': 'rating',
    'Genre': 'genre',
    'Language': 'language',
    'Description': 'description'
}, inplace=True)
df = pd.read_csv("movie.csv", encoding='utf-8', on_bad_lines='skip')

# Load Recommender system models
tfidf_matrix = pickle.load(open("recommender_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load Sentiment models
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
sentiment_vectorizer = pickle.load(open("sentiment_vectorizer.pkl", "rb"))

REVIEWS_FILE = "reviews.csv"
WATCHLIST_FILE = "watchlist.csv"
FAVORITES_FILE = "favorites.csv"
POSTER_CACHE_FILE = "posters.csv"
USERS_FILE = "users.csv"
SETTINGS_FILE = "user_settings.csv"
API_KEY = "8e6f23ad"

# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User Class
class User(UserMixin):
    def __init__(self, id, username, password_hash, security_question=None, security_answer=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.security_question = security_question
        self.security_answer = security_answer

def load_users():
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header
            
            # Simple check for schema version based on header length
            has_security = False
            if header and "security_question" in header:
                has_security = True
                
            for row in reader:
                if len(row) >= 3:
                    security_question = row[3] if has_security and len(row) > 3 else None
                    security_answer = row[4] if has_security and len(row) > 4 else None
                    users[row[0]] = User(row[0], row[1], row[2], security_question, security_answer)
    return users

def save_user(username, password, security_question=None, security_answer=None):
    users = load_users()
    for user in users.values():
        if user.username.lower() == username.lower():
            return False # User exists
    
    new_id = str(len(users) + 1)
    password_hash = generate_password_hash(password)
    
    # Store security answer hash for privacy
    security_answer_hash = generate_password_hash(security_answer) if security_answer else None
    
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) == 0:
            writer.writerow(["id", "username", "password_hash", "security_question", "security_answer"])
        
        writer.writerow([new_id, username, password_hash, security_question, security_answer_hash])
    return True

def update_user_password(user_id, new_password):
    users_data = []
    updated = False
    
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            users_data = list(reader)

    if not users_data:
        return False

    header = users_data[0]
    rows = users_data[1:]
    
    new_password_hash = generate_password_hash(new_password)
    
    with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            if row[0] == str(user_id):
                row[2] = new_password_hash
                updated = True
            writer.writerow(row)
            
    return updated

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    return users.get(user_id)

# Ensure files exist
if not os.path.exists(REVIEWS_FILE):
    with open(REVIEWS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movie_title", "review", "sentiment", "date"])

# Migration check
if os.path.exists("favorites.csv") and not os.path.exists(WATCHLIST_FILE):
    try:
        os.rename("favorites.csv", WATCHLIST_FILE)
        print("Migrated favorites.csv to watchlist.csv")
    except Exception as e:
        print(f"Error migrating favorites: {e}")

if not os.path.exists(WATCHLIST_FILE):
    with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_title"])

if not os.path.exists(FAVORITES_FILE):
    with open(FAVORITES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "movie_title"])

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "username", "password_hash", "security_question", "security_answer"])

# Auto-migrate USERS_FILE to include security columns if missing
if os.path.exists(USERS_FILE):
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            items = list(csv.reader(f))
            if items and len(items) > 0:
                header = items[0]
                if "security_question" not in header:
                    print("Migrating users.csv to include security questions...")
                    new_header = ["id", "username", "password_hash", "security_question", "security_answer"]
                    new_rows = []
                    for row in items[1:]:
                        # Pad with empty values
                        while len(row) < 3: row.append("")
                        new_rows.append([row[0], row[1], row[2], "", ""])
                    
                    with open(USERS_FILE, 'w', newline='', encoding='utf-8') as fw:
                        writer = csv.writer(fw)
                        writer.writerow(new_header)
                        writer.writerows(new_rows)
    except Exception as e:
        print(f"Migration error: {e}")

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "preferred_language", "favorite_genre", "watchlist_enabled", "favorites_enabled", "selected_platforms", "optimization_enabled", "notifications_enabled", "dark_mode"])

poster_cache = {}

# Load poster cache
if os.path.exists(POSTER_CACHE_FILE):
    try:
        with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    poster_cache[row[0]] = row[1]
    except Exception as e:
        print(f"Error loading poster cache: {e}")

def save_poster_to_cache(title, url):
    poster_cache[title] = url
    try:
        with open(POSTER_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([title, url])
    except Exception as e:
        print(f"Error saving to poster cache: {e}")

def load_user_settings(user_id):
    settings = {
        "preferred_language": "English",
        "favorite_genre": "Action",
        "watchlist_enabled": "true",
        "favorites_enabled": "true",
        "selected_platforms": "Netflix,Prime Video",
        "optimization_enabled": "true",
        "notifications_enabled": "true",
        "dark_mode": "false"
    }
    
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row[0] == str(user_id):
                    # Ensure row has enough columns
                    if len(row) >= 9:
                        settings = {
                            "preferred_language": row[1],
                            "favorite_genre": row[2],
                            "watchlist_enabled": row[3],
                            "favorites_enabled": row[4],
                            "selected_platforms": row[5],
                            "optimization_enabled": row[6],
                            "notifications_enabled": row[7],
                            "dark_mode": row[8]
                        }
                    break
    return settings

def save_user_settings(user_id, new_settings):
    rows = []
    updated = False
    
    # Load all existing
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
    header = rows[0] if rows else ["user_id", "preferred_language", "favorite_genre", "watchlist_enabled", "favorites_enabled", "selected_platforms", "optimization_enabled", "notifications_enabled", "dark_mode"]
    data = rows[1:] if rows else []
    
    # Update logic
    final_data = []
    for row in data:
        if row[0] == str(user_id):
            # Update this row
            final_data.append([
                str(user_id),
                new_settings.get("preferred_language", row[1]),
                new_settings.get("favorite_genre", row[2]),
                new_settings.get("watchlist_enabled", row[3]),
                new_settings.get("favorites_enabled", row[4]),
                new_settings.get("selected_platforms", row[5]),
                new_settings.get("optimization_enabled", row[6]),
                new_settings.get("notifications_enabled", row[7]),
                new_settings.get("dark_mode", row[8])
            ])
            updated = True
        else:
            final_data.append(row)
            
    if not updated:
        # Append new
        final_data.append([
            str(user_id),
            new_settings.get("preferred_language", "English"),
            new_settings.get("favorite_genre", "Action"),
            new_settings.get("watchlist_enabled", "true"),
            new_settings.get("favorites_enabled", "true"),
            new_settings.get("selected_platforms", "Netflix,Prime Video"),
            new_settings.get("optimization_enabled", "true"),
            new_settings.get("notifications_enabled", "true"),
            new_settings.get("dark_mode", "false")
        ])
        
    with open(SETTINGS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_data)

def get_poster(title):
    if title in poster_cache:
        url = poster_cache[title]
        return url if url != "N/A" else None

    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    try:
        r = requests.get(url).json()
        if r.get("Response") == "True":
            poster = r.get("Poster")
            if poster and poster != "N/A":
                save_poster_to_cache(title, poster)
                return poster
            else:
                save_poster_to_cache(title, "N/A")
                return None
        else:
            save_poster_to_cache(title, "N/A")
            return None
    except:
        return None

def get_trailer_id(title):
    try:
        # Prepare search query
        query = f"{title} official trailer"
        # Search on YouTube
        search_url = f"https://www.youtube.com/results?search_query={query}"
        response = requests.get(search_url)
        # Extract video IDs
        video_ids = re.findall(r"watch\?v=(\w{11})", response.text)
        if video_ids:
            return video_ids[0] # Return the first result
        return None
    except Exception as e:
        print(f"Error fetching trailer: {e}")
        return None

def fetch_posters_parallel(movies_df):
    results = {}
    uncached_indices = []
    
    for i, row in movies_df.iterrows():
        title = row['title']
        if title in poster_cache:
            url = poster_cache[title]
            results[i] = url if url != "N/A" else None
        else:
            uncached_indices.append(i)
    
    if uncached_indices:
        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = {executor.submit(get_poster, movies_df.loc[i, 'title']): i for i in uncached_indices}
            
            for future in tasks:
                idx = tasks[future]
                try:
                    poster = future.result()
                    results[idx] = poster
                except:
                    results[idx] = None
                    
    return results

def get_top_rated_movies():
    # Return ALL movies sorted by rating (as per user request "show all")
    candidates = movies.sort_values(by='rating', ascending=False).copy()
    poster_map = fetch_posters_parallel(candidates)
    candidates['poster'] = candidates.index.map(poster_map)
    valid_movies = candidates[candidates['poster'].notna()]
    return valid_movies.to_dict(orient="records")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        users = load_users()
        user = next((u for u in users.values() if u.username == username), None)
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        security_question = request.form.get("security_question")
        security_answer = request.form.get("security_answer")
        
        if save_user(username, password, security_question, security_answer):
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Username already exists", "error")
            
    return render_template("register.html")

@app.route("/social_login/<provider>")
def social_login(provider):
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    
    if provider.lower() == 'google':
        # Real Google Login
        redirect_uri = url_for('google_auth', _external=True)
        print(f"DEBUG: Sending redirect_uri: {redirect_uri}")
        return google.authorize_redirect(redirect_uri)
    
    return redirect(url_for("login"))

@app.route('/callback')
def google_auth():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            # Fallback for some versions or if userinfo not in token
            user_info = google.post('https://openidconnect.googleapis.com/v1/userinfo').json()
        
        email = user_info['email']
        name = user_info.get('name', email.split('@')[0])
        
        # Use email as username for uniqueness
        username = email
        dummy_password = "google_oauth_dummy_password" # Or random
        
        users = load_users()
        user = next((u for u in users.values() if u.username.lower() == username.lower()), None)
        
        if not user:
            save_user(username, dummy_password)
            users = load_users()
            user = next((u for u in users.values() if u.username.lower() == username.lower()), None)
            flash(f"Account created via Google for {name}!", "success")
        else:
            flash(f"Welcome back, {name}!", "success")
            
        login_user(user)
        return redirect(url_for('home'))
        
    except Exception as e:
        flash(f"Google Login Failed: {str(e)}. Did you add the Client Secret?", "error")
        return redirect(url_for('login'))



@app.route("/profile")
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    user_settings = load_user_settings(current_user.id)
    
    if request.method == "POST":
        # Check if it's a specific action or full update
        action = request.form.get("action")
        
        if action == "clear_watchlist":
            return redirect(url_for("clear_watchlist_action"))
        elif action == "clear_favorites":
            return redirect(url_for("clear_favorites_action"))
        elif action == "change_password":
            current_password = request.form.get("current_password")
            new_password = request.form.get("new_password")
            confirm_password = request.form.get("confirm_password")
            
            # Re-load current user to get latest generic password_hash check
            users = load_users()
            user = users.get(current_user.id)
            
            if not user or not check_password_hash(user.password_hash, current_password):
                flash("Incorrect current password", "error")
            elif new_password != confirm_password:
                flash("New passwords do not match", "error")
            elif len(new_password) < 6:
                flash("Password must be at least 6 characters", "error")
            else:
                if update_user_password(current_user.id, new_password):
                    flash("Password updated successfully!", "success")
                else:
                    flash("Failed to update password", "error")
            return redirect(url_for("settings"))
            
        # Update settings
        new_settings = {
            "preferred_language": request.form.get("preferred_language", user_settings["preferred_language"]),
            "favorite_genre": request.form.get("favorite_genre", user_settings["favorite_genre"]),
            "watchlist_enabled": "true" if request.form.get("watchlist_enabled") else "false",
            "favorites_enabled": "true" if request.form.get("favorites_enabled") else "false",
            "selected_platforms": ",".join(request.form.getlist("selected_platforms")),
            "optimization_enabled": "true" if request.form.get("optimization_enabled") else "false",
            "notifications_enabled": "true" if request.form.get("notifications_enabled") else "false",
            "dark_mode": "true" if request.form.get("dark_mode") else "false"
        }
        
        save_user_settings(current_user.id, new_settings)
        flash("Settings updated successfully!", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", settings=user_settings, page="settings")

@app.route("/settings/clear_watchlist")
@login_required
def clear_watchlist_action():
    if os.path.exists(WATCHLIST_FILE):
        rows = []
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        header = rows[0] if rows else ["user_id", "movie_title"]
        # Keep rows NOT belonging to current user
        data_rows = [r for r in rows[1:] if len(r) > 0 and r[0] != str(current_user.id)]
        
        with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
            
    flash("Watchlist cleared!", "success")
    return redirect(request.referrer or url_for("settings"))

@app.route("/settings/clear_favorites")
@login_required
def clear_favorites_action():
    if os.path.exists(FAVORITES_FILE):
        rows = []
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        header = rows[0] if rows else ["user_id", "movie_title"]
        # Keep rows NOT belonging to current user
        data_rows = [r for r in rows[1:] if len(r) > 0 and r[0] != str(current_user.id)]
        
        with open(FAVORITES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
            
    flash("Favorites cleared!", "success")
    return redirect(request.referrer or url_for("settings"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
def logo():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return render_template("logo.html")

@app.route("/index")
@app.route("/home")
@login_required 
def home():
    genre = request.args.get('genre')
    language = request.args.get('language')
    rating_filter = request.args.get('rating')

    filtered_movies = movies.copy()

    # Apply filters if present...
    if genre:
        filtered_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, case=False, na=False)]
    if language:
        filtered_movies = filtered_movies[filtered_movies['language'].str.contains(language, case=False, na=False)]
    if rating_filter:
        if rating_filter == '8+':
            filtered_movies = filtered_movies[filtered_movies['rating'] >= 8]
        elif rating_filter == '7-8':
            filtered_movies = filtered_movies[(filtered_movies['rating'] >= 7) & (filtered_movies['rating'] < 8)]
        elif rating_filter == '6-7':
            filtered_movies = filtered_movies[(filtered_movies['rating'] >= 6) & (filtered_movies['rating'] < 7)]
        elif rating_filter == '5-6':
            filtered_movies = filtered_movies[(filtered_movies['rating'] >= 5) & (filtered_movies['rating'] < 6)]

    # If filters are applied, show results sorted by rating
    if genre or language or rating_filter:
        filtered_movies = filtered_movies.sort_values(by='rating', ascending=False).head(50)
        final_movies = []
        for _, row in filtered_movies.iterrows():
            m = row.to_dict()
            m['poster'] = get_poster(m['title'])
            final_movies.append(m)
        return render_template('index.html', movies=final_movies, page='home')

    # Default Home: Show 10 RANDOM movies (Refreshable)
    else:
        # Sample 10 random movies
        random_indices = random.sample(range(len(movies)), min(10, len(movies)))
        random_selection = movies.iloc[random_indices]
        
        final_movies = []
        for _, row in random_selection.iterrows():
            m = row.to_dict()
            m['poster'] = get_poster(m['title'])
            final_movies.append(m)
            
        return render_template('index.html', movies=final_movies, page='home')


@app.route("/top_rated")
@login_required
def top_rated():
    top_movies = get_top_rated_movies()
    return render_template('index.html', movies=top_movies, page='top_rated')
    

@app.route("/attributes")
@login_required
def attributes():
    return render_template("attributes.html", page="attributes")

@app.route("/services")
@login_required
def services():
    return render_template("services.html", page="services")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return redirect(url_for("home"))
        
    query_lower = query.lower()
    
    # 1. Check if query is a Service Name
    matched_services = []
    
    # Helper to clean service names for comparison
    def clean_svc_name(n): return n.lower().replace("+", "plus").replace(" ", "")
    
    query_clean = clean_svc_name(query_lower)
    
    # Map cleaned names back to display names
    svc_map = {clean_svc_name(k): k for k in SUBSCRIPTION_PRICES.keys()}
    
    # Check exact match or "movies on X" pattern
    detected_service = None
    if query_clean in svc_map:
        detected_service = svc_map[query_clean]
    else:
        # Check "movies on [service]"
        for k_clean, k_real in svc_map.items():
            if k_clean in query_clean:
                detected_service = k_real
                break
                
    if detected_service:
        # Get all movies for this service
        # Using titles intersection
        titles_on_service = set()
        # Look up in SERVICE_TITLES
        # We need to match the keys in SERVICE_TITLES to detected_service
        # SERVICE_TITLES keys might be slightly different than SUBSCRIPTION_PRICES keys in some cases
        # Let's fuzzy match
        
        target_titles = set()
        for svc_key, titles in SERVICE_TITLES.items():
            if clean_svc_name(detected_service) in clean_svc_name(svc_key):
               target_titles.update(titles)

        # Filter movies.csv
        # This is strictly matching by title string, which can be brittle, but consistent with current design
        candidates = movies[movies['title'].astype(str).str.lower().str.strip().isin(target_titles)].copy()
        
        # Sort by rating to show best first
        candidates = candidates.sort_values(by='rating', ascending=False).head(50)
        
        poster_map = fetch_posters_parallel(candidates)
        candidates['poster'] = candidates.index.map(poster_map)
        valid_movies = candidates[candidates['poster'].notna()]
        results = valid_movies.to_dict(orient="records")
        
        flash(f"Showing movies available on {detected_service}", "info")
        return render_template("index.html", movies=results, search_query=query, page="search")

    # 2. Check if query is a Genre (Attribute)
    # Get all unique genres
    all_genres = set()
    for g_str in movies['genre'].dropna().unique():
        for g in g_str.split(','):
            all_genres.add(g.strip().lower())
            
    if query_lower in all_genres:
        # Filter by genre
        candidates = movies[movies['genre'].str.contains(query, case=False, na=False)].sort_values(by='rating', ascending=False).head(50)
        
        poster_map = fetch_posters_parallel(candidates)
        candidates['poster'] = candidates.index.map(poster_map)
        valid_movies = candidates[candidates['poster'].notna()]
        results = valid_movies.to_dict(orient="records")
        
        flash(f"Showing {query.title()} movies", "info")
        return render_template("index.html", movies=results, search_query=query, page="search")

    # 3. Fallback: Vector / Content Search
    # Check for Exact Title Match first
    exact_match_movie = None
    # We use a case-insensitive match against the DataFrame
    # Note: query_lower is already defined above
    exact_match_row = movies[movies['title'].astype(str).str.lower().str.strip() == query_lower]
    
    if not exact_match_row.empty:
        # Direct Redirect for Exact Match
        found_title = exact_match_row.iloc[0]['title']
        return redirect(url_for('movie_details', title=found_title))

    # Vector Search for recommendations
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    idx = similarity.argsort()[-50:][::-1]
    candidates = movies.iloc[idx].copy()
    
    # If we have an exact match, exclude it from candidates to avoid duplication
    if exact_match_movie:
        candidates = candidates[candidates['title'].str.lower().str.strip() != query_lower]

    poster_map = fetch_posters_parallel(candidates)
    candidates['poster'] = candidates.index.map(poster_map)
    valid_movies = candidates[candidates['poster'].notna()]
    
    # Convert to list of dicts
    rec_results = valid_movies.head(50).to_dict(orient="records")
    
    # Combine: Exact Match + Recommendations
    final_results = []
    if exact_match_movie:
        # Ensure exact match has a poster (or placeholder) before adding
        if not exact_match_movie.get('poster'):
             exact_match_movie['poster'] = "https://via.placeholder.com/300x450?text=No+Poster"
        final_results.append(exact_match_movie)
        
    final_results.extend(rec_results)
    
    return render_template("index.html", movies=final_results, search_query=query, page="search")

# Service Data Cache
SERVICE_TITLES = {}

def preload_services():
    global SERVICE_TITLES
    service_files = {
        "Netflix": "netflix_titles.csv",
        "Hulu": "hulu_titles.csv",
        "Disney+": "disney_plus_titles.csv",
        "Prime Video": "amazon_prime_titles.csv",
        "Peacock Premium": "peacock.csv",
        "Apple TV+": "appletv+.csv.csv",
        "Max": "max.csv.csv",
        "Paramount+": "paramount+.csv.csv"
    }
    
    print("Preloading service data...")
    for service, filename in service_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Normalize columns
                df.columns = [c.lower().strip() for c in df.columns]
                
                titles = set()
                if 'title' in df.columns:
                    # Filter for movies if type exists
                    if 'type' in df.columns:
                         df = df[df['type'].astype(str).str.lower() == 'movie']
                    titles = set(df['title'].astype(str).str.strip().str.lower())
                elif 'movie name' in df.columns:
                    titles = set(df['movie name'].astype(str).str.strip().str.lower())
                
                SERVICE_TITLES[service] = titles
                print(f"Loaded {len(titles)} titles for {service}")
            except Exception as e:
                print(f"Error preloading {service}: {e}")
        else:
            print(f"File missing for {service}: {filename}")

# Call preloader at startup
preload_services()

@app.route("/movie/<title>")
def movie_details(title):
    movie_row = movies[movies['title'] == title]
    
    if movie_row.empty:
        # Fallback: Search in Service CSVs
        found_in_service = False
        movie = {}
        
        service_files = {
            "Netflix": "netflix_titles.csv",
            "Hulu": "hulu_titles.csv",
            "Disney+": "disney_plus_titles.csv",
            "Prime Video": "amazon_prime_titles.csv",
            "Peacock": "peacock.csv",
            "Apple TV+": "appletv+.csv.csv",
            "Max": "max.csv.csv",
            "Paramount+": "paramount+.csv.csv"
        }

        for service, filename in service_files.items():
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    # Normalize columns
                    df.columns = [c.lower().strip() for c in df.columns]
                    
                    # Identify title column
                    title_col = 'title' if 'title' in df.columns else ('movie name' if 'movie name' in df.columns else None)
                    
                    if title_col:
                        # Case-insensitive matching
                        match = df[df[title_col].astype(str).str.lower().str.strip() == title.lower().strip()]
                        if not match.empty:
                            row = match.iloc[0]
                            
                            # Extract metadata
                            genre = row.get('listed_in', row.get('genre', 'N/A'))
                            desc = row.get('description', row.get('synopsis', ''))
                            
                            # Handle different rating columns or generate random
                            rating = 'N/A'
                            if 'rating' in row: rating = row['rating']
                            elif 'imdb_rating' in row: rating = row['imdb_rating']
                            else: rating = round(random.uniform(6.0, 9.0), 1)

                            movie = {
                                'title': title, # Use requested title for consistency
                                'genre': genre,
                                'description': desc,
                                'rating': rating,
                                'language': 'English', # Default
                                'poster': get_poster(title)
                            }
                            found_in_service = True
                            break
                except:
                    continue
        
        if not found_in_service:
            return "Movie not found", 404
    else:
        movie = movie_row.iloc[0].to_dict()
        movie['poster'] = get_poster(title)
    
    # Check Watch Options
    watch_options = []
    norm_title = title.lower().strip()
    for service, titles in SERVICE_TITLES.items():
        if norm_title in titles:
            watch_options.append(service)
    
    # Get reviews
    reviews = []
    if os.path.exists(REVIEWS_FILE):
        df = pd.read_csv(REVIEWS_FILE)
        # Handle cases where rating might not exist yet
        if 'rating' not in df.columns:
            df['rating'] = None
            
        reviews = df[df['movie_title'] == title].to_dict(orient="records")
        # Sort reviews by date desc if possible, or just reverse
        reviews.reverse()

    # Check if in watchlist
    is_in_watchlist = False
    if current_user.is_authenticated and os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 1 and row[1] == title and row[0] == current_user.id:
                    is_in_watchlist = True
                    break

    # Recommendations (if any positive reviews exist or just based on content)
    recommendations = []
    # Simple content-based recs for this movie
    desc = movie['description'] if pd.notna(movie['description']) else ""
    query_vec = vectorizer.transform([desc])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sim_indices = similarity.argsort()[-6:][::-1] # Top 5 + self
    
    # Bounds check added here
    sim_indices = [i for i in sim_indices if i < len(movies) and movies.iloc[i]['title'] != title]
    
    rec_candidates = movies.iloc[sim_indices].copy()
    poster_map = fetch_posters_parallel(rec_candidates)
    rec_candidates['poster'] = rec_candidates.index.map(poster_map)
    recommendations = rec_candidates[rec_candidates['poster'].notna()].head(5).to_dict(orient="records")

    # Get Trailer ID
    trailer_id = get_trailer_id(title)

    return render_template("details.html", movie=movie, reviews=reviews, is_in_watchlist=is_in_watchlist, recommendations=recommendations, watch_options=watch_options, trailer_id=trailer_id)

@app.route("/submit_review", methods=["POST"])
@login_required
def submit_review():
    movie_title = request.form.get("movie_title")
    review_text = request.form.get("review")
    rating = request.form.get("rating") # New rating field
    
    if not movie_title or not review_text:
        flash("Missing data", "error")
        return redirect(url_for("movie_details", title=movie_title))
    
    # Ensure items exist in reviews file header if creating new
    if not os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["movie_title", "review", "sentiment", "date", "rating"]) # Added rating

    vec = sentiment_vectorizer.transform([review_text])
    pred = sentiment_model.predict(vec)[0]

    try:
        # Check current header to see if we need to append rating or if column exists
        header = []
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
             reader = csv.reader(f)
             header = next(reader, [])
        
        # If 'rating' not in header, we might have an issue appending plainly, 
        # but for simplicity in this project we just append. 
        # Ideally we would migrate the CSV.
        
        with open(REVIEWS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # If the file was created by us just now, it has rating.
            # If existing file doesn't have rating column, this append will effectively add it as 5th col.
            # Read logic handles missing col by defaulting to None.
            writer.writerow([movie_title, review_text, pred, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
            
    except Exception as e:
        print(f"Error saving review: {e}")
        flash("Error saving review", "error")
        return redirect(url_for("movie_details", title=movie_title))

    flash(f"Review submitted! Sentiment: {pred}", "success")
    return redirect(url_for("movie_details", title=movie_title))

# --- Subscription Optimization Logic ---
SUBSCRIPTION_PRICES = {
    "Netflix": 15.49,
    "Hulu": 7.99,
    "Disney+": 13.99,
    "Prime Video": 14.99,
    "Peacock": 5.99,
    "Apple TV+": 9.99,
    "Max": 15.99,
    "Paramount+": 5.99
}

def calculate_subscription_value(watchlist_titles):
    service_counts = {k: 0 for k in SUBSCRIPTION_PRICES.keys()}
    
    # Map for service lookup
    service_map = {
        "Netflix": "netflix",
        "Hulu": "hulu",
        "Disney+": "disney_plus",
        "Prime Video": "amazon_prime",
        "Peacock": "peacock",
        "Apple TV+": "appletv",
        "Max": "max",
        "Paramount+": "paramount"
    }

    # Pre-load all service titles for optimization
    # In a real app, we would cache this better.
    # We can use the global SERVICE_TITLES if available, otherwise reuse load logic
    # But since SERVICE_TITLES is global and populated at startup, let's use that!
    
    # --- Greedy Set Cover Algorithm for Bundle Optimization ---
    
    # 1. Identify all unique movies in watchlist that are available on at least one service
    all_coverable_movies = set()
    service_to_movies = {} # Map service -> set of coverable movies
    
    for display_name, service_key in service_map.items():
        # Get titles for this service
        titles_set = set()
        for k, v in SERVICE_TITLES.items():
            if service_key in k.lower().replace(" ", ""):
                 titles_set = v
                 break
        if not titles_set:
             titles_set = SERVICE_TITLES.get(display_name, set())
        if not titles_set and display_name == "Peacock":
            titles_set = SERVICE_TITLES.get("Peacock Premium", set())
            
        covered_in_this_service = set()
        for title in watchlist_titles:
            if title.lower().strip() in titles_set:
                covered_in_this_service.add(title)
                all_coverable_movies.add(title)
        
        service_to_movies[display_name] = covered_in_this_service
        service_counts[display_name] = len(covered_in_this_service) # Keep track for individual stats

    # 2. Greedy Loop
    uncovered_movies = all_coverable_movies.copy()
    bundle = []
    total_bundle_cost = 0.0
    
    while uncovered_movies:
        best_service = None
        best_cover_count = 0
        
        # Find service that covers the most *currently uncovered* movies
        for service, movies_set in service_to_movies.items():
            if service in [b['service'] for b in bundle]:
                continue # Already picked
            
            new_cover_count = len(movies_set.intersection(uncovered_movies))
            if new_cover_count > best_cover_count:
                best_cover_count = new_cover_count
                best_service = service
        
        if best_service:
            # Add to bundle
            bundle.append({
                'service': best_service,
                'price': SUBSCRIPTION_PRICES.get(best_service, 0),
                'newly_covered': len(service_to_movies[best_service].intersection(uncovered_movies))
            })
            total_bundle_cost += SUBSCRIPTION_PRICES.get(best_service, 0)
            uncovered_movies -= service_to_movies[best_service]
        else:
            # Cannot cover remaining movies (should not happen if all_coverable logic is correct)
            break
            
    # Calculate stats for the bundle
    bundle_coverage_pct = 0
    if watchlist_titles:
        bundle_coverage_pct = int((len(all_coverable_movies) / len(watchlist_titles)) * 100)
    
    # Individual Rankings (Original Logic preserved for breakdown)
    individual_results = []
    for service, count in service_counts.items():
        if count > 0:
            price = SUBSCRIPTION_PRICES.get(service, 0)
            cost_per_movie = price / count
            individual_results.append({
                'service': service,
                'count': count,
                'price': price,
                'cost_per_movie': cost_per_movie
            })
    individual_results.sort(key=lambda x: (-x['count'], x['cost_per_movie']))
    
    return {
        'bundle': bundle,
        'total_cost': total_bundle_cost,
        'coverage_pct': bundle_coverage_pct,
        'individual': individual_results,
        'total_watchlist_count': len(watchlist_titles),
        'coverable_count': len(all_coverable_movies)
    }


@app.route("/watchlist")
@login_required
def watchlist():
    watchlist_movies = []
    
    # Load reviews first for lookup
    user_reviews = {}
    if os.path.exists(REVIEWS_FILE):
        try:
            df = pd.read_csv(REVIEWS_FILE)
            # Group by title and get the latest review
            if not df.empty:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date', ascending=False)
                
                # Create a dict of title -> review dict
                for _, row in df.iterrows():
                    title = row['movie_title']
                    if title not in user_reviews:
                        user_reviews[title] = {
                            'review': row['review'],
                            'sentiment': row['sentiment'],
                            'date': row['date'].strftime("%Y-%m-%d") if isinstance(row['date'], pd.Timestamp) else row['date']
                        }
        except Exception as e:
            print(f"Error loading reviews for watchlist: {e}")

    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            watchlist_titles = [row[1] for row in reader if len(row) > 1 and row[0] == current_user.id]
            
        for title in watchlist_titles:
            movie_row = movies[movies['title'] == title]
            if not movie_row.empty:
                m = movie_row.iloc[0].to_dict()
                m['poster'] = get_poster(title)
                
                # Attach review if exists
                if title in user_reviews:
                    m['user_review'] = user_reviews[title]['review']
                    m['review_sentiment'] = user_reviews[title]['sentiment']
                    m['review_date'] = user_reviews[title]['date']
                
                if m['poster']:
                    watchlist_movies.append(m)
    
    # --- Subscription Optimization ---
    optimization = calculate_subscription_value(watchlist_titles)
    
    return render_template("watchlist.html", movies=watchlist_movies, page="watchlist", 
                         optimization=optimization)

@app.route("/watchlist/add", methods=["POST"])
@login_required
def add_to_watchlist():
    if request.is_json:
        title = request.json.get("title")
    else:
        title = request.form.get("title")
        
    if not title:
        if request.is_json:
            return {"status": "error", "message": "No title"}, 400
        return redirect(url_for("home"))
        
    # Check if exists
    exists = False
    rows = []
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            for row in rows:
                if len(row) > 1 and row[1] == title and row[0] == current_user.id:
                    exists = True
    
    if not exists:
        with open(WATCHLIST_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_user.id, title])
            
    if request.is_json:
        return {"status": "success", "message": "Added to watchlist"}
    return redirect(url_for("movie_details", title=title))

@app.route("/watchlist/remove", methods=["POST"])
@login_required
def remove_from_watchlist():
    title = request.form.get("title")
    origin = request.form.get("origin", "details") # details or watchlist page
    
    if not title:
        return redirect(url_for("home"))
        
    rows = []
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
    header = rows[0] if rows else ["user_id", "movie_title"]
    data_rows = [r for r in rows[1:] if not (len(r) > 1 and r[1] == title and r[0] == current_user.id)]
    
    with open(WATCHLIST_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
        
    if origin == "watchlist":
        return redirect(url_for("watchlist"))
    return redirect(url_for("movie_details", title=title))

@app.route("/favorites")
@login_required
def favorites():
    favorite_movies = []
    
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            fav_titles = [row[1] for row in reader if len(row) > 1 and row[0] == current_user.id]
            
        for title in fav_titles:
            movie_row = movies[movies['title'] == title]
            if not movie_row.empty:
                m = movie_row.iloc[0].to_dict()
                m['poster'] = get_poster(title)
                if m['poster']:
                    favorite_movies.append(m)
    
    return render_template("favorites.html", movies=favorite_movies, page="favorites")

@app.route("/favorites/add", methods=["POST"])
@login_required
def add_to_favorites():
    data = request.json
    title = data.get("title")
    if not title:
        return {"status": "error", "message": "No title provided"}, 400
        
    # Check if exists
    exists = False
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1 and row[1] == title and row[0] == current_user.id:
                    exists = True
                    break
    
    if not exists:
        with open(FAVORITES_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_user.id, title])
        return {"status": "success", "message": "Added to favorites", "action": "added"}
    else:
        # Toggle: Remove it
        rows = []
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        header = rows[0]
        data_rows = [r for r in rows[1:] if not (len(r) > 1 and r[1] == title and r[0] == current_user.id)]
        
        with open(FAVORITES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        return {"status": "success", "message": "Removed from favorites", "action": "removed"}

@app.route("/favorites/remove", methods=["POST"])
@login_required
def remove_from_favorites():
    title = request.form.get("title")
    if not title:
        return redirect(url_for("favorites"))
        
    rows = []
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
    header = rows[0] if rows else ["user_id", "movie_title"]
    data_rows = [r for r in rows[1:] if not (len(r) > 1 and r[1] == title and r[0] == current_user.id)]
    
    with open(FAVORITES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
        
    return redirect(url_for("favorites"))

@app.context_processor
def inject_context():
    user_settings = {}
    favs = []
    if current_user.is_authenticated:
        user_settings = load_user_settings(current_user.id)
        if os.path.exists(FAVORITES_FILE):
            try:
                with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    favs = [row[1] for row in reader if len(row) > 1 and row[0] == current_user.id]
            except:
                pass
    return dict(user_settings=user_settings, user_favorites=favs)

@app.route("/recent")
def recent_reviews():
    recent_items = []
    if os.path.exists(REVIEWS_FILE):
        try:
            df = pd.read_csv(REVIEWS_FILE)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date', ascending=False)
            
            # Get top 20 recent reviews
            recent_reviews_df = df.head(20)
            
            for _, row in recent_reviews_df.iterrows():
                title = row['movie_title']
                movie_row = movies[movies['title'] == title]
                
                if not movie_row.empty:
                    m = movie_row.iloc[0].to_dict()
                    m['poster'] = get_poster(title)
                    
                    # Add review data
                    m['user_review'] = row['review']
                    m['review_sentiment'] = row['sentiment']
                    m['review_date'] = row['date'].strftime("%Y-%m-%d %H:%M") if isinstance(row['date'], pd.Timestamp) else row['date']
                    
                    if m['poster']:
                        recent_items.append(m)
        except Exception as e:
            print(f"Error: {e}")
            
    return render_template("recent.html", movies=recent_items, page="recent")

def load_service_data(service_name):
    # Map service names to filenames
    file_map = {
        "netflix": "netflix_titles.csv",
        "hulu": "hulu_titles.csv",
        "disney_plus": "disney_plus_titles.csv",
        "amazon_prime": "amazon_prime_titles.csv",
        "peacock": "peacock.csv",
        "appletv": "appletv+.csv.csv",
        "max": "max.csv.csv",
        "paramount": "paramount+.csv.csv"
    }
    
    filename = file_map.get(service_name)
    if not filename or not os.path.exists(filename):
        print(f"File not found for {service_name}: {filename}")
        return []
    
    try:
        df = pd.read_csv(filename)
        items = []
        
        # Schema detection
        columns = [c.lower().strip() for c in df.columns]
        df.columns = columns
        
        # Schema 1: Standard (show_id, type, title...)
        if 'title' in columns:
            if 'type' in df.columns:
                # Case insensitive check for 'Movie'
                df = df[df['type'].astype(str).str.lower() == 'movie']
            
            for _, row in df.iterrows():
                item = {
                    'title': row['title'],
                    'year': row['release_year'] if 'release_year' in row else (row['year'] if 'year' in row else 'N/A'),
                    'description': row['description'] if 'description' in row else '',
                    'genre': row['listed_in'] if 'listed_in' in row else (row['genres'] if 'genres' in row else 'N/A'),
                    'service': service_name.replace("_", " ").capitalize(),
                    'rating': round(random.uniform(6.0, 9.9), 1)
                }
                # Check for real rating if exists
                if 'rating' in row and pd.notna(row['rating']):
                     try: item['rating'] = float(row['rating'])
                     except: pass
                items.append(item)
                
        # Schema 2: Custom Simple (Movie Name, Rating(10), Genre...) - e.g. Peacock
        elif 'movie name' in columns:
            for _, row in df.iterrows():
                item = {
                    'title': row['movie name'],
                    'year': 'N/A', # Simple schema might not have year
                    'description': row['description'] if 'description' in row else '',
                    'genre': row['genre'] if 'genre' in row else 'N/A',
                    'service': service_name.replace("_", " ").capitalize(),
                    'rating': round(random.uniform(6.0, 9.9), 1)
                }
                # Check for real rating if exists
                if 'rating' in row and pd.notna(row['rating']):
                     try: item['rating'] = float(row['rating'])
                     except: pass
                items.append(item)
                
        else:
            print(f"Unknown schema for {filename}. Columns: {columns}")

        return items
    except Exception as e:
        print(f"Error loading {service_name}: {e}")
        return []

@app.route("/calculate_cinematrix", methods=["POST"])
@login_required
def calculate_cinematrix():
    data = request.json
    time = data.get("time") # short, medium, long
    mood = data.get("mood") # happy, emotional, thrilled, relaxed, intense
    who = data.get("who") # alone, date, friends, family
    energy = data.get("energy") # high, medium, low

    # Heuristic Mappings
    
    # 1. MOOD MAPPING
    mood_genres = {
        "happy": ["Comedy", "Animation", "Family", "Musical"],
        "emotional": ["Drama", "Romance", "Biography"],
        "thrilled": ["Thriller", "Horror", "Mystery", "Crime"],
        "relaxed": ["Documentary", "Family", "Animation"],
        "intense": ["Action", "War", "Sci-Fi", "Crime"]
    }
    
    # 2. WHO MAPPING
    who_genres = {
        "alone": ["Drama", "Sci-Fi", "Thriller", "Horror"],
        "date": ["Romance", "Comedy", "Drama"],
        "friends": ["Action", "Comedy", "Horror", "Adventure"],
        "family": ["Family", "Animation", "Adventure"]
    }
    
    # 3. ENERGY MAPPING (Pace)
    energy_genres = {
        "high": ["Action", "Adventure", "Sci-Fi", "Thriller", "Horror"],
        "medium": ["Comedy", "Crime", "Mystery", "Musical"],
        "low": ["Drama", "Romance", "Family", "Biography", "History"]
    }
    
    # 4. TIME MAPPING (Heuristic based on genre often)
    time_genres = {
        "short": ["Animation", "Comedy", "Horror"],
        "medium": ["Action", "Thriller", "Romance", "Crime", "Mystery"],
        "long": ["Drama", "History", "Biography", "Sci-Fi", "Adventure"]
    }
    
    scored_movies = []
    
    target_mood = mood_genres.get(mood, [])
    target_who = who_genres.get(who, [])
    target_energy = energy_genres.get(energy, [])
    target_time = time_genres.get(time, [])
    
    # Get user active platforms for "Available on" check
    user_settings = load_user_settings(current_user.id)
    selected_platforms_str = user_settings.get("selected_platforms", "")
    # Clean up platform names for matching
    user_platforms = [p.strip().lower().replace(" ", "") for p in selected_platforms_str.split(",")]
    # Map friendly names back
    platform_display_map = {
        "netflix": "Netflix", "hulu": "Hulu", "disney+": "Disney+", "amazonprime": "Prime Video", 
        "peacock": "Peacock", "appletv+": "Apple TV+", "max": "Max", "paramount+": "Paramount+"
    }
    
    for _, row in movies.iterrows():
        score = 0
        genres = str(row['genre']).split('|')
        
        # Scoring
        for g in genres:
            g = g.strip()
            if g in target_mood: score += 30
            if g in target_who: score += 20
            if g in target_energy: score += 20
            if g in target_time: score += 10
            
        # Add rating boost (0-10 points)
        try:
            rating = float(row['rating'])
            score += rating
        except:
            pass
            
        # Check Availability
        available_on = []
        norm_title = str(row['title']).lower().strip()
        
        # Check in SERVICE_TITLES
        for svc, titles in SERVICE_TITLES.items():
            if norm_title in titles:
                # Clean service name to match user settings
                clean_svc = svc.lower().replace(" ", "").replace("premium", "")
                # Flexible matching
                for up in user_platforms:
                    if up in clean_svc or clean_svc in up:
                         available_on.append(svc)
                         score += 50 # Huge boost for available movies
                         break

        # Random jitter to vary results slightly
        score += random.uniform(0, 5)
        
        # Only consider decent matches
        if score > 40:
             scored_movies.append({
                 'title': row['title'],
                 'rating': row['rating'],
                 'poster': get_poster(row['title']),
                 'match_score': int(min(score, 100)),
                 'available_on': available_on,
                 'genres': genres
             })
             
    # Sort and take top 1
    scored_movies.sort(key=lambda x: x['match_score'], reverse=True)
    
    if not scored_movies:
        return []

    top_movie = scored_movies[0]
    
    # Construct AI Explanation
    explanation_points = []
    
    # Mood point
    explanation_points.append(f"Matches your {mood} mood")
    
    # Who point (if family or friends)
    if who == "family":
        explanation_points.append("Family-friendly choice")
    elif who == "friends":
         explanation_points.append("Great for watching with friends")
    elif who == "date":
         explanation_points.append("Perfect for a date night")
    
    # Energy point
    if energy == "low":
        explanation_points.append("fits your low-energy vibe")
    elif energy == "high":
        explanation_points.append("High-energy excitement")

    # Rating point
    try:
        r = float(top_movie['rating'])
        if r >= 8.0:
            percentage = int(r * 10)
            explanation_points.append(f"High positive reviews ({percentage}%)")
    except:
        pass

    # Availability point
    if top_movie['available_on']:
        svc = top_movie['available_on'][0]
        explanation_points.append(f"Available on your active subscription ({svc})")
    
    # Assign explanation string
    # We'll join them with newlines or special chars for frontend parsing, 
    # but the prompt asked for "Why this movie?" format. 
    # Let's pass array
    top_movie['explanation'] = explanation_points
    
    return [top_movie] # Return single item list for frontend compatibility

@app.route("/services/<service_name>")
@login_required
def service_catalog(service_name):
    valid_services = ["netflix", "hulu", "disney_plus", "amazon_prime", "peacock", "appletv", "max", "paramount"]
    
    if service_name not in valid_services:
        flash("Service not found", "error")
        return redirect(url_for("services"))
        
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Load data
    all_movies = load_service_data(service_name)
    
    # Pagination
    total_items = len(all_movies)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    current_items = all_movies[start_idx:end_idx]
    
    # DataFrame for poster fetching compatibility
    if current_items:
        df_page = pd.DataFrame(current_items)
        poster_map = fetch_posters_parallel(df_page)
        
        final_items = []
        for item in current_items:
            # Add poster
            # Careful with index matching if duplicates exist, but here we iterate list
            # We need to find the poster for THIS specific item title
            # fetch_posters_parallel returns {index: url} based on the input df index
            
            # Re-map using the temporary df index
            # Find the index in df_page where title matches
            matches = df_page[df_page['title'] == item['title']].index
            if not matches.empty:
                idx = matches[0]
                item['poster'] = poster_map.get(idx)
            else:
                item['poster'] = None
                
            final_items.append(item)
    else:
        final_items = []

    total_pages = (total_items + per_page - 1) // per_page
    
    display_name = service_name.replace("_", " ").title()
    if service_name == "appletv": display_name = "Apple TV+"
    if service_name == "max": display_name = "Max"
    if service_name == "disney_plus": display_name = "Disney+"
    if service_name == "amazon_prime": display_name = "Prime Video"
    if service_name == "paramount": display_name = "Paramount+"
    if service_name == "peacock": display_name = "Peacock Premium"
    
    return render_template("service_catalog.html", 
                           movies=final_items, 
                           service=display_name, 
                           page_num=page, 
                           total_pages=total_pages,
                           current_page="services",
                           # Pass raw service_name for pagination links
                            service_slug=service_name)

def get_all_movies_from_all_sources():
    all_movies = []
    
    # 1. Add from main movie.csv
    # Normalize keys to match a common schema
    for i, row in movies.iterrows():
        all_movies.append({
            'title': row['title'],
            'genre': row['genre'] if pd.notna(row['genre']) else '',
            'description': row['description'] if pd.notna(row['description']) else '',
            'rating': row['rating'] if pd.notna(row['rating']) else round(random.uniform(5.5, 9.9), 1),
            'poster': None, # Will fetch later
            'source': 'Main Library'
        })

    # 2. Add from service CSVs
    service_files = {
        "Netflix": "netflix_titles.csv",
        "Hulu": "hulu_titles.csv",
        "Disney+": "disney_plus_titles.csv",
        "Prime Video": "amazon_prime_titles.csv",
        "Peacock": "peacock.csv",
        "Apple TV+": "appletv+.csv.csv",
        "Max": "max.csv.csv",
        "Paramount+": "paramount+.csv.csv"
    }

    for service, filename in service_files.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Normalize columns
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Check for 'type' column and filter for 'Movie'
                if 'type' in df.columns:
                     df = df[df['type'].astype(str).str.lower() == 'movie']

                for _, row in df.iterrows():
                    title = 'N/A'
                    genre = 'N/A'
                    desc = ''
                    
                    if 'title' in df.columns: title = row['title']
                    elif 'movie name' in df.columns: title = row['movie name']
                    
                    if 'listed_in' in df.columns: genre = row['listed_in']
                    elif 'genres' in df.columns: genre = row['genres']
                    elif 'genre' in df.columns: genre = row['genre']
                    
                    if 'description' in df.columns: desc = row['description']
                    
                    if title != 'N/A':
                        rating = round(random.uniform(6.0, 9.9), 1)
                        # Try to find real rating if exists in exotic schema
                        if 'rating' in df.columns and pd.notna(row['rating']):
                            try:
                                rating = float(row['rating'])
                            except:
                                pass # Keep random if fail

                        all_movies.append({
                            'title': title,
                            'genre': str(genre),
                            'description': str(desc),
                            'rating': rating,
                            'poster': None,
                            'source': service
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return all_movies

def check_attribute(movie, category):
    genres = movie.get('genre', '').lower()
    
    if category == "slow-paced":
        slow_keywords = ['drama', 'romance', 'biography', 'history', 'documentary', 'arts']
        fast_keywords = ['action', 'thriller', 'adventure', 'horror', 'sci-fi', 'mystery', 'survival']
        is_slow = any(k in genres for k in slow_keywords)
        is_fast = any(k in genres for k in fast_keywords)
        return is_slow and not is_fast
        
    elif category == "fast-paced":
        keywords = ['action', 'thriller', 'adventure', 'sci-fi', 'horror', 'mystery']
        return any(k in genres for k in keywords)
        
    elif category == "simple-plot":
        keywords = ['comedy', 'family', 'animation', 'musical', 'romance']
        complex_keywords = ['sci-fi', 'mystery', 'crime', 'thriller', 'psychological']
        is_simple = any(k in genres for k in keywords)
        is_complex = any(k in genres for k in complex_keywords)
        return is_simple and not is_complex
        
    elif category == "complex-plot":
        keywords = ['sci-fi', 'mystery', 'crime', 'thriller', 'psychological', 'suspense']
        return any(k in genres for k in keywords)
        
    elif category == "light-theme":
        keywords = ['comedy', 'family', 'animation', 'musical', 'fantasy']
        dark_keywords = ['horror', 'crime', 'mystery', 'thriller', 'war', 'dark']
        is_light = any(k in genres for k in keywords)
        is_dark = any(k in genres for k in dark_keywords)
        return is_light and not is_dark
        
    elif category == "dark-theme":
        keywords = ['horror', 'crime', 'mystery', 'thriller', 'war', 'dark', 'noir']
        return any(k in genres for k in keywords)
        
    elif category == "watch-myself":
        # Introspective or deep genres
        keywords = ['drama', 'biography', 'documentary', 'history', 'war']
        return any(k in genres for k in keywords)
        
    elif category == "watch-friends":
        # Fun, exciting, or scary genres
        keywords = ['action', 'comedy', 'horror', 'adventure', 'sport', 'musical']
        return any(k in genres for k in keywords)
        
    return False

@app.route("/attributes/<category>")
@login_required
def show_attribute(category):
    valid_categories = [
        "slow-paced", "fast-paced", 
        "simple-plot", "complex-plot", 
        "light-theme", "dark-theme", 
        "watch-myself", "watch-friends"
    ]
    
    if category in valid_categories:
        all_movies = get_all_movies_from_all_sources()
        filtered_movies = [m for m in all_movies if check_attribute(m, category)]
        
        # De-duplicate by title
        seen_titles = set()
        unique_movies = []
        for m in filtered_movies:
            if m['title'].lower() not in seen_titles:
                unique_movies.append(m)
                seen_titles.add(m['title'].lower())
        
        display_movies = unique_movies[:20] 
        
        # Convert list of dicts to DataFrame for fetch_posters_parallel compatibility
        if display_movies:
            df = pd.DataFrame(display_movies)
            poster_map = fetch_posters_parallel(df)
            
            final_movies = []
            for i, movie in enumerate(display_movies):
                poster_url = poster_map.get(i)
                movie['poster'] = poster_url
                if poster_url:
                    final_movies.append(movie)
        else:
            final_movies = []

        title_map = {
            "slow-paced": "Slow-Paced Movies",
            "fast-paced": "Fast-Paced Movies",
            "simple-plot": "Simple Plot Movies",
            "complex-plot": "Complex Plot Movies",
            "light-theme": "Light Theme Movies",
            "dark-theme": "Dark Theme Movies",
            "watch-myself": "Movies to Watch Alone",
            "watch-friends": "Movies to Watch with Friends"
        }

        return render_template("index.html", movies=final_movies, page="attributes", search_query=title_map.get(category, category))
    
    return redirect(url_for("attributes"))

if __name__ == "__main__":
    app.run(debug=True)
