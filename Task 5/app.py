# --------------------
#Movie Recommendation
# -------------------

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[["title", "overview", "vote_average"]]
    movies["overview"] = movies["overview"].fillna("")
    return movies

movies = load_data()

# TF-IDF Vectorization
@st.cache_resource
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies["title"].str.lower()).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(movies)

# Recommendation Function
def recommend_movie(title, top_n=5):
    title = title.lower()
    if title not in indices:
        return pd.DataFrame()
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return movies.iloc[sim_indices][["title", "overview", "vote_average"]]

# -------------
# Streamlit UI
# ------------
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get top 5 recommendations with overview and IMDb rating (TF-IDF + Cosine Similarity).")

# Dropdown to select movie
movie_list = movies["title"].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Get Recommendations"):
    recommendations = recommend_movie(selected_movie, top_n=5)
    if not recommendations.empty:
        st.subheader(f"Top 5 Recommendations for '{selected_movie}':")
        for _, row in recommendations.iterrows():
            st.markdown(f"**🎬 {row['title']}**")
            st.write(f"IMDb Rating: {row['vote_average']}")
            st.write(row['overview'] if row['overview'].strip() else "_No overview available_")
            st.write("---")
    else:
        st.error("Movie not found in dataset!")
