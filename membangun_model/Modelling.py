print(">>> TES: SKRIP Modelling.py MULAI DI SINI !!! <<<", flush=True)
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import joblib


def clean_text_for_token(text):
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""

def create_content_soup(df):
    print("CREATE_CONTENT_SOUP: Memulai pembuatan 'content soup'...", flush=True)
    df_copy = df.copy()
    df_copy['overview_processed'] = df_copy['overview'].fillna('').astype(str).str.lower()
    def process_genres_list(genre_list):
        if isinstance(genre_list, list):
            return " ".join([clean_text_for_token(genre) for genre in genre_list])
        elif isinstance(genre_list, str):
            try:
                actual_list = ast.literal_eval(genre_list)
                return " ".join([clean_text_for_token(genre) for genre in actual_list])
            except: return ""
        return ""
    df_copy['genres_soup'] = df_copy.get('genres_processed', "").apply(process_genres_list)
    df_copy['director_soup'] = df_copy.get('director', "").apply(lambda x: clean_text_for_token(x) if pd.notnull(x) else "")
    df_copy['soup'] = (df_copy['overview_processed'] + ' ' +
                       df_copy['genres_soup'] + ' ' + df_copy['genres_soup'] + ' ' +
                       df_copy['director_soup'] + ' ' + df_copy['director_soup'])
    print("CREATE_CONTENT_SOUP: 'Content soup' berhasil dibuat.", flush=True)
    return df_copy[['id', 'title', 'soup']]

def load_data_and_generate_soup(dataset_path):
    print(f"LOAD_AND_SOUP: Mencoba memuat dataset dari: {dataset_path}", flush=True)
    try:
        df = pd.read_csv(dataset_path)
        print(f"LOAD_AND_SOUP: Dataset berhasil dimuat. Baris: {len(df)}, Kolom: {len(df.columns)}", flush=True)
        return create_content_soup(df)
    except Exception as e:
        print(f"LOAD_AND_SOUP: ERROR - {e}", flush=True)
        return None

def get_recommendations(movie_title, cosine_sim_matrix_input, data, movie_indices, top_n=10):
    if movie_title not in movie_indices:
        return pd.Series(dtype='object')
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix_input[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices_output = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices_output]

if __name__ == "__main__":
    import argparse
    print(">>> BLOK __main__ di Modelling.py TERPANGGIL <<<", flush=True)

    parser = argparse.ArgumentParser(description="Content-based Movie Recommender Training with MLflow")
    parser.add_argument('--dataset_path', type=str, default="Membangun_Model/tmdb_movies_processed.csv", help='Path to the processed movie dataset CSV')
    args = parser.parse_args()

    # Inisialisasi MLflow experiment (semua run akan dicatat di experiment ini)
    experiment_name = "Movie Recommender - Content Based"
    mlflow.set_experiment(experiment_name)

    # Aktifkan autologging untuk scikit-learn (parameter, model, dsb akan otomatis dicatat)
    mlflow.autolog()

    dataset_path = Path(args.dataset_path)
    print(f"MAIN: Akan memuat data dan membuat soup dari: {dataset_path}", flush=True)
    movie_data_with_soup = load_data_and_generate_soup(dataset_path)

    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        # Mulai MLflow run (semua parameter, artifact, dsb akan dicatat di run ini)
        with mlflow.start_run(run_name="ContentBasedRecommender_Run1") as run:
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=3, max_df=0.7)
            tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'].fillna(''))
            cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            joblib.dump(tfidf, "tfidf_vectorizer.pkl")
            np.savez_compressed("cosine_matrix.npz", cosine_sim_matrix=cosine_sim_matrix)

            mlflow.log_artifact("tfidf_vectorizer.pkl")
            mlflow.log_artifact("cosine_matrix.npz")

            indices = pd.Series(movie_data_with_soup.index, index=movie_data_with_soup['title']).drop_duplicates()
            test_movie_title = movie_data_with_soup['title'].iloc[0]
            recommendations = get_recommendations(test_movie_title, cosine_sim_matrix, movie_data_with_soup, indices)

            print(f"\n--- Rekomendasi untuk '{test_movie_title}' ---")
            if not recommendations.empty:
                print("\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)]))
            else:
                print("Tidak ada rekomendasi ditemukan.")

    print("\nSkrip modelling.py selesai.", flush=True)
