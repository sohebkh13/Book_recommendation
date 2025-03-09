import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Functions for data loading
@st.cache_data
def load_data():
    # List of encodings to try
    encodings = ["cp1252"]
    
    for encoding in encodings:
        try:
            # Try loading with current encoding
            users = pd.read_csv("P505/Users_clean.csv", encoding=encoding)
            books = pd.read_csv("P505/Books_clean.csv", encoding=encoding)
            ratings = pd.read_csv("P505/Ratings_clean.csv", encoding=encoding)
            
            #st.success(f"Successfully loaded clean datasets with {encoding} encoding")
            return users, books, ratings
            
        except UnicodeDecodeError:
            # If this encoding doesn't work, try the next one
            continue
        except FileNotFoundError as e:
            # If files aren't found, show error
            st.error(f"Could not find clean datasets. Error: {str(e)}")
            st.warning("Please make sure P505/Users_clean.csv, P505/Books_clean.csv, and P505/Ratings_clean.csv exist.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # If we've tried all encodings and none worked
    st.error("Could not load datasets with any of the attempted encodings.")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Function to display missing values
def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={df.index.name:'col_name', 0:'Missing Values', 1:'% of Total Values'})
    mz_table['Data_type'] = df.dtypes
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)
    return mz_table.reset_index()

# Create recommendation system functions
def get_user_book_ratings(user_id, ratings_df, books_df):
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    user_books = pd.merge(user_ratings, books_df, on='ISBN')
    return user_books

def get_popular_books(books_df, ratings_df, n=10):
    # Get books with highest average rating and minimum number of ratings
    book_ratings = ratings_df.groupby('ISBN')['Book-Rating'].agg(['mean', 'count']).reset_index()
    # Filter books with at least 10 ratings
    popular_books = book_ratings[book_ratings['count'] >= 10].sort_values('mean', ascending=False)
    # Get book details
    popular_books = pd.merge(popular_books, books_df, on='ISBN')
    return popular_books.head(n)

# Original collaborative filtering function
def get_book_recommendations(user_id, ratings_df, books_df, n=5):
    """
    Generate personalized book recommendations using a more sophisticated collaborative filtering approach
    """
    random.seed(42)
    np.random.seed(42)
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    # Calculate user similarity based on common book ratings
    user_ratings = ratings_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating', fill_value=0)
    
    # Get the target user's ratings vector
    target_user_ratings = user_ratings.loc[user_id] if user_id in user_ratings.index else pd.Series(0, index=user_ratings.columns)
    
    # Find similar users
    user_similarities = []
    for other_user in user_ratings.index:
        if other_user != user_id:
            # Calculate cosine similarity between rating vectors
            other_user_ratings = user_ratings.loc[other_user]
            
            # Get common books (non-zero ratings)
            common_books = target_user_ratings.multiply(other_user_ratings) != 0
            common_count = common_books.sum()
            
            # Only consider users with at least one common book
            if common_count > 0:
                # Calculate similarity score
                similarity = np.dot(target_user_ratings, other_user_ratings) / (
                    np.sqrt(np.dot(target_user_ratings, target_user_ratings)) * 
                    np.sqrt(np.dot(other_user_ratings, other_user_ratings))
                )
                if not np.isnan(similarity):
                    user_similarities.append((other_user, similarity, common_count))
    
    # Sort similar users by similarity score
    user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Take top 20 similar users
    similar_users = [user for user, _, _ in user_similarities[:20]]
    
    # Get books rated by similar users but not by the target user
    user_rated_isbns = set(user_books['ISBN'])
    recommendations = {}
    
    # Find books rated highly by similar users
    for idx, similar_user in enumerate(similar_users):
        # Get books rated by this similar user
        similar_user_books = ratings_df[ratings_df['User-ID'] == similar_user]
        
        # Weight factor decreases as we go down the similarity list
        weight = 1.0 / (idx + 1.5)  # Diminishing weight for less similar users
        
        # Add weighted rating to recommendations
        for _, row in similar_user_books.iterrows():
            isbn = row['ISBN']
            rating = row['Book-Rating']
            
            # Skip books the target user has already rated
            if isbn in user_rated_isbns:
                continue
                
            # Skip zero ratings
            if rating == 0:
                continue
                
            # Add weighted rating to recommendations
            if isbn not in recommendations:
                recommendations[isbn] = {'weighted_sum': 0, 'count': 0}
            
            recommendations[isbn]['weighted_sum'] += rating * weight
            recommendations[isbn]['count'] += 1
    
    # Calculate final scores
    for isbn in recommendations:
        recommendations[isbn]['score'] = recommendations[isbn]['weighted_sum'] / recommendations[isbn]['count']
    
    # Sort by score
    sorted_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: (x[1]['score'], x[1]['count']), 
                                   reverse=True)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in sorted_recommendations[:n*2]]  # Get more than needed to ensure we have enough after merge
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        
        # Add the score information
        for isbn in top_isbns:
            if isbn in recommendations:
                mask = recommended_books['ISBN'] == isbn
                recommended_books.loc[mask, 'mean'] = recommendations[isbn]['score']
                recommended_books.loc[mask, 'count'] = recommendations[isbn]['count']
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# Simplified recommendation function
def get_book_recommendations_simple(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate personalized book recommendations using a simplified approach
    that's more efficient for a large dataset
    """
    random.seed(42)
    np.random.seed(42)
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)
    user_rated_isbns = set(user_books['ISBN'])
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Find users who rated at least one book that the target user has also rated
    common_users = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()
    common_users = [u for u in common_users if u != user_id]
    
    if progress_bar:
        progress_bar.progress(30)
    
    if len(common_users) == 0:
        return get_popular_books(books_df, ratings_df, n)
    
    # Limit to a reasonable number of similar users for performance
    if len(common_users) > 100:
        common_users = common_users[:100]
    
    # Get all ratings from these users
    similar_users_ratings = ratings_df[ratings_df['User-ID'].isin(common_users)]
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Filter out books the user has already rated
    candidate_books = similar_users_ratings[~similar_users_ratings['ISBN'].isin(user_rated_isbns)]
    
    # Group by ISBN and calculate average rating and count
    book_stats = candidate_books.groupby('ISBN').agg(
        score=('Book-Rating', 'mean'),
        supporting_users=('User-ID', 'nunique')
    ).reset_index()
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Sort by score and number of supporting users
    book_stats = book_stats.sort_values(['score', 'supporting_users'], ascending=False)
    
    # Get top N books
    top_isbns = book_stats.head(n*2)['ISBN'].tolist()  # Get more than needed to ensure we have enough after merge
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        # Merge with scores
        recommended_books = pd.merge(recommended_books, book_stats, on='ISBN')
        # Sort and get top N
        recommended_books = recommended_books.sort_values(['score', 'supporting_users'], ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# NEW FUNCTIONS FOR KNN, SVD AND NMF

# KNN Recommendation Function
@st.cache_data
def prepare_surprise_data(ratings_df):
    """Prepare data for Surprise library models"""
    # Load the data into the Surprise format
    reader = Reader(rating_scale=(0, 10))
    ratings_data = Dataset.load_from_df(
        ratings_df[['User-ID', 'ISBN', 'Book-Rating']], 
        reader
    )
    # Create the training set
    trainset = ratings_data.build_full_trainset()
    return trainset

@st.cache_data
def train_knn_model(ratings_df, k=20, sim_options=None, sample_size=10000):
    # Subsample if dataset is too large
    if len(ratings_df) > sample_size:
        ratings_sample = ratings_df.sample(sample_size, random_state=42)
    else:
        ratings_sample = ratings_df
        
    trainset = prepare_surprise_data(ratings_sample)
    
    # Build the KNN model
    algo = KNNBasic(k=k, sim_options=sim_options)
    algo.fit(trainset)
    
    return algo

def get_knn_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate book recommendations using a memory-efficient KNN collaborative filtering approach
    """
    random.seed(42)
    np.random.seed(42)
    # Get user's rated books
    user_books = ratings_df[ratings_df['User-ID'] == user_id]
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Get books rated by the user and their ratings
    user_rated_isbns = set(user_books['ISBN'].unique())
    user_ratings_dict = dict(zip(user_books['ISBN'], user_books['Book-Rating']))
    
    if progress_bar:
        progress_bar.progress(20)
    
    # Find users who rated at least one book that the target user has also rated
    users_who_rated_same_books = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()
    similar_users = [u for u in users_who_rated_same_books if u != user_id]
    
    if progress_bar:
        progress_bar.progress(30)
    
    # If too many similar users, sample for performance
    if len(similar_users) > 200:
        similar_users = random.sample(similar_users, 200)
    
    # Calculate similarity scores for each similar user
    user_similarities = []
    
    if progress_bar:
        progress_bar.progress(40)
    
    # Get ratings from similar users for books that the target user has rated
    similar_users_ratings = ratings_df[(ratings_df['User-ID'].isin(similar_users)) & 
                                      (ratings_df['ISBN'].isin(user_rated_isbns))]
    
    # Create a dictionary of ratings for each user
    similar_user_ratings = {}
    for _, row in similar_users_ratings.iterrows():
        if row['User-ID'] not in similar_user_ratings:
            similar_user_ratings[row['User-ID']] = {}
        similar_user_ratings[row['User-ID']][row['ISBN']] = row['Book-Rating']
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Calculate similarities - only for users who rated at least 2 books in common
    for sim_user, ratings in similar_user_ratings.items():
        # Find books rated by both users
        common_books = set(ratings.keys()).intersection(user_ratings_dict.keys())
        
        if len(common_books) < 2:
            continue
            
        # Extract ratings for common books
        user_ratings_array = np.array([user_ratings_dict[isbn] for isbn in common_books])
        sim_user_ratings_array = np.array([ratings[isbn] for isbn in common_books])
        
        # Calculate cosine similarity if we have enough common books
        if len(common_books) > 0:
            # Avoid division by zero
            user_magnitude = np.sqrt(np.sum(user_ratings_array**2))
            sim_user_magnitude = np.sqrt(np.sum(sim_user_ratings_array**2))
            
            if user_magnitude > 0 and sim_user_magnitude > 0:
                similarity = np.dot(user_ratings_array, sim_user_ratings_array) / (user_magnitude * sim_user_magnitude)
                user_similarities.append((sim_user, similarity, len(common_books)))
    
    if progress_bar:
        progress_bar.progress(60)
    
    # Sort similar users by similarity score
    user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Take top k most similar users
    top_similar_users = [user for user, _, _ in user_similarities[:20]]
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Get books rated by similar users but not by the target user
    books_rated_by_similar_users = ratings_df[(ratings_df['User-ID'].isin(top_similar_users)) & 
                                              (~ratings_df['ISBN'].isin(user_rated_isbns)) &
                                              (ratings_df['Book-Rating'] > 0)]
    
    if progress_bar:
        progress_bar.progress(80)
    
    # If no similar users found or no recommendations possible
    if len(books_rated_by_similar_users) == 0:
        return get_popular_books(books_df, ratings_df, n)
    
    # Calculate weighted ratings for recommendations
    book_scores = {}
    
    for sim_user, sim_score, _ in user_similarities:
        if sim_user in top_similar_users:
            # Get books rated by this similar user
            user_books = books_rated_by_similar_users[books_rated_by_similar_users['User-ID'] == sim_user]
            
            # Weight factor based on similarity
            weight = sim_score
            
            # Add weighted rating to recommendations
            for _, row in user_books.iterrows():
                isbn = row['ISBN']
                rating = row['Book-Rating']
                
                if isbn not in book_scores:
                    book_scores[isbn] = {'weighted_sum': 0, 'count': 0}
                
                book_scores[isbn]['weighted_sum'] += rating * weight
                book_scores[isbn]['count'] += 1
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Calculate final scores and sort
    for isbn in book_scores:
        book_scores[isbn]['score'] = book_scores[isbn]['weighted_sum'] / book_scores[isbn]['count']
    
    # Sort by score
    sorted_recommendations = sorted(book_scores.items(), 
                                   key=lambda x: (x[1]['score'], x[1]['count']), 
                                   reverse=True)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in sorted_recommendations[:n*2]]
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        
        # Add the score information
        for isbn, data in book_scores.items():
            if isbn in top_isbns:
                mask = recommended_books['ISBN'] == isbn
                recommended_books.loc[mask, 'mean'] = data['score']
                recommended_books.loc[mask, 'count'] = data['count']
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# SVD Recommendation Function
@st.cache_data
def train_svd_model(ratings_df, n_factors=50):
    """Train an SVD model using Surprise library"""
    trainset = prepare_surprise_data(ratings_df)
    
    # Build the SVD model
    algo = SVD(n_factors=n_factors)
    algo.fit(trainset)
    
    return algo

def get_svd_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """Generate book recommendations using SVD collaborative filtering"""
    random.seed(42)
    np.random.seed(42)
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)
    
    if len(user_books) == 0:
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(20)
    
    # Create and train the SVD model
    svd_model = train_svd_model(ratings_df)
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Get all books the user hasn't rated
    user_rated_isbns = set(user_books['ISBN'])
    all_isbns = set(books_df['ISBN'])
    unrated_isbns = all_isbns - user_rated_isbns
    
    # Sample if too many (for performance)
    if len(unrated_isbns) > 1000:
        unrated_isbns = set(random.sample(list(unrated_isbns), 1000))
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Get predictions for all unrated books
    trainset = prepare_surprise_data(ratings_df)
    predictions = []
    
    for isbn in unrated_isbns:
        try:
            uid = trainset.to_inner_uid(user_id)
            iid = trainset.to_inner_iid(isbn)
            # This works if both user and item are in the trainset
            pred = svd_model.estimate(uid, iid)
            predictions.append((isbn, pred))
        except ValueError:
            # If user or item was not in the trainset
            continue
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N books
    top_isbns = [isbn for isbn, _ in predictions[:n*2]]
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        
        # Add scores to the recommendations
        for isbn, score in predictions:
            if isbn in top_isbns:
                mask = recommended_books['ISBN'] == isbn
                recommended_books.loc[mask, 'mean'] = score
                recommended_books.loc[mask, 'count'] = 1
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)

# NMF Recommendation Function
@st.cache_data
def train_nmf_model(ratings_df, n_components=50):
    """Train an NMF model using scikit-learn"""
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='User-ID', 
        columns='ISBN', 
        values='Book-Rating', 
        fill_value=0
    )
    
    # Map original user IDs to matrix indices
    user_map = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}
    item_map = {isbn: i for i, isbn in enumerate(user_item_matrix.columns)}
    reverse_user_map = {i: user_id for user_id, i in user_map.items()}
    reverse_item_map = {i: isbn for isbn, i in item_map.items()}
    
    # Convert matrix to numpy array
    matrix = user_item_matrix.values
    
    # Train NMF model
    model = NMF(n_components=n_components, init='random', random_state=0)
    user_features = model.fit_transform(matrix)
    item_features = model.components_
    
    return model, user_features, item_features, user_map, item_map, reverse_user_map, reverse_item_map

def get_nmf_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate recommendations using NMF with memory optimization
    """
    # Set fixed seed for consistent results
    random.seed(42)
    np.random.seed(42)
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    if len(user_ratings) == 0:
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(20)
    
    # Subsample data if too large (memory optimization)
    MAX_USERS = 5000
    MAX_ITEMS = 5000
    
    # Find most active users and most rated books to create a denser submatrix
    user_counts = ratings_df['User-ID'].value_counts().head(MAX_USERS).index
    book_counts = ratings_df['ISBN'].value_counts().head(MAX_ITEMS).index
    
    # Make sure target user is included
    if user_id not in user_counts:
        user_counts = np.append(user_counts, user_id)
    
    # Filter ratings to include only selected users and books
    subset_ratings = ratings_df[
        (ratings_df['User-ID'].isin(user_counts)) & 
        (ratings_df['ISBN'].isin(book_counts))
    ]
    
    if progress_bar:
        progress_bar.progress(30)
    
    # Create user and book indices
    user_ids = subset_ratings['User-ID'].unique()
    book_ids = subset_ratings['ISBN'].unique()
    
    user_to_idx = {user: i for i, user in enumerate(user_ids)}
    book_to_idx = {book: i for i, book in enumerate(book_ids)}
    
    # Get user index
    user_idx = user_to_idx.get(user_id)
    if user_idx is None:
        return get_popular_books(books_df, ratings_df, n)
    
    if progress_bar:
        progress_bar.progress(40)
    
    # Create sparse rating matrix
    from scipy.sparse import csr_matrix
    
    # Create sparse matrix
    rows = [user_to_idx[user] for user in subset_ratings['User-ID']]
    cols = [book_to_idx[book] for book in subset_ratings['ISBN']]
    data = subset_ratings['Book-Rating'].values
    
    ratings_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(book_ids)))
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Train NMF with fewer components to save memory
    from sklearn.decomposition import NMF
    
    # Use fewer components (factors) to save memory
    model = NMF(
        n_components=15,  # Fewer components
        init='random',
        random_state=42,
        max_iter=50  # Fewer iterations
    )
    
    # Train model with sparse data
    user_factors = model.fit_transform(ratings_matrix)
    item_factors = model.components_
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Get items not rated by the user
    user_rated_books = user_ratings['ISBN'].values
    user_rated_indices = [book_to_idx[isbn] for isbn in user_rated_books if isbn in book_to_idx]
    
    # Generate predictions for unrated items
    user_vector = user_factors[user_idx]
    predictions = []
    
    # Process in batches to save memory
    BATCH_SIZE = 1000
    for i in range(0, len(book_ids), BATCH_SIZE):
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(book_ids))))
        batch_predictions = np.dot(user_vector, item_factors[:, batch_indices])
        
        for j, idx in enumerate(batch_indices):
            if idx not in user_rated_indices:
                isbn = book_ids[idx]
                predictions.append((isbn, batch_predictions[j]))
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in predictions[:n*2]]
    
    # Get book details
    if top_isbns:
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()
        
        # Add predicted scores
        for isbn, score in predictions:
            if isbn in top_isbns:
                mask = recommended_books['ISBN'] == isbn
                recommended_books.loc[mask, 'score'] = score
        
        # Add supporting users count (not really applicable to NMF, but for interface consistency)
        recommended_books['supporting_users'] = 0
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('score', ascending=False).head(n)
        return recommended_books
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)
    
def preprocess_book_data(books_df):
    """
    Preprocess book data for content-based filtering
    """
    # Fill NaN values
    books_df_copy = books_df.copy()
    books_df_copy['Book-Title'] = books_df_copy['Book-Title'].fillna('')
    books_df_copy['Book-Author'] = books_df_copy['Book-Author'].fillna('')
    books_df_copy['Publisher'] = books_df_copy['Publisher'].fillna('')
    
    # Create a combined content field for better matching
    books_df_copy['content'] = (
        books_df_copy['Book-Title'] + ' ' + 
        books_df_copy['Book-Author'] + ' ' + 
        books_df_copy['Publisher']
    )
    
    return books_df_copy

def get_content_based_recommendations(book_title, books_df, n=5, progress_bar=None):
    """
    Recommend books similar to the given book title based on content similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Set fixed seed for consistent results
    random.seed(42)
    np.random.seed(42)
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Preprocess book data
    processed_books = preprocess_book_data(books_df)
    
    if progress_bar:
        progress_bar.progress(30)
        
    # Find the book(s) with the given title (partial match)
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False, regex=False, na=False)]
    
    if len(query_books) == 0:
        # If no exact match, try fuzzy matching
        all_titles = processed_books['Book-Title'].str.lower()
        title_lower = book_title.lower()
        
        # Simple fuzzy matching - see if the search term is contained in titles
        for idx, title in enumerate(all_titles):
            if str(title_lower) in str(title) or str(title) in str(title_lower):
                query_books = processed_books.iloc[[idx]]
                break
                
        if len(query_books) == 0:
            return books_df.sample(n)  # No match found
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Get the first matching book
    query_book = query_books.iloc[0]
    
    # Create TF-IDF vectors for book content
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(processed_books['content'])
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Get the TF-IDF vector for the query book
    query_idx = query_book.name
    
    # Calculate similarity scores
    cosine_sim = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix).flatten()
    
    if progress_bar:
        progress_bar.progress(80)
    
    # Create a series of similarity scores and sort
    similar_indices = np.argsort(cosine_sim)[::-1]
    
    # Remove the query book itself and get more books than needed to allow for duplicate filtering
    similar_indices = [idx for idx in similar_indices if idx != query_idx][:n*3]
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Get similar books
    similar_books = processed_books.iloc[similar_indices]
    
    # Add similarity score
    similar_books = similar_books.copy()
    for i, idx in enumerate(similar_indices):
        if 'mean' not in similar_books.columns:
            similar_books['mean'] = 0.0
        row_idx = similar_books.index.get_loc(similar_books.index[i])
        similar_books.iloc[row_idx, similar_books.columns.get_loc('mean')] = cosine_sim[idx]
    
    if progress_bar:
        progress_bar.progress(100)
    
    return similar_books

def get_word_embedding_recommendations(book_title, books_df, n=5, progress_bar=None):
    """Recommend books using pre-trained word embeddings"""
    import numpy as np
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Preprocess book data
    processed_books = preprocess_book_data(books_df)
    
    # Find the query book
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
    if len(query_books) == 0:
        return books_df.sample(n)
    
    if progress_bar:
        progress_bar.progress(30)
    
    # Load pre-trained word vectors (would need to download these first)
    try:
        import gensim.downloader as api
        word_vectors = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe vectors
    except Exception as e:
        st.warning(f"Failed to load word vectors: {e}. Please install gensim with: pip install gensim")
        return get_content_based_recommendations(book_title, books_df, n, progress_bar)
    
    if progress_bar:
        progress_bar.progress(60)
    
    # Create document embeddings by averaging word vectors
    def get_book_vector(text):
        words = text.lower().split()
        vectors = [word_vectors[word] for word in words if word in word_vectors]
        if not vectors:
            return np.zeros(word_vectors.vector_size)
        return np.mean(vectors, axis=0)
    
    # Get embedding for query book
    query_book = query_books.iloc[0]
    query_vector = get_book_vector(query_book['content'])
    
    if progress_bar:
        progress_bar.progress(80)
    
    # Calculate similarity for all books
    book_similarities = []
    for idx, book in processed_books.iterrows():
        book_vector = get_book_vector(book['content'])
        # Compute cosine similarity
        similarity = np.dot(query_vector, book_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(book_vector)) if np.linalg.norm(query_vector) > 0 and np.linalg.norm(book_vector) > 0 else 0
        book_similarities.append((idx, similarity))
    
    # Get top similar books
    similar_indices = [idx for idx, _ in sorted(book_similarities, key=lambda x: x[1], reverse=True) if idx != query_book.name][:n]
    similar_books = processed_books.iloc[similar_indices].copy()
    
    # Add similarity scores
    for i, (idx, sim) in enumerate([(idx, sim) for idx, sim in book_similarities if idx in similar_indices]):
        similar_books.loc[idx, 'mean'] = sim
    
    return similar_books

def get_bert_recommendations(book_title, books_df, n=5, progress_bar=None):
    """Recommend books using BERT embeddings for better semantic understanding"""
    import numpy as np
    
    if progress_bar:
        progress_bar.progress(10)
    
    # Load BERT model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Smaller, efficient model
    except Exception as e:
        st.warning(f"Failed to load BERT model: {e}. Please install sentence-transformers with: pip install sentence-transformers")
        return get_content_based_recommendations(book_title, books_df, n, progress_bar)
    
    # Preprocess book data
    processed_books = preprocess_book_data(books_df)
    
    if progress_bar:
        progress_bar.progress(30)
    
    # Find the query book
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
    if len(query_books) == 0:
        return books_df.sample(n)
    
    query_book = query_books.iloc[0]
    
    if progress_bar:
        progress_bar.progress(50)
    
    # Generate embeddings for all books (can be slow for large datasets)
    # For production, consider pre-computing and storing these embeddings
    sample_size = min(1000, len(processed_books))  # Limit processing for demo purposes
    books_sample = processed_books.sample(sample_size) if len(processed_books) > sample_size else processed_books
    
    # Create embeddings
    titles = books_sample['Book-Title'].tolist()
    authors = books_sample['Book-Author'].tolist()
    book_texts = [f"{title} by {author}" for title, author in zip(titles, authors)]
    embeddings = model.encode(book_texts, show_progress_bar=False)
    
    if progress_bar:
        progress_bar.progress(70)
    
    # Generate embedding for query book
    query_text = f"{query_book['Book-Title']} by {query_book['Book-Author']}"
    query_embedding = model.encode([query_text])[0]
    
    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(embeddings):
        if books_sample.index[i] != query_book.name:
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((books_sample.index[i], similarity))
    
    if progress_bar:
        progress_bar.progress(90)
    
    # Sort by similarity and get top N
    similar_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)][:n]
    similar_books = processed_books.loc[similar_indices].copy()
    
    # Add similarity scores
    for i, (idx, sim) in enumerate([(idx, sim) for idx, sim in similarities if idx in similar_indices]):
        similar_books.loc[idx, 'mean'] = sim
    
    return similar_books

# Main app
def main():
    st.title("📚 Book Recommendation System")
    
    # Load data
    with st.spinner("Loading data... Please wait."):
        users, books, ratings = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "User Analysis", "Book Recommendations", "About"])
    
    if page == "Home":
        st.header("Welcome to the Book Recommendation System!")
        st.write("""
        This application helps you discover new books to read based on user ratings and preferences.
        Use the sidebar to navigate through different sections of the app.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"Number of Users: {users.shape[0]}")
            st.write(f"Number of Books: {books.shape[0]}")
            st.write(f"Number of Ratings: {ratings.shape[0]}")
        
        with col2:
            st.subheader("Sample Books")
            st.dataframe(books.sample(5)[['Book-Title', 'Book-Author', 'Year-Of-Publication']])
        
        st.subheader("Popular Books")
        popular_books = get_popular_books(books, ratings)
        st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'mean', 'count']])
        
        # After the Popular Books section in the Home page
        st.markdown("---")
        st.subheader("Browse All Books")
        st.write("Search and explore the complete book catalog")

        # Add search functionality
        col1, col2 = st.columns([1, 1])
        with col1:
            search_term = st.text_input("Search by book title or author", "")
        with col2:
            sort_by = st.selectbox("Sort by", ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"])

        # Filter books based on search term
        if search_term:
            filtered_books = books[
                books["Book-Title"].str.contains(search_term, case=False, na=False) |
                books["Book-Author"].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_books = books

        # Sort the filtered books
        filtered_books = filtered_books.sort_values(by=sort_by)

        # Pagination
        books_per_page = 10
        total_pages = max(1, len(filtered_books) // books_per_page + (1 if len(filtered_books) % books_per_page > 0 else 0))
        page_number = st.slider("Page", 1, total_pages, 1)

        start_idx = (page_number - 1) * books_per_page
        end_idx = min(start_idx + books_per_page, len(filtered_books))

        # Display page info
        st.write(f"Showing {start_idx+1}-{end_idx} of {len(filtered_books)} books")

        # Display the books for the current page
        current_page_books = filtered_books.iloc[start_idx:end_idx].reset_index(drop=True)

        # Display books as a table
        st.dataframe(current_page_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'ISBN']], 
                     use_container_width=True)

        # Allow downloading the full dataset as CSV
        if st.button("Download All Books as CSV"):
            csv = books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'ISBN']].to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="all_books.csv",
                mime="text/csv",
            )
    
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Books", "Users", "Ratings"])
        
        with tab1:
            st.subheader("Books Dataset")
            st.dataframe(books.head())
            
            st.subheader("Missing Values in Books Dataset")
            st.dataframe(missing_values(books))
            
            st.subheader("Books Published by Year")
            # Fixed this section to properly handle value_counts() result
            years_df = pd.DataFrame(books['Year-Of-Publication'].value_counts()).reset_index()
            # Explicitly rename columns to match what we expect
            years_df.columns = ['Year', 'Count']
            # Convert Year to numeric for proper filtering and sorting
            years_df['Year'] = pd.to_numeric(years_df['Year'], errors='coerce')
            # Sort by year and filter valid years
            years_df = years_df.sort_values('Year').reset_index(drop=True)
            valid_years = years_df[(years_df['Year'] >= 1900) & (years_df['Year'] <= 2010)]
            
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.lineplot(data=valid_years, x='Year', y='Count', ax=ax)
            plt.title('Number of Books Published by Year')
            st.pyplot(fig)
            
            # Add Top 10 Authors visualization from notebook
            st.subheader("Top 10 Authors by Number of Books")
            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            top_authors = books['Book-Author'].value_counts().head(10)
            sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax, palette='viridis')
            plt.title('Top 10 Authors')
            plt.xlabel('Number of Books')
            st.pyplot(fig)
            
            # Add Top 10 Publishers visualization from notebook
            st.subheader("Top 10 Publishers by Number of Books")
            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            top_publishers = books['Publisher'].value_counts().head(10)
            sns.barplot(x=top_publishers.values, y=top_publishers.index, ax=ax, palette='mako')
            plt.title('Top 10 Publishers')
            plt.xlabel('Number of Books')
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Users Dataset")
            st.dataframe(users.head())
            
            st.subheader("Missing Values in Users Dataset")
            st.dataframe(missing_values(users))
            
            st.subheader("Age Distribution")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100], ax=ax)
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Remove the Country section that's causing errors and add more visualizations
            
            # Age distribution - boxplot
            st.subheader("Age Distribution - Boxplot")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.boxplot(y='Age', data=users, ax=ax)
            plt.title('Age Distribution by Boxplot')
            st.pyplot(fig)
            
            # Age distribution - KDE plot
            st.subheader("Age Distribution - Density Plot")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.kdeplot(data=users['Age'].dropna(), fill=True, ax=ax)
            plt.title('Age Distribution Density Plot')
            plt.xlabel('Age')
            plt.ylabel('Density')
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Ratings Dataset")
            st.dataframe(ratings.head())
            
            st.subheader("Rating Distribution")
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            sns.countplot(x='Book-Rating', data=ratings, ax=ax)
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Add pair plot visualization
            st.subheader("Pair Plot: Age, Publication Year, and Book Rating")
            
            # Create button to generate the pair plot (since it can be computationally intensive)
            if st.button("Generate Pair Plot"):
                with st.spinner("Creating pair plot, this may take a moment..."):
                    # Merge datasets to get all required columns
                    df_merged = pd.merge(ratings, books[['ISBN', 'Year-Of-Publication']], on='ISBN')
                    df_merged = pd.merge(df_merged, users[['User-ID', 'Age']], on='User-ID')
                    
                    # Convert columns to numeric
                    df_merged['Year-Of-Publication'] = pd.to_numeric(df_merged['Year-Of-Publication'], errors='coerce')
                    df_merged['Age'] = pd.to_numeric(df_merged['Age'], errors='coerce')
                    df_merged['Book-Rating'] = pd.to_numeric(df_merged['Book-Rating'], errors='coerce')
                    
                    # Filter data to remove outliers and invalid values
                    df_merged_clean = df_merged[
                        (df_merged['Age'].between(5, 100)) & 
                        (df_merged['Year-Of-Publication'].between(1900, 2010)) &
                        (df_merged['Book-Rating'] > 0)
                    ]
                    
                    # Sample data if too large (pair plots can be computationally expensive)
                    if len(df_merged_clean) > 10000:
                        st.info(f"Sampling 10,000 datapoints from {len(df_merged_clean)} for faster rendering")
                        df_merged_clean = df_merged_clean.sample(10000, random_state=42)
                    
                    # Generate the pair plot
                    fig = plt.figure(figsize=(12, 10))
                    pair_plot = sns.pairplot(
                        df_merged_clean[['Age', 'Year-Of-Publication', 'Book-Rating']],
                        diag_kind="kde", 
                        hue='Book-Rating', 
                        palette="viridis", 
                        height=3,
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.3}
                    )
                    pair_plot.fig.subplots_adjust(top=0.95)
                    pair_plot.fig.suptitle("Pair Plot of Age, Publication Year, and Book Rating", fontsize=16)
                    
                    # Display the plot
                    st.pyplot(pair_plot.fig)
                    
                    # Add description
                    st.write("""
                    **Pair Plot Analysis:**
                    
                    This visualization shows the relationships between reader age, book publication year, and ratings.
                    The diagonal shows the distribution of each variable, while the scatter plots show relationships between pairs of variables.
                    Colors represent different rating values, helping identify patterns in how different age groups rate books of different publication years.
                    """)
    
    elif page == "User Analysis":
        st.header("User Analysis")
        
        # User selection
        user_id = st.selectbox("Select User ID", sorted(users['User-ID'].unique()))
        
        if user_id:
            # Show user info
            user_info = users[users['User-ID'] == user_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("User Information")
                st.write(f"**User ID:** {user_id}")
                st.write(f"**Age:** {user_info['Age'].values[0] if not pd.isna(user_info['Age'].values[0]) else 'Not specified'}")
                # Country column has been removed as it doesn't exist in the dataset
            
            # Show user's book ratings
            user_books = get_user_book_ratings(user_id, ratings, books)
            
            with col2:
                st.subheader("Rating Statistics")
                st.write(f"**Number of Books Rated:** {len(user_books)}")
                if len(user_books) > 0:
                    avg_rating = user_books['Book-Rating'].mean()
                    st.write(f"**Average Rating:** {avg_rating:.2f}")
                    # Add more statistics
                    median_rating = user_books['Book-Rating'].median()
                    st.write(f"**Median Rating:** {median_rating:.1f}")
                    most_common_rating = user_books['Book-Rating'].mode()[0]
                    st.write(f"**Most Common Rating:** {most_common_rating}")
            
            if len(user_books) > 0:
                # Create tabs for different visualizations
                user_tabs = st.tabs(["Books Rated", "Rating Distribution", "Rating Patterns", "Authors & Publishers"])
                
                with user_tabs[0]:
                    st.subheader("Books Rated by User")
                    user_books_display = user_books.sort_values('Book-Rating', ascending=False)[
                        ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Rating']
                    ]
                    st.dataframe(user_books_display)
                    
                    # Add download button for user's rated books
                    csv = user_books_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download User's Ratings",
                        data=csv,
                        file_name=f"user_{user_id}_ratings.csv",
                        mime="text/csv",
                    )
                
                with user_tabs[1]:
                    st.subheader("Rating Distribution")
                    col_hist, col_pie = st.columns(2)
                    
                    with col_hist:
                        # Histogram of ratings
                        fig, ax = plt.figure(figsize=(8, 4)), plt.axes()
                        sns.countplot(x='Book-Rating', data=user_books, ax=ax, palette='viridis')
                        plt.title(f'Rating Distribution for User {user_id}')
                        plt.xlabel('Rating')
                        plt.ylabel('Count')
                        st.pyplot(fig)
                    
                    with col_pie:
                        # Pie chart of ratings
                        fig, ax = plt.figure(figsize=(8, 4)), plt.axes()
                        user_books['Book-Rating'].value_counts().plot.pie(
                            autopct='%1.1f%%', 
                            ax=ax, 
                            shadow=True,
                            labels=None,  # Hide labels on the pie
                            colors=sns.color_palette('viridis', user_books['Book-Rating'].nunique())
                        )
                        plt.title('Rating Distribution (%)')
                        plt.ylabel('')  # Hide the default y-label
                        ax.legend(
                            title="Rating",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1)
                        )
                        st.pyplot(fig)
                
                with user_tabs[2]:
                    st.subheader("Rating Patterns")
                    
                    # Check if there are enough ratings with publication years
                    if user_books['Year-Of-Publication'].nunique() > 1:
                        # Convert publication year to numeric
                        user_books['Year-Of-Publication'] = pd.to_numeric(user_books['Year-Of-Publication'], errors='coerce')
                        
                        # Filter valid years
                        valid_years_books = user_books[user_books['Year-Of-Publication'].between(1900, 2010)]
                        
                        if len(valid_years_books) > 3:  # Only if we have enough data
                            # Plot ratings by publication year
                            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                            sns.scatterplot(
                                data=valid_years_books,
                                x='Year-Of-Publication',
                                y='Book-Rating',
                                alpha=0.7,
                                s=100
                            )
                            
                            # Add trend line
                            try:
                                import numpy as np
                                from scipy import stats
                                
                                # Get x and y data
                                x = valid_years_books['Year-Of-Publication']
                                y = valid_years_books['Book-Rating']
                                
                                # Calculate trend line
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                trend_line = slope * x + intercept
                                
                                # Plot trend line
                                plt.plot(x, trend_line, 'r--', linewidth=2)
                                
                                # Add correlation info
                                corr = round(r_value, 2)
                                plt.annotate(
                                    f'Correlation: {corr}',
                                    xy=(0.05, 0.95),
                                    xycoords='axes fraction',
                                    fontsize=12,
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                                )
                                
                            except Exception as e:
                                pass  # If trend line calculation fails, just show scatter plot
                            
                            plt.title(f'Ratings by Publication Year for User {user_id}')
                            plt.xlabel('Publication Year')
                            plt.ylabel('Rating')
                            st.pyplot(fig)
                            
                            if corr > 0.3:
                                st.info(f"This user seems to prefer newer books (positive correlation: {corr:.2f}).")
                            elif corr < -0.3:
                                st.info(f"This user seems to prefer older books (negative correlation: {corr:.2f}).")
                            else:
                                st.info(f"This user doesn't show a strong preference for books from any particular era (correlation: {corr:.2f}).")
                        else:
                            st.info("Not enough books with valid publication years to analyze patterns.")
                    else:
                        st.info("All rated books have the same publication year or missing values.")
                
                with user_tabs[3]:
                    st.subheader("Top Authors & Publishers")
                    
                    col_authors, col_publishers = st.columns(2)
                    
                    with col_authors:
                        # Top authors rated by user
                        author_counts = user_books['Book-Author'].value_counts()
                        if len(author_counts) > 1:
                            fig, ax = plt.figure(figsize=(8, min(10, len(author_counts)))), plt.axes()
                            
                            # Limit to top 10 authors
                            top_authors = author_counts.head(10)
                            sns.barplot(
                                x=top_authors.values,
                                y=top_authors.index,
                                palette='viridis',
                                ax=ax
                            )
                            plt.title(f'Top Authors Rated by User {user_id}')
                            plt.xlabel('Number of Books')
                            st.pyplot(fig)
                            
                            # Get average rating per author (for authors with multiple books)
                            author_ratings = user_books.groupby('Book-Author')['Book-Rating'].mean().sort_values(ascending=False)
                            author_counts = user_books.groupby('Book-Author')['Book-Rating'].count()
                            multi_book_authors = author_counts[author_counts > 1].index
                            
                            if len(multi_book_authors) > 0:
                                st.write("**Average Rating by Author** (authors with multiple books)")
                                author_data = pd.DataFrame({
                                    'Author': multi_book_authors,
                                    'Average Rating': author_ratings[multi_book_authors].values,
                                    'Books Rated': author_counts[multi_book_authors].values
                                }).sort_values('Average Rating', ascending=False)
                                st.dataframe(author_data)
                        else:
                            st.info("User has rated books from only one author.")
                    
                    with col_publishers:
                        # Top publishers rated by user
                        publisher_counts = user_books['Publisher'].value_counts()
                        if len(publisher_counts) > 1:
                            fig, ax = plt.figure(figsize=(8, min(10, len(publisher_counts)))), plt.axes()
                            
                            # Limit to top 10 publishers
                            top_publishers = publisher_counts.head(10)
                            sns.barplot(
                                x=top_publishers.values,
                                y=top_publishers.index,
                                palette='rocket',
                                ax=ax
                            )
                            plt.title(f'Top Publishers Rated by User {user_id}')
                            plt.xlabel('Number of Books')
                            st.pyplot(fig)
                        else:
                            st.info("User has rated books from only one publisher.")
            else:
                st.warning("This user has not rated any books.")
    
    elif page == "Book Recommendations":
        st.header("Book Recommendations")
        
        # Updated tab names and order
        tab1, tab2, tab3 = st.tabs(["Get Personalized Recommendations", "Find Similar Books", "View User's Top Rated Books"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Get Recommendations")
                # Create a selectbox with user IDs instead of manual input
                user_ids = sorted(users['User-ID'].unique())
                user_id = st.selectbox("Select User ID", user_ids)
                
                # Add model selection dropdown
                model_type = st.selectbox(
                    "Select Recommendation Model",
                    ["Collaborative Filtering (Original)", "KNN", "SVD", "NMF"]
                )
                
                num_recommendations = st.slider("Number of Recommendations", 1, 100, 5)
                recommend_button = st.button("Get Recommendations")
            
            if recommend_button:
                with col2:
                    st.subheader(f"Top {num_recommendations} Book Recommendations for User {user_id}")
                    
                    with st.spinner("Finding the best books for you..."):
                        # Add a progress bar for better UX during calculations
                        progress_bar = st.progress(0)
                        
                        try:
                            # Choose model based on selection
                            if model_type == "KNN":
                                recommended_books = get_knn_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            elif model_type == "SVD":
                                recommended_books = get_svd_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            elif model_type == "NMF":
                                recommended_books = get_nmf_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            else:  # Default to original collaborative filtering
                                recommended_books = get_book_recommendations_simple(user_id, ratings, books, num_recommendations, progress_bar)
                            
                            progress_bar.progress(100)
                            
                            if len(recommended_books) > 0:
                                # Display each recommended book with image
                                for i, (idx, book) in enumerate(recommended_books.iterrows()):
                                    col_img, col_info = st.columns([1, 3])
                                    
                                    with col_img:
                                        # Handle missing or invalid image URLs
                                        try:
                                            if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                                # Check if URL starts with 'http'
                                                img_url = book['Image-URL-M']
                                                if img_url.startswith('http'):
                                                    st.image(img_url, width=150)
                                                else:
                                                    # For local or relative paths
                                                    st.write("📚 Image path invalid")
                                            else:
                                                st.write("📚 No image provided")
                                        except Exception as e:
                                            st.write("📚 Image unavailable")
                                            # Uncomment for debugging: st.write(f"Error: {str(e)}")
                                    
                                    with col_info:
                                        st.subheader(f"{i+1}. {book['Book-Title']}")
                                        st.write(f"**Author:** {book['Book-Author']}")
                                        st.write(f"**Published:** {book['Year-Of-Publication']}")
                                        st.write(f"**Publisher:** {book['Publisher']}")
                                        
                                        # Handle different score column names
                                        if 'score' in book:
                                            st.write(f"**Recommendation Score:** {book['score']:.2f}")
                                            st.write(f"**Based on:** {int(book['supporting_users'])} similar users")
                                        elif 'mean' in book:
                                            st.write(f"**Recommendation Score:** {book['mean']:.2f}")
                                            if 'count' in book:
                                                st.write(f"**Based on:** {int(book['count'])} ratings")
                            else:
                                st.warning("Not enough data to make recommendations for this user. Showing popular books instead.")
                                popular_books = get_popular_books(books, ratings, num_recommendations)
                                st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'mean', 'count']])
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.info("Showing popular books instead.")
                            popular_books = get_popular_books(books, ratings, num_recommendations)
                            
                            # Display popular books in a more visual way
                            for i, (idx, book) in enumerate(popular_books.iterrows()):
                                col_img, col_info = st.columns([1, 3])
                                
                                with col_img:
                                    try:
                                        if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                            st.image(book['Image-URL-M'], use_container_width=True)
                                        else:
                                            st.write("📚 No image available")
                                    except Exception:
                                        st.write("📚 Image unavailable")
                                
                                with col_info:
                                    st.subheader(f"{i+1}. {book['Book-Title']}")
                                    st.write(f"**Author:** {book['Book-Author']}")
                                    st.write(f"**Published:** {book['Year-Of-Publication']}")
                                    st.write(f"**Publisher:** {book['Publisher']}")
                                    st.write(f"**Average Rating:** {book['mean']:.2f}")
                                    st.write(f"**Based on:** {int(book['count'])} ratings")
        
        # CONTENT-BASED RECOMMENDATIONS TAB with fixes for duplicates and added dropdown
        with tab2:
            st.subheader("Find Similar Books")
            st.write("Enter or select a book title to find books with similar content")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Remove Topic Modeling from the options
                method = st.selectbox(
                    "Recommendation Method",
                    ["TF-IDF (Default)", "Word Embeddings", "BERT Embeddings"],
                    help="TF-IDF: Fast text similarity. Word Embeddings: Better semantic understanding. BERT: Advanced NLP understanding."
                )
                
                # Create a list of unique book titles for dropdown
                unique_titles = sorted(books['Book-Title'].dropna().unique())
                
                # Add search box and dropdown options
                search_option = st.radio("Search method", ["Type book title", "Select from dropdown"])
                
                if search_option == "Type book title":
                    book_query = st.text_input("Book Title", "Harry Potter")
                else:
                    # Add a search box above dropdown to help find books
                    title_search = st.text_input("Search for a book", "")
                    filtered_titles = [title for title in unique_titles if title_search.lower() in title.lower()]
                    # Limit to first 1000 to avoid UI performance issues
                    if len(filtered_titles) > 1000:
                        st.info(f"Showing first 1000 of {len(filtered_titles)} matching titles. Please refine your search.")
                        filtered_titles = filtered_titles[:1000]
                    book_query = st.selectbox("Select a book", filtered_titles)
                    
                num_similar_books = st.slider("Number of Similar Books", 1, 100, 5, key="similar_books_slider")
                find_button = st.button("Find Similar Books")
            
            if find_button:
                with col2:
                    st.subheader(f"Books similar to '{book_query}'")
                    
                    with st.spinner("Finding similar books..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        
                        try:
                            # Remove the Topic Modeling option from this selector
                            if method == "Word Embeddings":
                                similar_books = get_word_embedding_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            elif method == "BERT Embeddings":
                                similar_books = get_bert_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            else:  # Default to TF-IDF
                                similar_books = get_content_based_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            
                            # Rest of your existing code for displaying similar books
                            if len(similar_books) > 0:
                                # Remove duplicates by keeping only the first occurrence of each book title
                                seen_titles = set()
                                unique_books_indices = []
                                
                                for idx, row in similar_books.iterrows():
                                    title = row['Book-Title']
                                    if title not in seen_titles:
                                        seen_titles.add(title)
                                        unique_books_indices.append(idx)
                                    
                                    # Stop once we have enough unique books
                                    if len(unique_books_indices) >= num_similar_books:
                                        break
                                
                                # Get unique books using the saved indices
                                unique_similar_books = similar_books.loc[unique_books_indices]
                                
                                # Display each similar book
                                for i, (idx, book) in enumerate(unique_similar_books.iterrows(), 1):
                                    col_img, col_info = st.columns([1, 3])
                                    
                                    with col_img:
                                        try:
                                            if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                                # Check if URL starts with 'http'
                                                img_url = book['Image-URL-M']
                                                if img_url.startswith('http'):
                                                    st.image(img_url, width=150)
                                                else:
                                                    # For local or relative paths
                                                    st.write("📚 Image path invalid")
                                            else:
                                                st.write("📚 No image provided")
                                        except Exception as e:
                                            st.write("📚 Image unavailable")
                                    
                                    with col_info:
                                        st.subheader(f"{i}. {book['Book-Title']}")
                                        st.write(f"**Author:** {book['Book-Author']}")
                                        st.write(f"**Published:** {book['Year-Of-Publication']}")
                                        st.write(f"**Publisher:** {book['Publisher']}")
                                        if 'mean' in book:
                                            st.write(f"**Similarity Score:** {book['mean']:.2f}")
                            else:
                                st.warning(f"No books similar to '{book_query}' found. Try another title.")
                        except Exception as e:
                            st.error(f"An error occurred while finding similar books: {str(e)}")
                            st.exception(e)  # This will show the full traceback for debugging
        with tab3:
            st.subheader("User's Top Rated Books")
            # Create a selectbox with user IDs
            user_ids = sorted(users['User-ID'].unique())
            selected_user_id = st.selectbox("Select User ID", user_ids, key="top_rated_user_select")
            max_books = st.slider("Maximum Number of Books to Show", 1, 100, 10)
            
            if st.button("Show Top Rated Books"):
                # Get user's book ratings
                user_books = get_user_book_ratings(selected_user_id, ratings, books)
                
                if len(user_books) > 0:
                    st.write(f"User {selected_user_id} has rated {len(user_books)} books.")
                    
                    # Sort by rating and display top N
                    top_rated = user_books.sort_values('Book-Rating', ascending=False).head(max_books)
                    
                    # Display each book
                    for i, (idx, book) in enumerate(top_rated.iterrows()):
                        col_img, col_info = st.columns([1, 3])
                        
                        with col_img:
                            try:
                                if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                    # Check if URL starts with 'http'
                                    img_url = book['Image-URL-M']
                                    if img_url.startswith('http'):
                                        st.image(img_url, width=150)
                                    else:
                                        # For local or relative paths
                                        st.write("📚 Image path invalid")
                                else:
                                    st.write("📚 No image provided")
                            except Exception as e:
                                st.write("📚 Image unavailable")
                        
                        with col_info:
                            st.subheader(f"{i+1}. {book['Book-Title']}")
                            st.write(f"**Author:** {book['Book-Author']}")
                            st.write(f"**Published:** {book['Year-Of-Publication']}")
                            st.write(f"**User's Rating:** {int(book['Book-Rating'])} / 10")
                else:
                    st.warning(f"User {selected_user_id} has not rated any books.")
    
    elif page == "About":
        st.header("About This Project")
        st.write("""
        ## Book Recommendation System
        
        This book recommendation system was developed as part of ExcelR's data science project. The system analyzes user ratings 
        to recommend books that users might enjoy based on their previous preferences and similar users' ratings.
        
        ### Dataset
        
        The dataset includes:
        - Books information (title, author, year, publisher)
        - User demographics (age, country)
        - User ratings for books
        
        ### Methods
        
        The recommendation system uses multiple approaches to find books that might interest users:
        
        1. **Collaborative Filtering** - Finding books based on similar users' preferences
        2. **K-Nearest Neighbors (KNN)** - Using nearest neighbor algorithms to find similar users
        3. **Singular Value Decomposition (SVD)** - Matrix factorization technique to identify latent factors
        4. **Non-negative Matrix Factorization (NMF)** - Alternative factorization approach with non-negative constraints
        5. **Content-Based Filtering** - Recommending books similar to a specific book title by analyzing text data
        
        ### Features
        
        - Data exploration and visualization
        - User analysis
        - Book recommendations based on user preferences using multiple algorithms
        - Content-based book recommendations to find books similar to a specific title
        - Popular book recommendations
        
        ### Developer
        
        - **Developers:** Soheb, Shafeeq, Arthi, and Navya
        - **GitHub:** [sohebkh13/ExcelR_Book_Recommendation_Project](https://github.com/sohebkh13/ExcelR_Book_Recommendation_Project)
        """)

if __name__ == "__main__":
    main()