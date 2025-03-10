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
@st.cache_data  # Cache the function to avoid reloading data on each interaction
def load_data():
    # List of encodings to try when reading CSV files
    encodings = ["cp1252"]  # cp1252 is a common Windows encoding for Western European languages
    
    for encoding in encodings:  # Try each encoding in the list
        try:
            # Attempt to load the datasets with the current encoding
            users = pd.read_csv("P505/Users_clean.csv", encoding=encoding)  # Load users data
            books = pd.read_csv("P505/Books_clean.csv", encoding=encoding)  # Load books data
            ratings = pd.read_csv("P505/Ratings_clean.csv", encoding=encoding)  # Load ratings data
            
            #st.success(f"Successfully loaded clean datasets with {encoding} encoding")  # Success message (commented out)
            return users, books, ratings  # Return the loaded dataframes
            
        except UnicodeDecodeError:  # Handle encoding errors
            # If this encoding doesn't work, try the next one
            continue
        except FileNotFoundError as e:  # Handle missing file errors
            # Show error message if files aren't found
            st.error(f"Could not find clean datasets. Error: {str(e)}")  # Display error with details
            st.warning("Please make sure P505/Users_clean.csv, P505/Books_clean.csv, and P505/Ratings_clean.csv exist.")  # Show path guidance
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Return empty dataframes
    
    # If we've tried all encodings and none worked
    st.error("Could not load datasets with any of the attempted encodings.")  # Display encoding failure message
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Return empty dataframes as fallback

# Function to display missing values
def missing_values(df):
    mis_val = df.isnull().sum()  # Count missing values for each column
    mis_val_percent = round(df.isnull().mean().mul(100), 2)  # Calculate percentage of missing values
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)  # Combine count and percentage into one table
    mz_table = mz_table.rename(
        columns={df.index.name:'col_name', 0:'Missing Values', 1:'% of Total Values'})  # Rename columns for clarity
    mz_table['Data_type'] = df.dtypes  # Add data type information to the table
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)  # Sort by percentage of missing values
    return mz_table.reset_index()  # Reset index and return the formatted table

# Create recommendation system functions
def get_user_book_ratings(user_id, ratings_df, books_df):
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]  # Filter ratings for the specific user
    user_books = pd.merge(user_ratings, books_df, on='ISBN')  # Merge with books data to get full book details
    return user_books  # Return the user's rated books with complete information

def get_popular_books(books_df, ratings_df, n=10):
    # Get books with highest average rating and minimum number of ratings
    book_ratings = ratings_df.groupby('ISBN')['Book-Rating'].agg(['mean', 'count']).reset_index()  # Calculate average rating and count per book
    # Filter books with at least 10 ratings
    popular_books = book_ratings[book_ratings['count'] >= 10].sort_values('mean', ascending=False)  # Keep only books with 10+ ratings, sort by average rating
    # Get book details
    popular_books = pd.merge(popular_books, books_df, on='ISBN')  # Merge with books dataframe to get book details
    return popular_books.head(n)  # Return top n popular books

# Original collaborative filtering function
def get_book_recommendations(user_id, ratings_df, books_df, n=5):
    """
    Generate personalized book recommendations using a more sophisticated collaborative filtering approach
    """
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set numpy random seed for reproducibility
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)  # Retrieve books rated by this user
    
    if len(user_books) == 0:  # If the user hasn't rated any books
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    # Calculate user similarity based on common book ratings
    user_ratings = ratings_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating', fill_value=0)  # Create user-book matrix
    
    # Get the target user's ratings vector
    target_user_ratings = user_ratings.loc[user_id] if user_id in user_ratings.index else pd.Series(0, index=user_ratings.columns)  # Get user's ratings or empty vector
    
    # Find similar users
    user_similarities = []  # Initialize list to store user similarities
    for other_user in user_ratings.index:  # Iterate through all users
        if other_user != user_id:  # Skip the target user
            # Calculate cosine similarity between rating vectors
            other_user_ratings = user_ratings.loc[other_user]  # Get other user's ratings vector
            
            # Get common books (non-zero ratings)
            common_books = target_user_ratings.multiply(other_user_ratings) != 0  # Find books rated by both users
            common_count = common_books.sum()  # Count common books
            
            # Only consider users with at least one common book
            if common_count > 0:  # Only proceed if there are common books
                # Calculate similarity score
                similarity = np.dot(target_user_ratings, other_user_ratings) / (
                    np.sqrt(np.dot(target_user_ratings, target_user_ratings)) * 
                    np.sqrt(np.dot(other_user_ratings, other_user_ratings))
                )  # Compute cosine similarity between users
                if not np.isnan(similarity):  # Only keep valid similarity scores
                    user_similarities.append((other_user, similarity, common_count))  # Store user ID, similarity score, and count of common books
    
    # Sort similar users by similarity score
    user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by similarity score (desc) then by common count (desc)
    
    # Take top 20 similar users
    similar_users = [user for user, _, _ in user_similarities[:20]]  # Extract user IDs of top 20 most similar users
    
    # Get books rated by similar users but not by the target user
    user_rated_isbns = set(user_books['ISBN'])  # Create set of books already rated by the target user
    recommendations = {}  # Initialize recommendations dictionary
    
    # Find books rated highly by similar users
    for idx, similar_user in enumerate(similar_users):  # Iterate through similar users
        # Get books rated by this similar user
        similar_user_books = ratings_df[ratings_df['User-ID'] == similar_user]  # Get all ratings from this similar user
        
        # Weight factor decreases as we go down the similarity list
        weight = 1.0 / (idx + 1.5)  # Diminishing weight for less similar users
        
        # Add weighted rating to recommendations
        for _, row in similar_user_books.iterrows():  # For each book rated by the similar user
            isbn = row['ISBN']  # Get the book's ISBN
            rating = row['Book-Rating']  # Get the rating given by the similar user
            
            # Skip books the target user has already rated
            if isbn in user_rated_isbns:  # Don't recommend books the user has already rated
                continue
                
            # Skip zero ratings
            if rating == 0:  # Ignore books with zero ratings
                continue
                
            # Add weighted rating to recommendations
            if isbn not in recommendations:  # If first time seeing this book
                recommendations[isbn] = {'weighted_sum': 0, 'count': 0}  # Initialize tracking for this book
            
            recommendations[isbn]['weighted_sum'] += rating * weight  # Add weighted rating to sum
            recommendations[isbn]['count'] += 1  # Increment count of recommendations
    
    # Calculate final scores
    for isbn in recommendations:  # For each recommended book
        recommendations[isbn]['score'] = recommendations[isbn]['weighted_sum'] / recommendations[isbn]['count']  # Calculate average weighted score
    
    # Sort by score
    sorted_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: (x[1]['score'], x[1]['count']), 
                                   reverse=True)  # Sort by score (desc) then by count (desc)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in sorted_recommendations[:n*2]]  # Get more than needed to ensure we have enough after merge
    
    # Get book details
    if top_isbns:  # If we have recommendations
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()  # Get details for recommended books
        
        # Add the score information
        for isbn in top_isbns:  # For each recommended book
            if isbn in recommendations:  # If we have a score for this book
                mask = recommended_books['ISBN'] == isbn  # Create mask to locate this book
                recommended_books.loc[mask, 'mean'] = recommendations[isbn]['score']  # Add score
                recommended_books.loc[mask, 'count'] = recommendations[isbn]['count']  # Add count
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)  # Sort by score and take top N
        return recommended_books  # Return the recommendations
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books

# Simplified recommendation function
def get_book_recommendations_simple(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate personalized book recommendations using a simplified approach
    that's more efficient for a large dataset
    """
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set numpy random seed for reproducibility
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)  # Retrieve books rated by this user
    user_rated_isbns = set(user_books['ISBN'])  # Create set of books already rated by the target user
    
    if len(user_books) == 0:  # If the user hasn't rated any books
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress bar to 10%
    
    # Find users who rated at least one book that the target user has also rated
    common_users = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()  # Find users who rated any of the same books
    common_users = [u for u in common_users if u != user_id]  # Exclude the target user
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress bar to 30%
    
    if len(common_users) == 0:  # If no users have rated the same books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    # Limit to a reasonable number of similar users for performance
    if len(common_users) > 100:  # If there are too many common users
        common_users = common_users[:100]  # Take only the first 100
    
    # Get all ratings from these users
    similar_users_ratings = ratings_df[ratings_df['User-ID'].isin(common_users)]  # Get all ratings from potentially similar users
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress bar to 50%
    
    # Filter out books the user has already rated
    candidate_books = similar_users_ratings[~similar_users_ratings['ISBN'].isin(user_rated_isbns)]  # Only keep books the user hasn't rated
    
    # Group by ISBN and calculate average rating and count
    book_stats = candidate_books.groupby('ISBN').agg(
        score=('Book-Rating', 'mean'),  # Average rating
        supporting_users=('User-ID', 'nunique')  # Number of unique users who rated this book
    ).reset_index()  # Reset index to have ISBN as a column
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress bar to 70%
    
    # Sort by score and number of supporting users
    book_stats = book_stats.sort_values(['score', 'supporting_users'], ascending=False)  # Sort by score (desc) then by support count (desc)
    
    # Get top N books
    top_isbns = book_stats.head(n*2)['ISBN'].tolist()  # Get more than needed to ensure we have enough after merge
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress bar to 90%
    
    # Get book details
    if top_isbns:  # If we have recommendations
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()  # Get details for recommended books
        # Merge with scores
        recommended_books = pd.merge(recommended_books, book_stats, on='ISBN')  # Join with score and support info
        # Sort and get top N
        recommended_books = recommended_books.sort_values(['score', 'supporting_users'], ascending=False).head(n)  # Sort and take top N
        return recommended_books  # Return the recommendations
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books

# NEW FUNCTIONS FOR KNN, SVD AND NMF

# KNN Recommendation Function
@st.cache_data  # Cache this function to avoid recomputing results on each run
def prepare_surprise_data(ratings_df):
    """Prepare data for Surprise library models"""
    # Load the data into the Surprise format
    reader = Reader(rating_scale=(0, 10))  # Create a Reader object with rating scale 0-10
    ratings_data = Dataset.load_from_df(
        ratings_df[['User-ID', 'ISBN', 'Book-Rating']], 
        reader
    )  # Convert pandas DataFrame to Surprise Dataset format
    # Create the training set
    trainset = ratings_data.build_full_trainset()  # Build the full training set from the data
    return trainset  # Return the prepared training set

@st.cache_data  # Cache model training to avoid retraining on each interaction
def train_knn_model(ratings_df, k=20, sim_options=None, sample_size=10000):
    # Subsample if dataset is too large
    if len(ratings_df) > sample_size:  # Check if dataset is larger than sample_size
        ratings_sample = ratings_df.sample(sample_size, random_state=42)  # Take a random sample for performance
    else:
        ratings_sample = ratings_df  # Use the complete dataset if it's small enough
        
    trainset = prepare_surprise_data(ratings_sample)  # Prepare data for Surprise library
    
    # Build the KNN model
    algo = KNNBasic(k=k, sim_options=sim_options)  # Initialize KNN model with parameters
    algo.fit(trainset)  # Train the model on the data
    
    return algo  # Return the trained model

def get_knn_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate book recommendations using a memory-efficient KNN collaborative filtering approach
    """
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set numpy random seed for consistent results
    # Get user's rated books
    user_books = ratings_df[ratings_df['User-ID'] == user_id]  # Filter ratings for the target user
    
    if len(user_books) == 0:  # Check if user has rated any books
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress to 10%
    
    # Get books rated by the user and their ratings
    user_rated_isbns = set(user_books['ISBN'].unique())  # Get unique ISBNs rated by user
    user_ratings_dict = dict(zip(user_books['ISBN'], user_books['Book-Rating']))  # Create ISBN-to-rating dictionary
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(20)  # Update progress to 20%
    
    # Find users who rated at least one book that the target user has also rated
    users_who_rated_same_books = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()  # Get users who rated same books
    similar_users = [u for u in users_who_rated_same_books if u != user_id]  # Exclude target user
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress to 30%
    
    # If too many similar users, sample for performance
    if len(similar_users) > 200:  # Check if there are more than 200 similar users
        similar_users = random.sample(similar_users, 200)  # Take a random sample of 200 users
    
    # Calculate similarity scores for each similar user
    user_similarities = []  # Initialize list to store user similarities
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(40)  # Update progress to 40%
    
    # Get ratings from similar users for books that the target user has rated
    similar_users_ratings = ratings_df[(ratings_df['User-ID'].isin(similar_users)) & 
                                      (ratings_df['ISBN'].isin(user_rated_isbns))]  # Filter ratings for similar users and common books
    
    # Create a dictionary of ratings for each user
    similar_user_ratings = {}  # Initialize dictionary for user ratings
    for _, row in similar_users_ratings.iterrows():  # Iterate through filtered ratings
        if row['User-ID'] not in similar_user_ratings:  # If first time seeing this user
            similar_user_ratings[row['User-ID']] = {}  # Initialize nested dictionary
        similar_user_ratings[row['User-ID']][row['ISBN']] = row['Book-Rating']  # Store rating for this book
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress to 50%
    
    # Calculate similarities - only for users who rated at least 2 books in common
    for sim_user, ratings in similar_user_ratings.items():  # Iterate through each similar user
        # Find books rated by both users
        common_books = set(ratings.keys()).intersection(user_ratings_dict.keys())  # Get books rated by both users
        
        if len(common_books) < 2:  # Skip if fewer than 2 books in common
            continue
            
        # Extract ratings for common books
        user_ratings_array = np.array([user_ratings_dict[isbn] for isbn in common_books])  # Get target user's ratings
        sim_user_ratings_array = np.array([ratings[isbn] for isbn in common_books])  # Get similar user's ratings
        
        # Calculate cosine similarity if we have enough common books
        if len(common_books) > 0:  # Double-check we have common books
            # Avoid division by zero
            user_magnitude = np.sqrt(np.sum(user_ratings_array**2))  # Calculate magnitude of target user's ratings
            sim_user_magnitude = np.sqrt(np.sum(sim_user_ratings_array**2))  # Calculate magnitude of similar user's ratings
            
            if user_magnitude > 0 and sim_user_magnitude > 0:  # Prevent division by zero
                similarity = np.dot(user_ratings_array, sim_user_ratings_array) / (user_magnitude * sim_user_magnitude)  # Calculate cosine similarity
                user_similarities.append((sim_user, similarity, len(common_books)))  # Store user ID, similarity, and common book count
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(60)  # Update progress to 60%
    
    # Sort similar users by similarity score
    user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by similarity (desc) then by common count (desc)
    
    # Take top k most similar users
    top_similar_users = [user for user, _, _ in user_similarities[:20]]  # Get top 20 most similar users
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress to 70%
    
    # Get books rated by similar users but not by the target user
    books_rated_by_similar_users = ratings_df[(ratings_df['User-ID'].isin(top_similar_users)) & 
                                              (~ratings_df['ISBN'].isin(user_rated_isbns)) &
                                              (ratings_df['Book-Rating'] > 0)]  # Filter for new books with positive ratings
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(80)  # Update progress to 80%
    
    # If no similar users found or no recommendations possible
    if len(books_rated_by_similar_users) == 0:  # Check if we found any new books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    # Calculate weighted ratings for recommendations
    book_scores = {}  # Initialize dictionary to store book scores
    
    for sim_user, sim_score, _ in user_similarities:  # Iterate through similar users
        if sim_user in top_similar_users:  # Only consider top similar users
            # Get books rated by this similar user
            user_books = books_rated_by_similar_users[books_rated_by_similar_users['User-ID'] == sim_user]  # Filter books for this user
            
            # Weight factor based on similarity
            weight = sim_score  # Use similarity score as weight
            
            # Add weighted rating to recommendations
            for _, row in user_books.iterrows():  # Iterate through each book
                isbn = row['ISBN']  # Get book ISBN
                rating = row['Book-Rating']  # Get rating value
                
                if isbn not in book_scores:  # If first rating for this book
                    book_scores[isbn] = {'weighted_sum': 0, 'count': 0}  # Initialize tracking
                
                book_scores[isbn]['weighted_sum'] += rating * weight  # Add weighted rating to sum
                book_scores[isbn]['count'] += 1  # Increment count
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress to 90%
    
    # Calculate final scores and sort
    for isbn in book_scores:  # Iterate through each book
        book_scores[isbn]['score'] = book_scores[isbn]['weighted_sum'] / book_scores[isbn]['count']  # Calculate average weighted score
    
    # Sort by score
    sorted_recommendations = sorted(book_scores.items(), 
                                   key=lambda x: (x[1]['score'], x[1]['count']), 
                                   reverse=True)  # Sort by score (desc) then by count (desc)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in sorted_recommendations[:n*2]]  # Get more than needed to ensure sufficient after merging
    
    # Get book details
    if top_isbns:  # If we have recommendations
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()  # Get book details
        
        # Add the score information
        for isbn, data in book_scores.items():  # Iterate through book scores
            if isbn in top_isbns:  # If this book is in our top recommendations
                mask = recommended_books['ISBN'] == isbn  # Create mask to find this book
                recommended_books.loc[mask, 'mean'] = data['score']  # Add score to dataframe
                recommended_books.loc[mask, 'count'] = data['count']  # Add count to dataframe
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)  # Sort and take top N
        return recommended_books  # Return the recommendations
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books

# SVD Recommendation Function
@st.cache_data  # Cache model to avoid retraining
def train_svd_model(ratings_df, n_factors=50):
    """Train an SVD model using Surprise library"""
    trainset = prepare_surprise_data(ratings_df)  # Prepare data for Surprise library
    
    # Build the SVD model
    algo = SVD(n_factors=n_factors)  # Initialize SVD model with specified number of latent factors
    algo.fit(trainset)  # Train the model
    
    return algo  # Return trained model

def get_svd_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """Generate book recommendations using SVD collaborative filtering"""
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set numpy random seed for consistent results
    # Get user's rated books
    user_books = get_user_book_ratings(user_id, ratings_df, books_df)  # Get user's ratings
    
    if len(user_books) == 0:  # Check if user has rated any books
        # If no ratings, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(20)  # Update progress to 20%
    
    # Create and train the SVD model
    svd_model = train_svd_model(ratings_df)  # Train or retrieve cached SVD model
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress to 50%
    
    # Get all books the user hasn't rated
    user_rated_isbns = set(user_books['ISBN'])  # Get ISBNs user has rated
    all_isbns = set(books_df['ISBN'])  # Get all ISBNs in dataset
    unrated_isbns = all_isbns - user_rated_isbns  # Find ISBNs user hasn't rated
    
    # Sample if too many (for performance)
    if len(unrated_isbns) > 1000:  # Check if there are more than 1000 unrated books
        unrated_isbns = set(random.sample(list(unrated_isbns), 1000))  # Sample 1000 random books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress to 70%
    
    # Get predictions for all unrated books
    trainset = prepare_surprise_data(ratings_df)  # Get trainset for mapping IDs
    predictions = []  # Initialize list for predictions
    
    for isbn in unrated_isbns:  # Iterate through each unrated book
        try:
            uid = trainset.to_inner_uid(user_id)  # Convert user ID to internal format
            iid = trainset.to_inner_iid(isbn)  # Convert ISBN to internal format
            # This works if both user and item are in the trainset
            pred = svd_model.estimate(uid, iid)  # Get predicted rating
            predictions.append((isbn, pred))  # Store ISBN and prediction
        except ValueError:
            # If user or item was not in the trainset
            continue  # Skip and continue to next ISBN
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress to 90%
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted rating (desc)
    
    # Get top N books
    top_isbns = [isbn for isbn, _ in predictions[:n*2]]  # Get more ISBNs than needed to ensure enough after filtering
    
    # Get book details
    if top_isbns:  # If we have recommendations
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()  # Get book details
        
        # Add scores to the recommendations
        for isbn, score in predictions:  # Iterate through predictions
            if isbn in top_isbns:  # If this ISBN is in top recommendations
                mask = recommended_books['ISBN'] == isbn  # Create mask to find this book
                recommended_books.loc[mask, 'mean'] = score  # Add score to dataframe
                recommended_books.loc[mask, 'count'] = 1  # Set count to 1 for SVD predictions
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('mean', ascending=False).head(n)  # Sort and take top N
        return recommended_books  # Return the recommendations
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books

# NMF Recommendation Function
@st.cache_data  # Cache model to avoid retraining
def train_nmf_model(ratings_df, n_components=50):
    """Train an NMF model using scikit-learn"""
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='User-ID', 
        columns='ISBN', 
        values='Book-Rating', 
        fill_value=0
    )  # Convert ratings to user-item matrix format
    
    # Map original user IDs to matrix indices
    user_map = {user_id: i for i, user_id in enumerate(user_item_matrix.index)}  # Map user IDs to row indices
    item_map = {isbn: i for i, isbn in enumerate(user_item_matrix.columns)}  # Map ISBNs to column indices
    reverse_user_map = {i: user_id for user_id, i in user_map.items()}  # Create reverse mapping for users
    reverse_item_map = {i: isbn for isbn, i in item_map.items()}  # Create reverse mapping for books
    
    # Convert matrix to numpy array
    matrix = user_item_matrix.values  # Extract numpy array from dataframe
    
    # Train NMF model
    model = NMF(n_components=n_components, init='random', random_state=0)  # Initialize NMF model
    user_features = model.fit_transform(matrix)  # Train model and get user features
    item_features = model.components_  # Get item features
    
    return model, user_features, item_features, user_map, item_map, reverse_user_map, reverse_item_map  # Return model and mappings

def get_nmf_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
    """
    Generate recommendations using NMF with memory optimization
    """
    # Set fixed seed for consistent results
    random.seed(42)  # Set random seed for reproducibility
    np.random.seed(42)  # Set numpy random seed for consistent results
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress to 10%
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]  # Filter ratings for target user
    if len(user_ratings) == 0:  # Check if user has any ratings
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(20)  # Update progress to 20%
    
    # Subsample data if too large (memory optimization)
    MAX_USERS = 5000  # Maximum number of users to include
    MAX_ITEMS = 5000  # Maximum number of books to include
    
    # Find most active users and most rated books to create a denser submatrix
    user_counts = ratings_df['User-ID'].value_counts().head(MAX_USERS).index  # Get most active users
    book_counts = ratings_df['ISBN'].value_counts().head(MAX_ITEMS).index  # Get most rated books
    
    # Make sure target user is included
    if user_id not in user_counts:  # Check if target user is in most active users
        user_counts = np.append(user_counts, user_id)  # Add target user to list
    
    # Filter ratings to include only selected users and books
    subset_ratings = ratings_df[
        (ratings_df['User-ID'].isin(user_counts)) & 
        (ratings_df['ISBN'].isin(book_counts))
    ]  # Create subset of ratings
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress to 30%
    
    # Create user and book indices
    user_ids = subset_ratings['User-ID'].unique()  # Get unique users in subset
    book_ids = subset_ratings['ISBN'].unique()  # Get unique books in subset
    
    user_to_idx = {user: i for i, user in enumerate(user_ids)}  # Map user IDs to indices
    book_to_idx = {book: i for i, book in enumerate(book_ids)}  # Map ISBNs to indices
    
    # Get user index
    user_idx = user_to_idx.get(user_id)  # Get index for target user
    if user_idx is None:  # Check if user index exists
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(40)  # Update progress to 40%
    
    # Create sparse rating matrix
    from scipy.sparse import csr_matrix  # Import for sparse matrix
    
    # Create sparse matrix
    rows = [user_to_idx[user] for user in subset_ratings['User-ID']]  # Get row indices
    cols = [book_to_idx[book] for book in subset_ratings['ISBN']]  # Get column indices
    data = subset_ratings['Book-Rating'].values  # Get rating values
    
    ratings_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(book_ids)))  # Create sparse matrix
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress to 50%
    
    # Train NMF with fewer components to save memory
    from sklearn.decomposition import NMF  # Import NMF
    
    # Use fewer components (factors) to save memory
    model = NMF(
        n_components=15,  # Fewer components for memory efficiency
        init='random',  # Random initialization
        random_state=42,  # Fixed random seed
        max_iter=50  # Fewer iterations for speed
    )  # Initialize NMF model
    
    # Train model with sparse data
    user_factors = model.fit_transform(ratings_matrix)  # Get user latent factors
    item_factors = model.components_  # Get item latent factors
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress to 70%
    
    # Get items not rated by the user
    user_rated_books = user_ratings['ISBN'].values  # Get books rated by user
    user_rated_indices = [book_to_idx[isbn] for isbn in user_rated_books if isbn in book_to_idx]  # Convert to indices
    
    # Generate predictions for unrated items
    user_vector = user_factors[user_idx]  # Get user's latent factor vector
    predictions = []  # Initialize list for predictions
    
    # Process in batches to save memory
    BATCH_SIZE = 1000  # Size of batch for memory efficiency
    for i in range(0, len(book_ids), BATCH_SIZE):  # Process books in batches
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(book_ids))))  # Get indices for current batch
        batch_predictions = np.dot(user_vector, item_factors[:, batch_indices])  # Calculate predictions for batch
        
        for j, idx in enumerate(batch_indices):  # Iterate through batch indices
            if idx not in user_rated_indices:  # Skip books user has already rated
                isbn = book_ids[idx]  # Get ISBN for this index
                predictions.append((isbn, batch_predictions[j]))  # Store ISBN and prediction
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress to 90%
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by prediction score (desc)
    
    # Get top N recommendations
    top_isbns = [isbn for isbn, _ in predictions[:n*2]]  # Get more ISBNs than needed
    
    # Get book details
    if top_isbns:  # If we have recommendations
        recommended_books = books_df[books_df['ISBN'].isin(top_isbns)].copy()  # Get book details
        
        # Add predicted scores
        for isbn, score in predictions:  # Iterate through predictions
            if isbn in top_isbns:  # If this ISBN is in top recommendations
                mask = recommended_books['ISBN'] == isbn  # Create mask to find this book
                recommended_books.loc[mask, 'score'] = score  # Add score to dataframe
        
        # Add supporting users count (not really applicable to NMF, but for interface consistency)
        recommended_books['supporting_users'] = 0  # Add placeholder column for UI consistency
        
        # Sort and get top N
        recommended_books = recommended_books.sort_values('score', ascending=False).head(n)  # Sort and take top N
        return recommended_books  # Return the recommendations
    else:
        # If no recommendations could be generated, return popular books
        return get_popular_books(books_df, ratings_df, n)  # Fall back to popular books
    
def preprocess_book_data(books_df):
    """
    Preprocess book data for content-based filtering
    """
    # Create a copy of the dataframe to avoid modifying the original
    books_df_copy = books_df.copy()
    # Replace missing titles with empty strings
    books_df_copy['Book-Title'] = books_df_copy['Book-Title'].fillna('')
    # Replace missing author names with empty strings
    books_df_copy['Book-Author'] = books_df_copy['Book-Author'].fillna('')
    # Replace missing publisher names with empty strings
    books_df_copy['Publisher'] = books_df_copy['Publisher'].fillna('')
    
    # Combine title, author and publisher into a single text field for content-based analysis
    books_df_copy['content'] = (
        books_df_copy['Book-Title'] + ' ' + 
        books_df_copy['Book-Author'] + ' ' + 
        books_df_copy['Publisher']
    )
    
    return books_df_copy  # Return the processed dataframe

def get_content_based_recommendations(book_title, books_df, n=5, progress_bar=None):
    """
    Recommend books similar to the given book title based on content similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer
    from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity metric
    import numpy as np  # Import numpy for array operations
    
    # Set random seeds for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress to 10%
    
    # Clean and prepare the book data
    processed_books = preprocess_book_data(books_df)
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress to 30%
        
    # Find books that contain the query title (case-insensitive partial match)
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False, regex=False, na=False)]
    
    if len(query_books) == 0:  # If no direct matches were found
        # Try fuzzy matching as a fallback
        all_titles = processed_books['Book-Title'].str.lower()  # Convert all titles to lowercase
        title_lower = book_title.lower()  # Convert query title to lowercase
        
        # Check if query is part of any title or vice versa
        for idx, title in enumerate(all_titles):
            if str(title_lower) in str(title) or str(title) in str(title_lower):
                query_books = processed_books.iloc[[idx]]  # Get the matching book
                break
                
        if len(query_books) == 0:  # If still no match found
            return books_df.sample(n)  # Return random books as fallback
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress to 50%
    
    # Get the first matching book as our reference
    query_book = query_books.iloc[0]
    
    # Create TF-IDF vectorizer with English stop words removed
    tfidf = TfidfVectorizer(stop_words='english')
    # Transform the book content field into TF-IDF feature matrix
    tfidf_matrix = tfidf.fit_transform(processed_books['content'])
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress to 70%
    
    # Get the index of the query book in the original dataframe
    query_idx = query_book.name
    
    # Calculate cosine similarity between query book and all other books
    cosine_sim = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix).flatten()
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(80)  # Update progress to 80%
    
    # Sort indices by similarity scores in descending order
    similar_indices = np.argsort(cosine_sim)[::-1]
    
    # Remove the query book from results and get more than needed to filter duplicates later
    similar_indices = [idx for idx in similar_indices if idx != query_idx][:n*3]
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress to 90%
    
    # Get the books corresponding to the similar indices
    similar_books = processed_books.iloc[similar_indices]
    
    # Add similarity scores to the result dataframe
    similar_books = similar_books.copy()  # Create a copy to avoid SettingWithCopyWarning
    for i, idx in enumerate(similar_indices):
        if 'mean' not in similar_books.columns:  # If 'mean' column doesn't exist
            similar_books['mean'] = 0.0  # Create it with default value 0.0
        row_idx = similar_books.index.get_loc(similar_books.index[i])  # Get location of current row
        similar_books.iloc[row_idx, similar_books.columns.get_loc('mean')] = cosine_sim[idx]  # Add similarity score
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(100)  # Complete the progress bar
    
    return similar_books  # Return the similar books with their similarity scores

def get_word_embedding_recommendations(book_title, books_df, n=5, progress_bar=None):
    """Recommend books using pre-trained word embeddings"""
    import numpy as np  # Import numpy for vector operations
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress to 10%
    
    # Clean and prepare the book data
    processed_books = preprocess_book_data(books_df)
    
    # Find books that contain the query title (case-insensitive)
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
    if len(query_books) == 0:  # If no matches found
        return books_df.sample(n)  # Return random sample as fallback
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress to 30%
    
    # Attempt to load pre-trained word embeddings from gensim
    try:
        import gensim.downloader as api  # Import gensim downloader API
        word_vectors = api.load("glove-wiki-gigaword-100")  # Load 100-dimensional GloVe vectors
    except Exception as e:  # If loading fails
        st.warning(f"Failed to load word vectors: {e}. Please install gensim with: pip install gensim")
        # Fall back to TF-IDF method if word vectors can't be loaded
        return get_content_based_recommendations(book_title, books_df, n, progress_bar)
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(60)  # Update progress to 60%
    
    # Define function to compute document embedding by averaging word vectors
    def get_book_vector(text):
        words = text.lower().split()  # Split text into lowercase words
        vectors = [word_vectors[word] for word in words if word in word_vectors]  # Get vector for each word
        if not vectors:  # If no words have vectors
            return np.zeros(word_vectors.vector_size)  # Return zero vector
        return np.mean(vectors, axis=0)  # Return average of all word vectors
    
    # Get the first matching book as our reference
    query_book = query_books.iloc[0]
    # Compute embedding for the query book's content
    query_vector = get_book_vector(query_book['content'])
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(80)  # Update progress to 80%
    
    # Calculate similarity between query book and all other books
    book_similarities = []
    for idx, book in processed_books.iterrows():  # Iterate through all books
        book_vector = get_book_vector(book['content'])  # Compute embedding for current book
        # Compute cosine similarity, avoiding division by zero
        similarity = np.dot(query_vector, book_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(book_vector)) if np.linalg.norm(query_vector) > 0 and np.linalg.norm(book_vector) > 0 else 0
        book_similarities.append((idx, similarity))  # Store index and similarity score
    
    # Sort by similarity (descending) and get top N books, excluding the query book
    similar_indices = [idx for idx, _ in sorted(book_similarities, key=lambda x: x[1], reverse=True) if idx != query_book.name][:n]
    similar_books = processed_books.iloc[similar_indices].copy()  # Get the similar books
    
    # Add similarity scores to the result dataframe
    for i, (idx, sim) in enumerate([(idx, sim) for idx, sim in book_similarities if idx in similar_indices]):
        similar_books.loc[idx, 'mean'] = sim  # Add similarity score under 'mean' column
    
    return similar_books  # Return the similar books with their similarity scores

def get_bert_recommendations(book_title, books_df, n=5, progress_bar=None):
    """Recommend books using BERT embeddings for better semantic understanding"""
    import numpy as np  # Import numpy for vector operations
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(10)  # Update progress to 10%
    
    # Try to load the BERT model
    try:
        from sentence_transformers import SentenceTransformer  # Import BERT model framework
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load smaller, efficient BERT model
    except Exception as e:  # If loading fails
        st.warning(f"Failed to load BERT model: {e}. Please install sentence-transformers with: pip install sentence-transformers")
        # Fall back to TF-IDF method if BERT can't be loaded
        return get_content_based_recommendations(book_title, books_df, n, progress_bar)
    
    # Clean and prepare the book data
    processed_books = preprocess_book_data(books_df)
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(30)  # Update progress to 30%
    
    # Find books that contain the query title (case-insensitive)
    query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
    if len(query_books) == 0:  # If no matches found
        return books_df.sample(n)  # Return random sample as fallback
    
    # Get the first matching book as our reference
    query_book = query_books.iloc[0]
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(50)  # Update progress to 50%
    
    # Limit the number of books to process for efficiency
    sample_size = min(1000, len(processed_books))  # Cap at 1000 books maximum
    # Sample from the books if dataset is large
    books_sample = processed_books.sample(sample_size) if len(processed_books) > sample_size else processed_books
    
    # Extract titles and authors for embedding
    titles = books_sample['Book-Title'].tolist()  # Get list of all book titles
    authors = books_sample['Book-Author'].tolist()  # Get list of all book authors
    # Combine title and author into meaningful text representations
    book_texts = [f"{title} by {author}" for title, author in zip(titles, authors)]
    # Generate embeddings for all books using BERT
    embeddings = model.encode(book_texts, show_progress_bar=False)
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(70)  # Update progress to 70%
    
    # Generate embedding for the query book
    query_text = f"{query_book['Book-Title']} by {query_book['Book-Author']}"
    query_embedding = model.encode([query_text])[0]  # Get the embedding vector
    
    # Calculate cosine similarity between query book and all other books
    similarities = []
    for i, embedding in enumerate(embeddings):  # Iterate through all embeddings
        if books_sample.index[i] != query_book.name:  # Skip the query book itself
            # Compute cosine similarity
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((books_sample.index[i], similarity))  # Store index and similarity
    
    if progress_bar:  # If progress bar is provided
        progress_bar.progress(90)  # Update progress to 90%
    
    # Sort by similarity (descending) and get top N books
    similar_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)][:n]
    # Get the books corresponding to the similar indices
    similar_books = processed_books.loc[similar_indices].copy()
    
    # Add similarity scores to the result dataframe
    for i, (idx, sim) in enumerate([(idx, sim) for idx, sim in similarities if idx in similar_indices]):
        similar_books.loc[idx, 'mean'] = sim  # Add similarity score under 'mean' column
    
    return similar_books  # Return the similar books with their similarity scores

# Main app
def main():
    # Display the app title with an emoji
    st.title("📚 Book Recommendation System")
    
    # Load data with a loading spinner
    with st.spinner("Loading data... Please wait."):
        users, books, ratings = load_data()
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    # Add radio buttons for page selection
    page = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "User Analysis", "Book Recommendations", "About"])
    
    if page == "Home":
        # Display welcome header on home page
        st.header("Welcome to the Book Recommendation System!")
        # Add descriptive text about the application
        st.write("""
        This application helps you discover new books to read based on user ratings and preferences.
        Use the sidebar to navigate through different sections of the app.
        """)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display dataset statistics in the left column
            st.subheader("Dataset Statistics")
            st.write(f"Number of Users: {users.shape[0]}")
            st.write(f"Number of Books: {books.shape[0]}")
            st.write(f"Number of Ratings: {ratings.shape[0]}")
        
        with col2:
            # Show random book samples in the right column
            st.subheader("Sample Books")
            st.dataframe(books.sample(5)[['Book-Title', 'Book-Author', 'Year-Of-Publication']])
        
        # Display popular books section
        st.subheader("Popular Books")
        # Get popular books using custom function
        popular_books = get_popular_books(books, ratings)
        # Show popular books in a dataframe
        st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'mean', 'count']])
        
        # Add horizontal separator
        st.markdown("---")
        # Add book browsing section
        st.subheader("Browse All Books")
        st.write("Search and explore the complete book catalog")

        # Create two columns for search and sort options
        col1, col2 = st.columns([1, 1])
        with col1:
            # Add text input for search functionality
            search_term = st.text_input("Search by book title or author", "")
        with col2:
            # Add dropdown to select sorting criteria
            sort_by = st.selectbox("Sort by", ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"])

        # Filter books based on search criteria
        if search_term:
            # Search across both title and author columns
            filtered_books = books[
                books["Book-Title"].str.contains(search_term, case=False, na=False) |
                books["Book-Author"].str.contains(search_term, case=False, na=False)
            ]
        else:
            # If no search term, show all books
            filtered_books = books

        # Apply sorting to filtered results
        filtered_books = filtered_books.sort_values(by=sort_by)

        # Implement pagination
        books_per_page = 10  # Number of books to display per page
        # Calculate total number of pages needed
        total_pages = max(1, len(filtered_books) // books_per_page + (1 if len(filtered_books) % books_per_page > 0 else 0))
        # Add slider to select page number
        page_number = st.slider("Page", 1, total_pages, 1)

        # Calculate start and end indices for current page
        start_idx = (page_number - 1) * books_per_page
        end_idx = min(start_idx + books_per_page, len(filtered_books))

        # Show pagination information
        st.write(f"Showing {start_idx+1}-{end_idx} of {len(filtered_books)} books")

        # Get books for current page
        current_page_books = filtered_books.iloc[start_idx:end_idx].reset_index(drop=True)

        # Display current page books in a table
        st.dataframe(current_page_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'ISBN']], 
                     use_container_width=True)

        # Add download functionality for the complete dataset
        if st.button("Download All Books as CSV"):
            # Convert dataframe to CSV
            csv = books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'ISBN']].to_csv(index=False)
            # Create download button with the CSV data
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="all_books.csv",
                mime="text/csv",
            )
    
    elif page == "Data Exploration":  # Check if the Data Exploration page is selected
        st.header("Data Exploration")  # Display the main header for this page
        
        tab1, tab2, tab3 = st.tabs(["Books", "Users", "Ratings"])  # Create three tabs for different dataset explorations
        
        with tab1:  # Start of Books tab content
            st.subheader("Books Dataset")  # Display subheader for books dataset section
            st.dataframe(books.head())  # Show the first few rows of the books dataframe
            
            st.subheader("Missing Values in Books Dataset")  # Display subheader for missing values analysis
            st.dataframe(missing_values(books))  # Show table of missing values statistics using custom function
            
            st.subheader("Books Published by Year")  # Display subheader for year analysis
            # Create a dataframe with publication year counts
            years_df = pd.DataFrame(books['Year-Of-Publication'].value_counts()).reset_index()
            # Rename columns for clarity
            years_df.columns = ['Year', 'Count']
            # Convert Year column to numeric, coercing errors to NaN
            years_df['Year'] = pd.to_numeric(years_df['Year'], errors='coerce')
            # Sort years chronologically and reset index
            years_df = years_df.sort_values('Year').reset_index(drop=True)
            # Filter to only include realistic publication years (1900-2010)
            valid_years = years_df[(years_df['Year'] >= 1900) & (years_df['Year'] <= 2010)]
            
            # Create figure and axes for the plot
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            # Draw line plot of books published per year
            sns.lineplot(data=valid_years, x='Year', y='Count', ax=ax)
            # Add title to the plot
            plt.title('Number of Books Published by Year')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
            
            st.subheader("Top 10 Authors by Number of Books")  # Display subheader for top authors analysis
            # Create figure and axes for the plot
            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            # Get the top 10 authors by book count
            top_authors = books['Book-Author'].value_counts().head(10)
            # Create horizontal bar chart of top authors
            sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax, palette='viridis')
            # Add title and x-axis label
            plt.title('Top 10 Authors')
            plt.xlabel('Number of Books')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
            
            st.subheader("Top 10 Publishers by Number of Books")  # Display subheader for top publishers
            # Create figure and axes for the plot
            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
            # Get the top 10 publishers by book count
            top_publishers = books['Publisher'].value_counts().head(10)
            # Create horizontal bar chart of top publishers
            sns.barplot(x=top_publishers.values, y=top_publishers.index, ax=ax, palette='mako')
            # Add title and x-axis label
            plt.title('Top 10 Publishers')
            plt.xlabel('Number of Books')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
        
        with tab2:  # Start of Users tab content
            st.subheader("Users Dataset")  # Display subheader for users dataset
            st.dataframe(users.head())  # Show the first few rows of the users dataframe
            
            st.subheader("Missing Values in Users Dataset")  # Display subheader for missing values analysis
            st.dataframe(missing_values(users))  # Show table of missing values using custom function
            
            st.subheader("Age Distribution")  # Display subheader for age distribution
            # Create figure and axes for the histogram
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            # Create histogram of user ages with specified bins
            users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100], ax=ax)
            # Add title and axis labels
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Count')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
            
            st.subheader("Age Distribution - Boxplot")  # Display subheader for age boxplot
            # Create figure and axes for the boxplot
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            # Create boxplot of user ages
            sns.boxplot(y='Age', data=users, ax=ax)
            # Add title to the plot
            plt.title('Age Distribution by Boxplot')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
            
            st.subheader("Age Distribution - Density Plot")  # Display subheader for age density plot
            # Create figure and axes for the KDE plot
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            # Create KDE (density) plot of user ages, dropping NaN values
            sns.kdeplot(data=users['Age'].dropna(), fill=True, ax=ax)
            # Add title and axis labels
            plt.title('Age Distribution Density Plot')
            plt.xlabel('Age')
            plt.ylabel('Density')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
        
        with tab3:  # Start of Ratings tab content
            st.subheader("Ratings Dataset")  # Display subheader for ratings dataset
            st.dataframe(ratings.head())  # Show the first few rows of the ratings dataframe
            
            st.subheader("Rating Distribution")  # Display subheader for rating distribution
            # Create figure and axes for the plot
            fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
            # Create bar chart of rating frequencies
            sns.countplot(x='Book-Rating', data=ratings, ax=ax)
            # Add title and axis labels
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            # Display the matplotlib figure in Streamlit
            st.pyplot(fig)
            
            st.subheader("Pair Plot: Age, Publication Year, and Book Rating")  # Display subheader for pair plot
            
            # Create a button to trigger the computationally intensive pair plot
            if st.button("Generate Pair Plot"):  # Check if button is clicked
                # Show spinner while plot is being generated
                with st.spinner("Creating pair plot, this may take a moment..."):
                    # Merge ratings with books to get publication year
                    df_merged = pd.merge(ratings, books[['ISBN', 'Year-Of-Publication']], on='ISBN')
                    # Merge with users to get age data
                    df_merged = pd.merge(df_merged, users[['User-ID', 'Age']], on='User-ID')
                    
                    # Convert columns to numeric format, handling errors
                    df_merged['Year-Of-Publication'] = pd.to_numeric(df_merged['Year-Of-Publication'], errors='coerce')
                    df_merged['Age'] = pd.to_numeric(df_merged['Age'], errors='coerce')
                    df_merged['Book-Rating'] = pd.to_numeric(df_merged['Book-Rating'], errors='coerce')
                    
                    # Filter data to remove outliers and invalid values
                    df_merged_clean = df_merged[
                        (df_merged['Age'].between(5, 100)) & 
                        (df_merged['Year-Of-Publication'].between(1900, 2010)) &
                        (df_merged['Book-Rating'] > 0)
                    ]
                    
                    # Sample data if there are too many rows for efficient plotting
                    if len(df_merged_clean) > 10000:
                        # Show info message about sampling
                        st.info(f"Sampling 10,000 datapoints from {len(df_merged_clean)} for faster rendering")
                        # Take random sample of 10,000 rows
                        df_merged_clean = df_merged_clean.sample(10000, random_state=42)
                    
                    # Create figure for pair plot
                    fig = plt.figure(figsize=(12, 10))
                    # Generate pair plot showing relationships between variables
                    pair_plot = sns.pairplot(
                        df_merged_clean[['Age', 'Year-Of-Publication', 'Book-Rating']],
                        diag_kind="kde",  # Show kernel density plots on diagonal
                        hue='Book-Rating',  # Color points by rating value
                        palette="viridis",  # Set color palette
                        height=3,  # Set height of each subplot
                        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.3}  # Set scatter plot styling
                    )
                    # Adjust subplot spacing
                    pair_plot.fig.subplots_adjust(top=0.95)
                    # Add overall title to the plot
                    pair_plot.fig.suptitle("Pair Plot of Age, Publication Year, and Book Rating", fontsize=16)
                    
                    # Display the pair plot in Streamlit
                    st.pyplot(pair_plot.fig)
                    
                    # Add explanatory text about the pair plot
                    st.write("""
                    **Pair Plot Analysis:**
                    
                    This visualization shows the relationships between reader age, book publication year, and ratings.
                    The diagonal shows the distribution of each variable, while the scatter plots show relationships between pairs of variables.
                    Colors represent different rating values, helping identify patterns in how different age groups rate books of different publication years.
                    """)
    
    # Check if "User Analysis" page is selected
    elif page == "User Analysis":
        # Display the main header for this page
        st.header("User Analysis")
        
        # Create a dropdown to select a user ID from sorted unique user IDs
        user_id = st.selectbox("Select User ID", sorted(users['User-ID'].unique()))
        
        # Continue only if a user ID is selected
        if user_id:
            # Filter users dataframe to get info for the selected user
            user_info = users[users['User-ID'] == user_id]
            
            # Create two equal-width columns for layout
            col1, col2 = st.columns(2)
            
            # Use the first column for user basic information
            with col1:
                # Display a subheader for user information section
                st.subheader("User Information")
                # Show the user ID
                st.write(f"**User ID:** {user_id}")
                # Show user age if available, otherwise show "Not specified"
                st.write(f"**Age:** {user_info['Age'].values[0] if not pd.isna(user_info['Age'].values[0]) else 'Not specified'}")
                # Country column has been removed as it doesn't exist in the dataset
            
            # Get all books rated by the selected user
            user_books = get_user_book_ratings(user_id, ratings, books)
            
            # Use the second column for rating statistics
            with col2:
                # Display a subheader for rating statistics section
                st.subheader("Rating Statistics")
                # Show the total number of books rated by the user
                st.write(f"**Number of Books Rated:** {len(user_books)}")
                # Only calculate statistics if user has rated at least one book
                if len(user_books) > 0:
                    # Calculate and display average rating
                    avg_rating = user_books['Book-Rating'].mean()
                    st.write(f"**Average Rating:** {avg_rating:.2f}")
                    # Calculate and display median rating
                    median_rating = user_books['Book-Rating'].median()
                    st.write(f"**Median Rating:** {median_rating:.1f}")
                    # Find and display the most common rating given by user
                    most_common_rating = user_books['Book-Rating'].mode()[0]
                    st.write(f"**Most Common Rating:** {most_common_rating}")
            
            # If user has rated at least one book, show detailed analysis
            if len(user_books) > 0:
                # Create tabs for different types of visualizations and analysis
                user_tabs = st.tabs(["Books Rated", "Rating Distribution", "Rating Patterns", "Authors & Publishers"])
                
                # First tab: Display books rated by user
                with user_tabs[0]:
                    # Display a subheader for this section
                    st.subheader("Books Rated by User")
                    # Sort books by rating in descending order and select relevant columns
                    user_books_display = user_books.sort_values('Book-Rating', ascending=False)[
                        ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Book-Rating']
                    ]
                    # Display the table of books rated by user
                    st.dataframe(user_books_display)
                    
                    # Add download button so users can export the data
                    csv = user_books_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download User's Ratings",
                        data=csv,
                        file_name=f"user_{user_id}_ratings.csv",
                        mime="text/csv",
                    )
                
                # Second tab: Show distribution of ratings
                with user_tabs[1]:
                    # Display a subheader for this section
                    st.subheader("Rating Distribution")
                    # Create two columns for different chart types
                    col_hist, col_pie = st.columns(2)
                    
                    # Left column: Histogram of ratings
                    with col_hist:
                        # Create figure and axes for the histogram
                        fig, ax = plt.figure(figsize=(8, 4)), plt.axes()
                        # Plot count of each rating value as a bar chart
                        sns.countplot(x='Book-Rating', data=user_books, ax=ax, palette='viridis')
                        # Add a title to the plot
                        plt.title(f'Rating Distribution for User {user_id}')
                        # Add label for x-axis
                        plt.xlabel('Rating')
                        # Add label for y-axis
                        plt.ylabel('Count')
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                    
                    # Right column: Pie chart of ratings
                    with col_pie:
                        # Create figure and axes for the pie chart
                        fig, ax = plt.figure(figsize=(8, 4)), plt.axes()
                        # Create pie chart showing percentage of each rating value
                        user_books['Book-Rating'].value_counts().plot.pie(
                            autopct='%1.1f%%',  # Show percentage labels with one decimal place
                            ax=ax,  # Use the current axes
                            shadow=True,  # Add shadow effect to pie chart
                            labels=None,  # Hide labels on the pie slices
                            colors=sns.color_palette('viridis', user_books['Book-Rating'].nunique())  # Use viridis color palette
                        )
                        # Add a title to the plot
                        plt.title('Rating Distribution (%)')
                        # Remove default y-label
                        plt.ylabel('')
                        # Add legend showing which color corresponds to which rating
                        ax.legend(
                            title="Rating",
                            loc="center left",  # Position legend to the left of the chart
                            bbox_to_anchor=(1, 0, 0.5, 1)  # Adjust legend position outside the pie chart
                        )
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                
                # Third tab: Analyze patterns in user ratings
                with user_tabs[2]:
                    # Display a subheader for this section
                    st.subheader("Rating Patterns")
                    
                    # Check if the user has rated books from different publication years
                    if user_books['Year-Of-Publication'].nunique() > 1:
                        # Convert publication year strings to numeric values, with error handling
                        user_books['Year-Of-Publication'] = pd.to_numeric(user_books['Year-Of-Publication'], errors='coerce')
                        
                        # Filter to include only books with realistic publication years (1900-2010)
                        valid_years_books = user_books[user_books['Year-Of-Publication'].between(1900, 2010)]
                        
                        # Only proceed if we have enough data points for meaningful analysis
                        if len(valid_years_books) > 3:  
                            # Create scatter plot of ratings vs publication year
                            fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                            sns.scatterplot(
                                data=valid_years_books,
                                x='Year-Of-Publication',
                                y='Book-Rating',
                                alpha=0.7,  # Set partial transparency
                                s=100  # Set point size
                            )
                            
                            # Try to add a trend line to identify rating patterns over time
                            try:
                                # Import required libraries for trend line calculation
                                import numpy as np
                                from scipy import stats
                                
                                # Get x and y data for trend line calculation
                                x = valid_years_books['Year-Of-Publication']
                                y = valid_years_books['Book-Rating']
                                
                                # Calculate linear regression parameters
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                # Generate y values for the trend line
                                trend_line = slope * x + intercept
                                
                                # Add the trend line to the scatter plot
                                plt.plot(x, trend_line, 'r--', linewidth=2)
                                
                                # Add correlation value as an annotation on the plot
                                corr = round(r_value, 2)
                                plt.annotate(
                                    f'Correlation: {corr}',  # Text to display
                                    xy=(0.05, 0.95),  # Position (5% from left, 95% from bottom)
                                    xycoords='axes fraction',  # Position relative to the axes
                                    fontsize=12,  # Font size
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)  # Add a white box behind text
                                )
                                
                            # If trend line calculation fails, continue without it
                            except Exception as e:
                                pass  
                            
                            # Add title and axis labels to the plot
                            plt.title(f'Ratings by Publication Year for User {user_id}')
                            plt.xlabel('Publication Year')
                            plt.ylabel('Rating')
                            # Display the plot in Streamlit
                            st.pyplot(fig)
                            
                            # Add interpretive text based on correlation value
                            if corr > 0.3:
                                st.info(f"This user seems to prefer newer books (positive correlation: {corr:.2f}).")
                            elif corr < -0.3:
                                st.info(f"This user seems to prefer older books (negative correlation: {corr:.2f}).")
                            else:
                                st.info(f"This user doesn't show a strong preference for books from any particular era (correlation: {corr:.2f}).")
                        else:
                            # Not enough data points for meaningful analysis
                            st.info("Not enough books with valid publication years to analyze patterns.")
                    else:
                        # User has rated books from only one year or missing year data
                        st.info("All rated books have the same publication year or missing values.")
                
                # Fourth tab: Analyze favorite authors and publishers
                with user_tabs[3]:
                    # Display a subheader for this section
                    st.subheader("Top Authors & Publishers")
                    
                    # Create two columns for author and publisher analysis
                    col_authors, col_publishers = st.columns(2)
                    
                    # Left column: Author analysis
                    with col_authors:
                        # Count books by each author
                        author_counts = user_books['Book-Author'].value_counts()
                        # Only create visualization if user has rated books from multiple authors
                        if len(author_counts) > 1:
                            # Create figure with dynamic height based on number of authors
                            fig, ax = plt.figure(figsize=(8, min(10, len(author_counts)))), plt.axes()
                            
                            # Limit to top 10 authors to keep visualization readable
                            top_authors = author_counts.head(10)
                            # Create horizontal bar chart of top authors
                            sns.barplot(
                                x=top_authors.values,  # Count of books for each author
                                y=top_authors.index,   # Author names
                                palette='viridis',     # Color palette
                                ax=ax                 # Use the current axes
                            )
                            # Add title and label
                            plt.title(f'Top Authors Rated by User {user_id}')
                            plt.xlabel('Number of Books')
                            # Display the plot in Streamlit
                            st.pyplot(fig)
                            
                            # Calculate average rating per author
                            author_ratings = user_books.groupby('Book-Author')['Book-Rating'].mean().sort_values(ascending=False)
                            # Count the number of books rated per author
                            author_counts = user_books.groupby('Book-Author')['Book-Rating'].count()
                            # Get only authors with multiple books rated
                            multi_book_authors = author_counts[author_counts > 1].index
                            
                            # If there are authors with multiple books rated, show their average ratings
                            if len(multi_book_authors) > 0:
                                # Display a header for this section
                                st.write("**Average Rating by Author** (authors with multiple books)")
                                # Create a dataframe with author ratings information
                                author_data = pd.DataFrame({
                                    'Author': multi_book_authors,
                                    'Average Rating': author_ratings[multi_book_authors].values,
                                    'Books Rated': author_counts[multi_book_authors].values
                                }).sort_values('Average Rating', ascending=False)
                                # Display the table
                                st.dataframe(author_data)
                        else:
                            # If user has rated books from only one author
                            st.info("User has rated books from only one author.")
                    
                    # Right column: Publisher analysis
                    with col_publishers:
                        # Count books by each publisher
                        publisher_counts = user_books['Publisher'].value_counts()
                        # Only create visualization if user has rated books from multiple publishers
                        if len(publisher_counts) > 1:
                            # Create figure with dynamic height based on number of publishers
                            fig, ax = plt.figure(figsize=(8, min(10, len(publisher_counts)))), plt.axes()
                            
                            # Limit to top 10 publishers to keep visualization readable
                            top_publishers = publisher_counts.head(10)
                            # Create horizontal bar chart of top publishers
                            sns.barplot(
                                x=top_publishers.values,  # Count of books for each publisher
                                y=top_publishers.index,   # Publisher names
                                palette='rocket',        # Different color palette for distinction
                                ax=ax                    # Use the current axes
                            )
                            # Add title and label
                            plt.title(f'Top Publishers Rated by User {user_id}')
                            plt.xlabel('Number of Books')
                            # Display the plot in Streamlit
                            st.pyplot(fig)
                        else:
                            # If user has rated books from only one publisher
                            st.info("User has rated books from only one publisher.")
            else:
                # Warning message if user hasn't rated any books
                st.warning("This user has not rated any books.")
    
    elif page == "Book Recommendations":  # Check if user selected Book Recommendations page
        st.header("Book Recommendations")  # Display page header
        
        # Create three tabs for different recommendation methods
        tab1, tab2, tab3 = st.tabs(["Get Personalized Recommendations", "Find Similar Books", "View User's Top Rated Books"])
        
        with tab1:  # First tab: Personalized recommendations
            col1, col2 = st.columns([1, 2])  # Create two columns with 1:2 ratio
            
            with col1:  # Left column for input controls
                st.subheader("Get Recommendations")  # Display subheader
                # Create dropdown to select a user ID from sorted list
                user_ids = sorted(users['User-ID'].unique())
                user_id = st.selectbox("Select User ID", user_ids)
                
                # Dropdown to choose recommendation algorithm
                model_type = st.selectbox(
                    "Select Recommendation Model",
                    ["Collaborative Filtering (Original)", "KNN", "SVD", "NMF"]
                )
                
                num_recommendations = st.slider("Number of Recommendations", 1, 100, 5)  # Slider to choose number of recommendations
                recommend_button = st.button("Get Recommendations")  # Button to trigger recommendations
            
            if recommend_button:  # Execute when recommendation button is clicked
                with col2:  # Display recommendations in right column
                    st.subheader(f"Top {num_recommendations} Book Recommendations for User {user_id}")  # Dynamic subheader
                    
                    with st.spinner("Finding the best books for you..."):  # Show loading spinner
                        # Initialize progress bar for user feedback during calculation
                        progress_bar = st.progress(0)
                        
                        try:  # Try block to handle potential errors
                            # Choose appropriate recommendation function based on selected model
                            if model_type == "KNN":
                                recommended_books = get_knn_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            elif model_type == "SVD":
                                recommended_books = get_svd_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            elif model_type == "NMF":
                                recommended_books = get_nmf_recommendations(user_id, ratings, books, num_recommendations, progress_bar)
                            else:  # Default to original collaborative filtering
                                recommended_books = get_book_recommendations_simple(user_id, ratings, books, num_recommendations, progress_bar)
                            
                            progress_bar.progress(100)  # Set progress bar to 100% when complete
                            
                            if len(recommended_books) > 0:  # Check if we got any recommendations
                                # Loop through each recommended book to display details
                                for i, (idx, book) in enumerate(recommended_books.iterrows()):
                                    col_img, col_info = st.columns([1, 3])  # Create columns for image and info
                                    
                                    with col_img:  # Left column for book cover image
                                        # Try to display book cover image with error handling
                                        try:
                                            if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                                # Check if URL starts with 'http'
                                                img_url = book['Image-URL-M']
                                                if img_url.startswith('http'):
                                                    st.image(img_url, width=150)  # Display image from URL
                                                else:
                                                    # For local or relative paths
                                                    st.write("📚 Image path invalid")
                                            else:
                                                st.write("📚 No image provided")
                                        except Exception as e:
                                            st.write("📚 Image unavailable")
                                            # Uncomment for debugging: st.write(f"Error: {str(e)}")
                                    
                                    with col_info:  # Right column for book details
                                        st.subheader(f"{i+1}. {book['Book-Title']}")  # Display numbered book title
                                        st.write(f"**Author:** {book['Book-Author']}")  # Display author
                                        st.write(f"**Published:** {book['Year-Of-Publication']}")  # Display publication year
                                        st.write(f"**Publisher:** {book['Publisher']}")  # Display publisher name
                                        
                                        # Handle different score column names from different algorithms
                                        if 'score' in book:
                                            st.write(f"**Recommendation Score:** {book['score']:.2f}")  # Display rounded score
                                            st.write(f"**Based on:** {int(book['supporting_users'])} similar users")  # Display support count
                                        elif 'mean' in book:
                                            st.write(f"**Recommendation Score:** {book['mean']:.2f}")  # Display rounded mean score
                                            if 'count' in book:
                                                st.write(f"**Based on:** {int(book['count'])} ratings")  # Display count of ratings
                            else:  # Handle case when no recommendations are found
                                st.warning("Not enough data to make recommendations for this user. Showing popular books instead.")
                                popular_books = get_popular_books(books, ratings, num_recommendations)  # Get popular books as fallback
                                st.dataframe(popular_books[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'mean', 'count']])  # Display in table format
                        except Exception as e:  # Catch any errors during recommendation process
                            st.error(f"An error occurred: {str(e)}")  # Display error message
                            st.info("Showing popular books instead.")  # Inform user about fallback
                            popular_books = get_popular_books(books, ratings, num_recommendations)  # Get popular books as fallback
                            
                            # Display popular books in a visual format with image and details
                            for i, (idx, book) in enumerate(popular_books.iterrows()):
                                col_img, col_info = st.columns([1, 3])  # Create columns for image and info
                                
                                with col_img:  # Image column
                                    try:  # Try to display book cover image
                                        if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                            st.image(book['Image-URL-M'], use_container_width=True)  # Show image
                                        else:
                                            st.write("📚 No image available")  # Show placeholder text if no image
                                    except Exception:
                                        st.write("📚 Image unavailable")  # Handle image loading errors
                                
                                with col_info:  # Book details column
                                    st.subheader(f"{i+1}. {book['Book-Title']}")  # Display numbered title
                                    st.write(f"**Author:** {book['Book-Author']}")  # Display author
                                    st.write(f"**Published:** {book['Year-Of-Publication']}")  # Display publication year
                                    st.write(f"**Publisher:** {book['Publisher']}")  # Display publisher
                                    st.write(f"**Average Rating:** {book['mean']:.2f}")  # Display rounded average rating
                                    st.write(f"**Based on:** {int(book['count'])} ratings")  # Display count of ratings
        
        # Second tab for content-based book recommendations
        with tab2:
            st.subheader("Find Similar Books")  # Display tab subheader
            st.write("Enter or select a book title to find books with similar content")  # Instructions for user
            
            col1, col2 = st.columns([1, 2])  # Create two columns with 1:2 ratio
            
            with col1:  # Input controls column
                # Selection for recommendation method
                method = st.selectbox(
                    "Recommendation Method",
                    ["TF-IDF (Default)", "Word Embeddings", "BERT Embeddings"],
                    help="TF-IDF: Fast text similarity. Word Embeddings: Better semantic understanding. BERT: Advanced NLP understanding."
                )
                
                # Create list of unique book titles for dropdown
                unique_titles = sorted(books['Book-Title'].dropna().unique())
                
                # Allow users to either type or select a book title
                search_option = st.radio("Search method", ["Type book title", "Select from dropdown"])
                
                if search_option == "Type book title":  # Free text input option
                    book_query = st.text_input("Book Title", "Harry Potter")  # Text input with default value
                else:  # Dropdown selection option
                    # Add search box to filter dropdown options
                    title_search = st.text_input("Search for a book", "")
                    # Filter titles based on search text
                    filtered_titles = [title for title in unique_titles if title_search.lower() in title.lower()]
                    # Limit results to prevent UI performance issues
                    if len(filtered_titles) > 1000:
                        st.info(f"Showing first 1000 of {len(filtered_titles)} matching titles. Please refine your search.")
                        filtered_titles = filtered_titles[:1000]  # Limit to first 1000 matches
                    book_query = st.selectbox("Select a book", filtered_titles)  # Dropdown with filtered titles
                    
                num_similar_books = st.slider("Number of Similar Books", 1, 100, 5, key="similar_books_slider")  # Slider for number of results
                find_button = st.button("Find Similar Books")  # Button to trigger search
            
            if find_button:  # Execute when find button is clicked
                with col2:  # Results column
                    st.subheader(f"Books similar to '{book_query}'")  # Dynamic subheader with search query
                    
                    with st.spinner("Finding similar books..."):  # Show loading spinner
                        # Create progress bar for user feedback
                        progress_bar = st.progress(0)
                        
                        try:  # Try block for error handling
                            # Choose appropriate content-based recommendation function
                            if method == "Word Embeddings":
                                similar_books = get_word_embedding_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            elif method == "BERT Embeddings":
                                similar_books = get_bert_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            else:  # Default to TF-IDF
                                similar_books = get_content_based_recommendations(book_query, books, num_similar_books*3, progress_bar)
                            
                            # Process and display results
                            if len(similar_books) > 0:  # Check if any similar books were found
                                # Remove duplicate titles to show more unique recommendations
                                seen_titles = set()  # Set to track seen titles
                                unique_books_indices = []  # List to store unique book indices
                                
                                # Process books to remove duplicates
                                for idx, row in similar_books.iterrows():
                                    title = row['Book-Title']
                                    if title not in seen_titles:  # Check if title already processed
                                        seen_titles.add(title)  # Add to seen titles
                                        unique_books_indices.append(idx)  # Store index of unique book
                                    
                                    # Stop once we have enough unique books
                                    if len(unique_books_indices) >= num_similar_books:
                                        break
                                
                                # Get unique books using saved indices
                                unique_similar_books = similar_books.loc[unique_books_indices]
                                
                                # Display each similar book
                                for i, (idx, book) in enumerate(unique_similar_books.iterrows(), 1):
                                    col_img, col_info = st.columns([1, 3])  # Create columns for image and info
                                    
                                    with col_img:  # Image column
                                        try:  # Try to display book cover image
                                            if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                                # Check if URL starts with 'http'
                                                img_url = book['Image-URL-M']
                                                if img_url.startswith('http'):
                                                    st.image(img_url, width=150)  # Display image from URL
                                                else:
                                                    # For local or relative paths
                                                    st.write("📚 Image path invalid")
                                            else:
                                                st.write("📚 No image provided")
                                        except Exception as e:
                                            st.write("📚 Image unavailable")
                                    
                                    with col_info:  # Book details column
                                        st.subheader(f"{i}. {book['Book-Title']}")  # Display numbered title
                                        st.write(f"**Author:** {book['Book-Author']}")  # Display author
                                        st.write(f"**Published:** {book['Year-Of-Publication']}")  # Display publication year
                                        st.write(f"**Publisher:** {book['Publisher']}")  # Display publisher
                                        if 'mean' in book:
                                            st.write(f"**Similarity Score:** {book['mean']:.2f}")  # Display rounded similarity score
                            else:  # Handle case when no similar books are found
                                st.warning(f"No books similar to '{book_query}' found. Try another title.")
                        except Exception as e:  # Catch and display errors
                            st.error(f"An error occurred while finding similar books: {str(e)}")  # Show error message
                            st.exception(e)  # Show detailed exception information for debugging

        # Third tab for viewing a user's top rated books
        with tab3:
            st.subheader("User's Top Rated Books")  # Display tab subheader
            # Create dropdown to select user ID
            user_ids = sorted(users['User-ID'].unique())
            selected_user_id = st.selectbox("Select User ID", user_ids, key="top_rated_user_select")  # Unique key to avoid conflicts
            max_books = st.slider("Maximum Number of Books to Show", 1, 100, 10)  # Slider for result count
            
            if st.button("Show Top Rated Books"):  # Button to trigger display
                # Get all books rated by the selected user
                user_books = get_user_book_ratings(selected_user_id, ratings, books)
                
                if len(user_books) > 0:  # Check if user has rated any books
                    st.write(f"User {selected_user_id} has rated {len(user_books)} books.")  # Show rating count
                    
                    # Sort by rating (descending) and get top N books
                    top_rated = user_books.sort_values('Book-Rating', ascending=False).head(max_books)
                    
                    # Display each top-rated book
                    for i, (idx, book) in enumerate(top_rated.iterrows()):
                        col_img, col_info = st.columns([1, 3])  # Create columns for image and info
                        
                        with col_img:  # Image column
                            try:  # Try to display book cover image
                                if pd.notna(book['Image-URL-M']) and book['Image-URL-M'].strip() != '':
                                    # Check if URL starts with 'http'
                                    img_url = book['Image-URL-M']
                                    if img_url.startswith('http'):
                                        st.image(img_url, width=150)  # Display image from URL
                                    else:
                                        # For local or relative paths
                                        st.write("📚 Image path invalid")
                                else:
                                    st.write("📚 No image provided")
                            except Exception as e:
                                st.write("📚 Image unavailable")
                        
                        with col_info:  # Book details column
                            st.subheader(f"{i+1}. {book['Book-Title']}")  # Display numbered title
                            st.write(f"**Author:** {book['Book-Author']}")  # Display author
                            st.write(f"**Published:** {book['Year-Of-Publication']}")  # Display publication year
                            st.write(f"**User's Rating:** {int(book['Book-Rating'])} / 10")  # Display user's rating out of 10
                else:  # Handle case when user hasn't rated any books
                    st.warning(f"User {selected_user_id} has not rated any books.")
    elif page == "About":  # About page content
        st.header("About This Book Recommendation System")  # Display page header
        
        # Create tabs for different sections of the about page
        about_tabs = st.tabs(["Project Overview", "Technical Details", "Models & Algorithms", "Data Processing", "Error Handling"])
        
        with about_tabs[0]:  # Project Overview tab
            st.subheader("Project Overview")
            st.write("""
            This advanced book recommendation system was developed as part of ExcelR's data science curriculum. The application 
            analyzes reading preferences and user behavior to suggest personalized book recommendations using multiple 
            state-of-the-art algorithms.
            
            The system provides:
            - Personalized book recommendations based on user ratings
            - Similar book discovery based on content similarity
            - Comprehensive data exploration and visualization tools
            - Detailed user behavior analysis
            - Multiple recommendation models with different strengths
            
            ### Dataset
            
            The project utilizes the Book-Crossing dataset which includes:
            - **Books**: 271,379 books with metadata (title, author, year, publisher, etc.)
            - **Users**: 278,858 users with optional demographic information
            - **Ratings**: 1,149,780 book ratings (explicit and implicit)
            
            ### Development Team
            
            - **Developers:** Soheb, Shafeeq, Arthi, and Navya
            - **GitHub:** [sohebkh13/ExcelR_Book_Recommendation_Project](https://github.com/sohebkh13/book_recommendation/)
            """)
        
        with about_tabs[1]:  # Technical Details tab
            st.subheader("Technical Implementation")
            st.write("""
            ### Application Architecture
            
            The application is built using:
            - **Streamlit**: For the interactive web interface
            - **Pandas & NumPy**: For data manipulation and numerical operations
            - **Matplotlib & Seaborn**: For data visualization
            - **Scikit-learn**: For machine learning algorithms and evaluation metrics
            - **Surprise**: For collaborative filtering recommendation algorithms
            - **NLTK & Gensim**: For natural language processing and text analysis
            - **Sentence-Transformers**: For BERT-based text embeddings
            
            ### Performance Optimizations
            
            The application implements several optimizations to handle large datasets efficiently:
            - **Data caching**: Using `@st.cache_data` to avoid reloading data
            - **Sampling techniques**: Dynamically sampling large datasets for visualization and modeling
            - **Batch processing**: Processing recommendations in batches to manage memory usage
            - **Sparse matrices**: Using scipy's sparse matrices for efficient memory usage with large user-item matrices
            - **Progressive loading**: Using progress bars to keep users informed during computation-heavy operations
            """)
        
        with about_tabs[2]:  # Models & Algorithms tab
            st.subheader("Recommendation Models")
            st.write("""
            ### Collaborative Filtering Models
            
            1. **Original Collaborative Filtering**
               - Identifies users with similar rating patterns
               - Calculates weighted ratings to predict user preferences
               - Includes diminishing weight factors for less similar users
               - Falls back to popularity-based recommendations when needed
               - **Example:** For a user who rated "Harry Potter and the Sorcerer's Stone" highly, might recommend "The Lord of the Rings" and "The Chronicles of Narnia" based on similar users' ratings, regardless of content similarities
            
            2. **K-Nearest Neighbors (KNN)**
               - Identifies the most similar users based on common book ratings
               - Uses cosine similarity to measure user preference patterns
               - Optimized for memory-efficiency with large datasets
               - Parameters: k=20 (number of neighbors)
               - **Example:** For the same Harry Potter fan, might recommend "Percy Jackson & the Olympians" because the 20 most similar users also enjoyed that series, focusing on the closest matching users
            
            3. **Singular Value Decomposition (SVD)**
               - Matrix factorization technique for dimensionality reduction
               - Identifies latent factors in user-item interactions
               - Implemented using the Surprise library for collaborative filtering
               - Parameters: n_factors=50 (number of latent factors)
               - **Example:** Might recommend "Eragon" or "A Series of Unfortunate Events" based on underlying patterns in user preferences that capture the latent "young adult fantasy" factor, even if users haven't explicitly rated both series
            
            4. **Non-negative Matrix Factorization (NMF)**
               - Factorizes the rating matrix with non-negativity constraints
               - Particularly effective for sparse rating data
               - Memory-optimized implementation for large datasets
               - Parameters: n_components=15 (number of components)
               - **Example:** For Harry Potter fans, might recommend "The Golden Compass" by identifying component patterns like "British authors" or "magical coming-of-age stories" that connect these books
            
            ### Content-Based Filtering Models
            
            1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
               - Analyzes book titles, authors, and publishers
               - Removes English stop words to focus on meaningful terms
               - Uses cosine similarity to find books with similar content
               - Fastest content-based approach, requiring minimal resources
               - **Example:** For "Harry Potter and the Chamber of Secrets," would recommend other Harry Potter books first (due to exact word matches), then possibly books with "wizard" or "magic" in the title
            
            2. **Word Embeddings**
               - Uses pre-trained GloVe word vectors (100-dimensional)
               - Captures semantic relationships between words
               - Creates document embeddings by averaging word vectors
               - More nuanced understanding of text than TF-IDF
               - **Example:** Would recommend "The Magicians" by Lev Grossman because terms like "wizard," "magic," and "school" are semantically related in the vector space, even if they don't share exact words with Harry Potter
            
            3. **BERT Embeddings**
               - Uses SentenceTransformer with paraphrase-MiniLM-L6-v2 model
               - State-of-the-art natural language understanding
               - Captures complex contextual relationships in text
               - Most advanced semantic understanding of book content
               - **Example:** Could recommend "Jonathan Strange & Mr Norrell" by Susanna Clarke – a completely different writing style and target audience, but BERT understands the shared context of "British magical reality" and "alternative history with magic" concepts
            """)
            
        with about_tabs[3]:  # Data Processing tab
            st.subheader("Data Processing & Validation")
            st.write("""
            ### Data Loading & Preprocessing
            
            - **Multiple encoding support**: Attempts various encodings (cp1252) when reading CSV files
            - **Missing value handling**: Comprehensive analysis and appropriate handling of missing data
            - **Data type conversion**: Proper conversion of numeric fields with error handling
            - **Outlier filtering**:
              - Publication years limited to realistic range (1900-2010)
              - User ages limited to reasonable range (5-100)
              - Zero ratings filtered out for certain analyses
            
            ### Input Validation
            
            - **User ID validation**: Checks for existence in dataset and rating history
            - **Book title validation**: Implements fuzzy matching for partial title inputs
            - **Parameter validation**: Ensures recommendation counts, similarity thresholds, etc. are within acceptable ranges
            - **Empty result handling**: Falls back to alternative recommendation sources when primary methods yield no results
            
            ### Data Transformation
            
            - **Text preprocessing**: Cleaning and normalizing book titles, authors, and publishers
            - **Feature creation**: Combining text features for content-based analysis
            - **Matrix creation**: Converting dataframes to user-item matrices for collaborative filtering
            - **Vector normalization**: Normalizing feature vectors for similarity calculations
            - **Dimensionality reduction**: Applying SVD and NMF to reduce feature dimensions
            """)
            
        with about_tabs[4]:  # Error Handling tab
            st.subheader("Error Handling & Graceful Degradation")
            st.write("""
            ### Comprehensive Error Handling
            
            - **File loading errors**: Graceful handling of missing files with informative error messages
            - **Encoding errors**: Multiple encoding attempts with fallback options
            - **Model loading failures**: Alternative model suggestions when primary models fail to load
            - **Computation errors**: Try-except blocks around computational operations with fallbacks
            - **Image loading errors**: Graceful handling of missing or invalid image URLs
            
            ### Fallback Mechanisms
            
            - **Popular book recommendations**: When user-specific recommendations cannot be generated
            - **Method switching**: Automatic switching to simpler methods when advanced methods fail
            - **Random sampling**: Providing random selections when all recommendation attempts fail
            - **TF-IDF fallback**: When word embeddings or BERT models cannot be loaded
            - **Subsampling**: Dynamically adjusting data size to prevent memory issues
            
            ### User Feedback
            
            - **Progress bars**: Visual indication of computation progress for long-running operations
            - **Information messages**: Clear explanations when fallback mechanisms are activated
            - **Warning messages**: Alerts about potential issues in data or recommendations
            - **Error messages**: User-friendly explanations when errors occur
            - **Loading spinners**: Visual feedback during data loading and processing
            """)
        
        # Add contact and additional resources section
        st.markdown("---")
        st.subheader("Additional Resources")
        
        # Create columns for resources and contact info
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("### Documentation")
            st.markdown("""
            - [Surprise Library Documentation](https://surprise.readthedocs.io/)
            - [Streamlit Documentation](https://docs.streamlit.io/)
            - [Sentence-Transformers Documentation](https://www.sbert.net/)
            """)
            
        with res_col2:
            st.write("### Contact")
            st.write("For questions or suggestions about this project:")
            st.write("📧 Email: sohebkhan10@gmail.com")
            st.write("🐱 GitHub Issues: [Report a Bug](https://github.com/sohebkh13/book_recommendation/issues)")
    elif page == "About":  # About page content
        st.header("About This Project")  # Display page header
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
        """)  # Display project information using markdown

if __name__ == "__main__":  # Entry point check
    main()  # Call the main function when script is executed directly