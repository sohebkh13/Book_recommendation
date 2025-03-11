```markdown
# Book Recommendation Systems - Comprehensive Guide

This guide explains different recommendation models used in our book recommendation system, including code examples and theoretical explanations.

## Table of Contents
1. [BERT Model](#bert-model)
2. [Word Embeddings Model](#word-embeddings-model)
3. [KNN (K-Nearest Neighbors)](#knn-k-nearest-neighbors)
4. [SVD (Singular Value Decomposition)](#svd-singular-value-decomposition)
5. [NMF (Non-Negative Matrix Factorization)](#nmf-non-negative-matrix-factorization)
6. [Collaborative Filtering](#collaborative-filtering-original)
7. [Comparing All Models](#comparing-all-models-at-a-glance)

## BERT Model

### Introduction
This explains how a BERT-based recommendation system transforms books into vectors to find similar books.

### Line-by-Line Code Explanation

This function takes a book title, your books database, how many recommendations you want, and a progress bar.

```python
import numpy as np
```
Imports NumPy which we'll use for vector math operations.

```python
if progress_bar:
    progress_bar.progress(10)
```
Updates the progress bar to show we're starting.

```python
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    st.warning(f"Failed to load BERT model: {e}")
    return get_content_based_recommendations(book_title, books_df, n, progress_bar)
```
This loads the "brain" of our system - the Sentence Transformer model. We're using a specific pre-trained model called 'paraphrase-MiniLM-L6-v2'. If it fails to load, we fall back to the simpler TF-IDF model.

```python
processed_books = preprocess_book_data(books_df)
```
Gets our books ready for the model by filling missing information and combining book title, author, and publisher into one text field called 'content'.

```python
query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
if len(query_books) == 0:
    return books_df.sample(n)
query_book = query_books.iloc[0]
```
Finds the book you're interested in. If it can't find your book, it returns random books.

```python
sample_size = min(1000, len(processed_books))
books_sample = processed_books.sample(sample_size) if len(processed_books) > sample_size else processed_books
```
Limits analysis to 1000 books maximum for speed (BERT is computationally intensive).

```python
titles = books_sample['Book-Title'].tolist()
authors = books_sample['Book-Author'].tolist()
book_texts = [f"{title} by {author}" for title, author in zip(titles, authors)]
embeddings = model.encode(book_texts, show_progress_bar=False)
```
**THIS IS WHERE THE MAGIC HAPPENS!**
- Creates text descriptions like "Harry Potter by J.K. Rowling"
- The model.encode() transforms each book text into a vector (list of 384 numbers)

```python
query_text = f"{query_book['Book-Title']} by {query_book['Book-Author']}"
query_embedding = model.encode([query_text])[0]
```
Does the same transformation for your chosen book.

```python
similarities = []
for i, embedding in enumerate(embeddings):
    if books_sample.index[i] != query_book.name:
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((books_sample.index[i], similarity))
```
Compares your book's vector to every other book's vector using "cosine similarity" - a math formula that tells how similar two vectors are. Higher number = more similar books.

```python
similar_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)][:n]
similar_books = processed_books.loc[similar_indices].copy()
```
Sorts all books by similarity and picks the top N most similar books.

```python
for i, (idx, sim) in enumerate([(idx, sim) for idx, sim in similarities if idx in similar_indices]):
    similar_books.loc[idx, 'mean'] = sim
```
Adds the similarity score to each recommendation so you can see how similar each book is.

### How Sentence Transformer Makes Words Into Vectors

The Sentence Transformer uses pre-defined vectors in a very sophisticated way:

#### Pre-training
The 'paraphrase-MiniLM-L6-v2' model was pre-trained on BILLIONS of sentences from books, Wikipedia, websites, etc. During this pre-training, it learned to understand language.

#### How It Learns Word Relationships
- During pre-training, the model sees sentences like: "The wizard cast a magic spell" and "The sorcerer used his magical powers"
- It learns these appear in similar contexts and have similar meanings
- The model adjusts its internal parameters to place "wizard" and "sorcerer" in similar locations in vector space

#### Vector Space
- Each word doesn't have just one vector - it depends on context!
- "Magic" in "Magic Kingdom" (theme park) is different from "Magic spell" (wizardry)
- Your model represents each word with 384 dimensions (numbers)
- Words with similar meanings are positioned closer together in this 384-dimensional space

#### Sentence Vectors
When you feed in "Harry Potter by J.K. Rowling", the model:
1. Breaks it into tokens (smaller word pieces)
2. Passes these through 6 transformer layers that analyze context
3. Creates a single 384-dimensional vector representing the entire book description
4. This vector captures themes like "magic", "school", "coming of age", "fantasy", etc.

#### Why "Magic" and "Wizard" Are Connected

The model wasn't explicitly told "magic = wizard", but it learned this association by seeing:

- These words often appear near each other in text: "The wizard used magic"
- These words are used in similar contexts in millions of examples
- During pre-training, when the model tried to predict masked words:
  - "The _____ cast a spell" (wizard fits here)
  - "The sorcerer used _____ powers" (magic fits here)
- It learned these words are interchangeable in many contexts

The vector for "Harry Potter" ends up having high values in dimensions representing concepts like:
- Fantasy literature
- Magic/wizardry
- School settings
- Youth adventures
- Good vs. evil

Then a book like "The Worst Witch" also has high values in similar dimensions, even if it uses different exact words!

#### The Pre-defined Knowledge

All of this knowledge is "pre-defined" in the sense that:
- The model was trained on massive text datasets before you ever used it
- It contains compressed knowledge from reading millions of books and websites
- It already "knows" that wizards use magic and that Hogwarts is a school for magic
- Your code doesn't teach it these relationships - it just leverages the knowledge already embedded in the model

This is why BERT is so powerful - it already understands language and concepts very deeply before you even start using it for book recommendations!

-----------------------------------------------------------------------------------------------------------

## Word Embeddings Model

### What Are Word Embeddings?

Word embeddings are numerical representations of words in a multi-dimensional space. Think of them as giving each word a unique "address" in a mathematical universe.

### Basic Concept

- **Traditional approach:** Words are treated as discrete symbols (cat ≠ kitten)
- **Word embeddings:** Words are represented as vectors of numbers
- These vectors capture semantic relationships between words

### How They Work

1. Each word is mapped to a vector (typically 100-300 dimensions)
2. Words with similar meanings have similar vectors
3. Example: vector("king") - vector("man") + vector("woman") ≈ vector("queen")

### Line-by-Line Code Explanation

```python
processed_books = preprocess_book_data(books_df)
```
Prepares the book data by cleaning and combining title, author, and publisher information.

```python
query_books = processed_books[processed_books['Book-Title'].str.contains(book_title, case=False)]
if len(query_books) == 0:
    return books_df.sample(n)
```
Finds the book you're looking for or returns random books if not found.

```python
try:
    import gensim.downloader as api
    word_vectors = api.load("glove-wiki-gigaword-100")
except Exception as e:
    st.warning(f"Failed to load word vectors: {e}")
    return get_content_based_recommendations(book_title, books_df, n, progress_bar)
```
Loads pre-trained GloVe word vectors that contain 100-dimensional representations of common words. These vectors were trained on Wikipedia and Gigaword text data.

```python
def get_book_vector(text):
    words = text.lower().split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    if not vectors:
        return np.zeros(word_vectors.vector_size)
    return np.mean(vectors, axis=0)
```
This function:
- Splits text into individual words
- Looks up the vector for each word (if available)
- Averages all the word vectors to create one vector for the entire book

```python
query_book = query_books.iloc[0]
query_vector = get_book_vector(query_book['content'])
```
Gets the vector representation of your chosen book.

```python
book_similarities = []
for idx, book in processed_books.iterrows():
    book_vector = get_book_vector(book['content'])
    similarity = np.dot(query_vector, book_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(book_vector))
    book_similarities.append((idx, similarity))
```
Calculates how similar each book is to your chosen book by comparing their vectors.

### Word Embeddings vs. BERT
Word Embeddings (like GloVe) differ from BERT in several ways:

#### Context-free vs. Contextual:
- **Word Embeddings:** Each word always has the same vector regardless of context
- **BERT:** A word's representation changes based on surrounding words

#### Simplicity:
- **Word Embeddings:** Simpler model, typically 100-300 dimensions per word
- **BERT:** More complex, capturing deeper relationships (384+ dimensions)

#### Book Representation:
- **Word Embeddings:** Simple averaging of individual word vectors
- **BERT:** Sophisticated contextual analysis of the entire text

#### When You Might Use It:
- **Word Embeddings:** Faster, requires less computing power
- **BERT:** Better understanding but more resource-intensive

### How It Understands "Magic" and "Wizard"
Word embeddings learn relationships from massive text datasets:
- Words used in similar contexts end up with similar vectors
- The vectors for "magic" and "wizard" will be closer together than "magic" and "car"
- However, unlike BERT, it doesn't understand that "The boy who lived" refers to Harry Potter because it processes individual words without context

-----------------------------------------------------------------------------------------------------------

## KNN (K-Nearest Neighbors)

### Line-by-Line Code Explanation

```python
def get_knn_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
```
This function recommends books based on finding similar users (neighbors).

```python
user_books = ratings_df[ratings_df['User-ID'] == user_id]
if len(user_books) == 0:
    return get_popular_books(books_df, ratings_df, n)
```
Gets all books rated by the user or returns popular books if user hasn't rated anything.

```python
user_rated_isbns = set(user_books['ISBN'].unique())
user_ratings_dict = dict(zip(user_books['ISBN'], user_books['Book-Rating']))
```
Creates a dictionary of books the user has rated and their ratings.

```python
users_who_rated_same_books = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()
similar_users = [u for u in users_who_rated_same_books if u != user_id]
```
Finds other users who have rated at least one book that our target user has rated.

```python
if len(similar_users) > 200:
    similar_users = random.sample(similar_users, 200)
```
Limits the number of users for performance reasons.

```python
similar_users_ratings = ratings_df[(ratings_df['User-ID'].isin(similar_users)) & 
                                  (ratings_df['ISBN'].isin(user_rated_isbns))]
```
Gets ratings from similar users for books that our target user has rated.

```python
similar_user_ratings = {}
for _, row in similar_users_ratings.iterrows():
    if row['User-ID'] not in similar_user_ratings:
        similar_user_ratings[row['User-ID']] = {}
    similar_user_ratings[row['User-ID']][row['ISBN']] = row['Book-Rating']
```
Organizes ratings into a nested dictionary for easy access.

```python
for sim_user, ratings in similar_user_ratings.items():
    common_books = set(ratings.keys()).intersection(user_ratings_dict.keys())
    
    if len(common_books) < 2:
        continue
        
    user_ratings_array = np.array([user_ratings_dict[isbn] for isbn in common_books])
    sim_user_ratings_array = np.array([ratings[isbn] for isbn in common_books])
    
    if len(common_books) > 0:
        user_magnitude = np.sqrt(np.sum(user_ratings_array**2))
        sim_user_magnitude = np.sqrt(np.sum(sim_user_ratings_array**2))
        
        if user_magnitude > 0 and sim_user_magnitude > 0:
            similarity = np.dot(user_ratings_array, sim_user_ratings_array) / (user_magnitude * sim_user_magnitude)
            user_similarities.append((sim_user, similarity, len(common_books)))
```
This is the CORE of KNN: calculating how similar each user is to our target user.
- Find books rated by both users
- Extract ratings into arrays
- Calculate cosine similarity between rating patterns
- Store similarity score and number of books in common

```python
user_similarities.sort(key=lambda x: (x[1], x[2]), reverse=True)
top_similar_users = [user for user, _, _ in user_similarities[:20]]
```
Sorts users by similarity and picks the 20 most similar users.

```python
books_rated_by_similar_users = ratings_df[(ratings_df['User-ID'].isin(top_similar_users)) & 
                                          (~ratings_df['ISBN'].isin(user_rated_isbns)) &
                                          (ratings_df['Book-Rating'] > 0)]
```
Gets books that similar users liked but our target user hasn't read yet.

```python
for sim_user, sim_score, _ in user_similarities:
    if sim_user in top_similar_users:
        user_books = books_rated_by_similar_users[books_rated_by_similar_users['User-ID'] == sim_user]
        
        weight = sim_score
        
        for _, row in user_books.iterrows():
            isbn = row['ISBN']
            rating = row['Book-Rating']
            
            if isbn not in book_scores:
                book_scores[isbn] = {'weighted_sum': 0, 'count': 0}
            
            book_scores[isbn]['weighted_sum'] += rating * weight
            book_scores[isbn]['count'] += 1
```
Calculates weighted ratings for each recommended book, giving more weight to ratings from more similar users.

### How KNN Works in Recommendation Systems
KNN is a form of collaborative filtering that works like this:

#### Finding Similar Users:
- Imagine each user as a point in space, positioned based on their ratings
- Users who rate books similarly will be close together
- KNN finds the "nearest neighbors" to our target user

#### The K in KNN:
- K represents how many similar users we consider (in your code, it's 20)
- More neighbors = more stable recommendations but less personalized
- Fewer neighbors = more personalized but potentially less reliable

#### Recommendation Logic:
- "If you liked book X and Y, and another user with similar taste liked book Z, you might like Z too"
- More similar users' opinions are weighted more heavily

#### Advantages:
- Understands personal preference patterns
- Can recommend unexpected books that similar users enjoyed
- Doesn't need to understand book content, just rating patterns


-----------------------------------------------------------------------------------------------------------

## SVD (Singular Value Decomposition)

### Line-by-Line Code Explanation

```python
@st.cache_data
def train_svd_model(ratings_df, n_factors=50):
    trainset = prepare_surprise_data(ratings_df)
    algo = SVD(n_factors=n_factors)
    algo.fit(trainset)
    return algo
```
This function trains an SVD model with 50 latent factors (hidden patterns in the data).

```python
def get_svd_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
```
Main function to get SVD recommendations for a user.

```python
user_books = get_user_book_ratings(user_id, ratings_df, books_df)
if len(user_books) == 0:
    return get_popular_books(books_df, ratings_df, n)
```
Gets books the user has rated or returns popular books if none.

```python
svd_model = train_svd_model(ratings_df)
```
Creates and trains the SVD model.

```python
user_rated_isbns = set(user_books['ISBN'])
all_isbns = set(books_df['ISBN'])
unrated_isbns = all_isbns - user_rated_isbns
```
Finds books the user hasn't rated yet.

```python
trainset = prepare_surprise_data(ratings_df)
predictions = []

for isbn in unrated_isbns:
    try:
        uid = trainset.to_inner_uid(user_id)
        iid = trainset.to_inner_iid(isbn)
        pred = svd_model.estimate(uid, iid)
        predictions.append((isbn, pred))
    except ValueError:
        continue
```
This is the CORE of SVD recommendations:
- For each book the user hasn't read
- Estimate how the user would rate it using the SVD model
- This uses the learned latent factors to make predictions

```python
predictions.sort(key=lambda x: x[1], reverse=True)
top_isbns = [isbn for isbn, _ in predictions[:n*2]]
```
Sorts predictions by estimated rating and takes the top books.

### How SVD Works for Recommendations
SVD is a matrix factorization technique that works like this:

#### The Magic of Matrix Factorization:
- Starts with a big, sparse matrix of user ratings (many missing values)
- Breaks it down into three smaller matrices: U, Σ (Sigma), and V^T
- These matrices capture "latent factors" - hidden patterns in the data

#### What Are Latent Factors?
- These are hidden characteristics of books and user preferences
- Example factors might include:
  - How much "fantasy" content is in a book
  - How "cerebral" vs. "action-oriented" a book is
  - How much romance or adventure a book contains
- Users also get scores for how much they like each factor

#### Making Predictions:
- If User A likes fantasy books (high score for fantasy factor)
- And Book X has a high score for the fantasy factor
- Then User A will probably like Book X, even if they've never read it

#### Advantages Over KNN:
- Handles sparse data better
- Can discover deeper patterns in preferences
- Generally scales better to large datasets
- Often provides more diverse recommendations

-----------------------------------------------------------------------------------------------------------

## NMF (Non-Negative Matrix Factorization)

### Line-by-Line Code Explanation

```python
def get_nmf_recommendations(user_id, ratings_df, books_df, n=5, progress_bar=None):
```
Function to get recommendations using NMF.

```python
user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
if len(user_ratings) == 0:
    return get_popular_books(books_df, ratings_df, n)
```
Gets user's ratings or returns popular books if none.

```python
MAX_USERS = 5000
MAX_ITEMS = 5000

user_counts = ratings_df['User-ID'].value_counts().head(MAX_USERS).index
book_counts = ratings_df['ISBN'].value_counts().head(MAX_ITEMS).index
```
Limits the dataset size for memory efficiency.

```python
subset_ratings = ratings_df[
    (ratings_df['User-ID'].isin(user_counts)) & 
    (ratings_df['ISBN'].isin(book_counts))
]
```
Creates a subset of the data with only the most active users and most-rated books.

```python
from scipy.sparse import csr_matrix
rows = [user_to_idx[user] for user in subset_ratings['User-ID']]
cols = [book_to_idx[book] for book in subset_ratings['ISBN']]
data = subset_ratings['Book-Rating'].values
ratings_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(book_ids)))
```
Converts ratings into a sparse matrix for efficiency.

```python
model = NMF(
    n_components=15,  # Fewer components
    init='random',
    random_state=42,
    max_iter=50  # Fewer iterations
)
```
Creates an NMF model with 15 latent factors.

```python
user_factors = model.fit_transform(ratings_matrix)
item_factors = model.components_
```
This is the CORE of NMF:
- Learns user factors (how much each user likes each latent factor)
- Learns item factors (how much each book represents each latent factor)

```python
user_vector = user_factors[user_idx]
predictions = []

for i in range(0, len(book_ids), BATCH_SIZE):
    batch_indices = list(range(i, min(i + BATCH_SIZE, len(book_ids))))
    batch_predictions = np.dot(user_vector, item_factors[:, batch_indices])
    
    for j, idx in enumerate(batch_indices):
        if idx not in user_rated_indices:
            isbn = book_ids[idx]
            predictions.append((isbn, batch_predictions[j]))
```
Predicts ratings by multiplying user factors with item factors.

### How NMF Differs from SVD
NMF is similar to SVD but with key differences:

#### Non-Negativity Constraint:
- As the name suggests, all values in the factorized matrices must be positive
- This makes factors more interpretable as "topics" or "themes"
- A book can't have a "negative amount" of fantasy content

#### Interpretability:
- NMF factors tend to be more interpretable than SVD factors
- Each factor might clearly correspond to a genre or theme
- e.g., Factor 3 might represent "science fiction elements"

#### When to Use It:
- When you want factors that are easier to interpret
- When you know all factors should be additive (no negative impacts)
- When dealing with non-negative data like ratings

#### Mathematical Approach:
- While SVD provides the mathematically optimal decomposition
- NMF finds a decomposition that's interpretable but might be less mathematically precise

## Collaborative Filtering (Original)

### Line-by-Line Code Explanation

```python
def get_book_recommendations_simple(user_id, ratings_df, books_df, n=5, progress_bar=None):
```
Simple collaborative filtering recommendation function.

```python
user_books = get_user_book_ratings(user_id, ratings_df, books_df)
user_rated_isbns = set(user_books['ISBN'])
```
Gets books the user has rated.

```python
common_users = ratings_df[ratings_df['ISBN'].isin(user_rated_isbns)]['User-ID'].unique()
common_users = [u for u in common_users if u != user_id]
```
Finds users who have rated at least one book that our target user has rated.

```python
if len(common_users) > 100:
    common_users = common_users[:100]
```
Limits the number of users for performance.

```python
similar_users_ratings = ratings_df[ratings_df['User-ID'].isin(common_users)]
candidate_books = similar_users_ratings[~similar_users_ratings['ISBN'].isin(user_rated_isbns)]
```
Gets books rated by similar users that our user hasn't read yet.

```python
book_stats = candidate_books.groupby('ISBN').agg(
    score=('Book-Rating', 'mean'),
    supporting_users=('User-ID', 'nunique')
).reset_index()
```
Calculates the average rating and number of users for each candidate book.

```python
book_stats = book_stats.sort_values(['score', 'supporting_users'], ascending=False)
```
Sorts books by score and number of supporting users.

### How Collaborative Filtering Works
Collaborative filtering is the foundation of recommendation systems:

#### Basic Principle:
- "People who liked what you liked will also like what you will like"
- Uses patterns in user behavior rather than content features

#### Two Main Approaches:
- USER-BASED: Find similar users, recommend what they liked (what your code does)
- ITEM-BASED: Find similar items to what the user liked

#### Your Implementation:
- Simplified user-based collaborative filtering
- Instead of calculating exact similarity scores between users
- It assumes users who rated the same books are similar

#### Strengths and Weaknesses:
- **STRENGTH:** Simple to understand and implement
- **STRENGTH:** Can recommend surprising items
- **WEAKNESS:** Cold start problem (new users, new items)
- **WEAKNESS:** Sparsity problem (most users rate very few items)

#### How It Relates to Other Models:
- KNN: A more sophisticated way to find similar users
- SVD/NMF: More advanced collaborative filtering that handles sparse data better
- Content-based models (TF-IDF, Word Embeddings, BERT): Different approach entirely

## Comparing All Models at a Glance

### Content-Based Models:
- **TF-IDF:** "Books with the same words as your favorite book"
- **Word Embeddings:** "Books with similar concepts as your favorite book"
- **BERT:** "Books with similar themes and meaning as your favorite book"

### Collaborative Filtering Models:
- **Original CF:** "Books that similar users liked"
- **KNN:** "Books liked by your 20 most similar users"
- **SVD:** "Books that match your taste profile across 50 hidden factors"
- **NMF:** "Books that match your taste profile across 15 interpretable factors"

Each model has strengths and weaknesses, and the best recommendation systems often combine multiple approaches to provide diverse, accurate recommendations!
```

This comprehensive markdown file includes all the code examples and explanations for each recommendation model. You can use this for your presentation to explain the different approaches and how they work. The file includes:

1. BERT Model - With detailed explanation of how it transforms text into vectors
2. Word Embeddings - Explanation of traditional word embeddings vs. BERT
3. KNN (K-Nearest Neighbors) - How it finds similar users
4. SVD (Singular Value Decomposition) - Matrix factorization approach
5. NMF (Non-Negative Matrix Factorization) - Similar to SVD but more interpretable 
6. Collaborative Filtering - Simple approach for finding similar users
7. Comparison of all models - Quick reference of what each model doesThis comprehensive markdown file includes all the code examples and explanations for each recommendation model. You can use this for your presentation to explain the different approaches and how they work. The file includes:

1. BERT Model - With detailed explanation of how it transforms text into vectors
2. Word Embeddings - Explanation of traditional word embeddings vs. BERT
3. KNN (K-Nearest Neighbors) - How it finds similar users
4. SVD (Singular Value Decomposition) - Matrix factorization approach
5. NMF (Non-Negative Matrix Factorization) - Similar to SVD but more interpretable 
6. Collaborative Filtering - Simple approach for finding similar users
7. Comparison of all models - Quick reference of what each model does