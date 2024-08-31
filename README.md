# Movie Recommendation System


This repository contains a movie recommendation system built using Python. The system uses a combination of data preprocessing, feature engineering, and machine learning techniques to recommend movies based on their genres, keywords, cast, crew, and overview.

## Project Overview

The project involves the following key steps:

1. **Dataset Loading:**
   - Loaded two datasets: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
   - Merged the datasets on the `title` attribute.

2. **Data Preprocessing:**
   - Extracted relevant columns: `genres`, `keywords`, `overview`, `title`, `movie_id`, `cast`, `crew`.
   - Handled missing and duplicated data.
   - Converted stringified JSON objects to Python lists for `genres`, `keywords`, `cast`, and `crew`.
   - Processed and cleaned the data to remove spaces and convert text to lowercase.

3. **Feature Engineering:**
   - Created a `tags` column by combining `genres`, `keywords`, `overview`, `cast`, and `crew`.
   - Applied stemming to reduce words to their root form.

4. **Vectorization:**
   - Used `CountVectorizer` to convert the `tags` column into a matrix of token counts.
   - Limited the features to the top 5000 words and removed English stop words.

5. **Cosine Similarity:**
   - Calculated the cosine similarity between movie vectors to determine how similar they are.

6. **Movie Recommendation Function:**
   - Built a function to recommend movies based on the cosine similarity of the selected movie's vector.

## Setup and Installation

To run this project, you'll need Python installed along with the following libraries:

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from nltk.stem.porter import PorterStemmer
```

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn nltk
```

## Steps to Run the Project

### 1. Load the Datasets
```python
mv = pd.read_csv('path_to_tmdb_5000_movies.csv')
cd = pd.read_csv('path_to_tmdb_5000_credits.csv')
```

### 2. Data Preprocessing
- Merging datasets on the `title` column.
- Selecting relevant columns.
- Handling missing and duplicated data.

### 3. Feature Engineering
- Converting stringified JSON objects to lists.
- Extracting genres, keywords, cast, and crew information.
- Creating a combined `tags` column.

### 4. Vectorization
```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(ndf['tags']).toarray()
```

### 5. Cosine Similarity
```python
similarity = cosine_similarity(vector)
```

### 6. Movie Recommendation Function
```python
def Movie_recommend(movie):
    movie_index = ndf[ndf['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]
    for i in movie_list:
        print(ndf.iloc[i[0]].title)
```

### 7. Example Usage
```python
Movie_recommend('Batman Begins')
```

## Summary

This project demonstrates how to build a basic movie recommendation system using text processing and machine learning techniques. The system can recommend movies similar to a given movie based on various features such as genres, keywords, cast, crew, and overview.
