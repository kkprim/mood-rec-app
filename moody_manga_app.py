# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

#import os
#import sys

# Path to your virtual environment's Python executable
#venv_path = '/Users/kariprimiano/anaconda3/envs/myenv/bin/python'

# Activate the virtual environment
#if os.path.exists(venv_path):
    #os.execl(venv_path, venv_path, *sys.argv)

LOGGER = get_logger(__name__)
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from keras.models import Model
from keras.layers import Embedding, Flatten, Input, concatenate, Dense, Dropout
from keras.optimizers import Adam
from tensorflow import keras
from keras.regularizers import l1_l2
from keras import backend as K
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from keras.metrics import MeanAbsoluteError

st.image("images/custom_anime_header2.png", caption='Image by DALL-E', use_column_width=True)  
import pickle

# Define file paths for the pre-saved models and preprocessors
mood_classifier_path = 'saved_models/mood_classifier.pkl'
rating_predictor_path = 'saved_models/rating_predictor.pkl'
rfc_preprocessor_path = 'saved_models/rfc_preprocessor.pkl'
hybrid_preprocessor_path = 'saved_models/hybrid_preprocessor.pkl'

# Load the pre-saved models and preprocessors
with open(mood_classifier_path, 'rb') as mood_file:
    mood_classifier = pickle.load(mood_file)

with open(rating_predictor_path, 'rb') as rating_file:
    rating_predictor = pickle.load(rating_file)

with open(rfc_preprocessor_path, 'rb') as rfc_preprocessor_file:
    rfc_preprocessor = pickle.load(rfc_preprocessor_file)

with open(hybrid_preprocessor_path, 'rb') as hybrid_preprocessor_file:
    hybrid_preprocessor = pickle.load(hybrid_preprocessor_file)

# Load dataset
anime_data = pd.read_csv('zipped_data/data_with_emotions.csv')

# Ensure 'gender' is categorical
anime_data['gender'] = anime_data['gender'].astype('category')

def sort_anime_based_on_predicted_ratings(anime_data, user_age, user_gender, rating_predictor, hybrid_preprocessor):
    # Replace NaN values in 'synopsis' with a placeholder string (e.g., "")
    anime_data['synopsis'].fillna("", inplace=True)
    
    # Prepare additional input data
    additional_input_data = pd.DataFrame({
        'synopsis': anime_data['synopsis'],
        'age': [user_age] * len(anime_data),
        'gender': [user_gender] * len(anime_data)
    })

    # Preprocess the additional input data
    additional_input_transformed = hybrid_preprocessor.transform(additional_input_data).todense()

    # Extract user and item IDs from the anime data
    user_ids = anime_data['uid_review'].astype('category').cat.codes.values
    item_ids = anime_data['anime_uid'].astype('category').cat.codes.values

    # Predict ratings using the rating predictor model
    predicted_ratings = rating_predictor.predict([user_ids, item_ids, additional_input_transformed])

    # Add the predicted ratings to the anime data
    anime_data['predicted_rating'] = predicted_ratings

    # Sort the anime based on predicted ratings
    sorted_anime = anime_data.sort_values(by='predicted_rating', ascending=False)

    return sorted_anime


def generate_playlist(mood, age, gender, anime_data, mood_classifier, rating_predictor, rfc_preprocessor, rating_preprocessor, shuffle=False):

    # Predict the mood for each anime
    features = anime_data[['genre', 'popularity']]
    features_transformed = rfc_preprocessor.transform(features)
    anime_data['predicted_mood'] = mood_classifier.predict(features_transformed)

    # Implement mood-based filtering
    #genres_based_on_mood = mood_to_genre_mapping.get(mood, ['Default Genre'])
    #average_popularity = anime_data[anime_data['genre'].isin(genres_based_on_mood)]['popularity'].mean()
    #genres_str = ', '.join(genres_based_on_mood)
    #mood_input_df = pd.DataFrame({'genre': [genres_str], 'popularity': [average_popularity]})
    #mood_input_transformed = rfc_preprocessor.transform(mood_input_df)
    #predicted_mood_category = mood_classifier.predict(mood_input_transformed)[0]
    filtered_anime = anime_data[anime_data['predicted_mood'] == mood]

    # Drop duplicates based on 'title' to ensure unique anime titles
    filtered_anime = filtered_anime.drop_duplicates(subset='title', keep='first')

    sorted_anime = sort_anime_based_on_predicted_ratings(filtered_anime, age, gender, rating_predictor, hybrid_preprocessor)

    # Additional step to select specific columns
    playlist = sorted_anime[['title', 'synopsis', 'Overall', 'Story', 'Animation', 'Sound', 'Character', 'Enjoyment', 'predicted_mood', 'predicted_rating']].copy()

    # Return the top N anime as the playlist
    return playlist.head(10)


# Streamlit UI
st.title('MoodyManga')
st.subheader('Mood-Based Playlist Generator', divider='rainbow')

# User inputs
user_mood = st.selectbox('Select your mood', ['Happy', 'Sad', 'Excited', 'Angry', 'Fear', 'Neutral'])
user_age = st.number_input('Enter your age', min_value=0, max_value=100, value=25)
user_gender = st.selectbox('Select your gender', ['Male', 'Female', 'Non-Binary'])
#shuffle_playlist = st.checkbox('Shuffle Playlist')

# Button to generate playlist
if st.button('Generate Playlist'):
    playlist = generate_playlist(user_mood, user_age, user_gender, anime_data, mood_classifier, rating_predictor, rfc_preprocessor, hybrid_preprocessor)    
    # Display playlist with Styler
    st.write(playlist.style.set_properties(**{'background-color': '#36454F', 'border': '1px solid black'}))

# Social sharing and like buttons for each anime
    for index, anime in playlist.iterrows():
        anime_title = anime['title']
        share_text = f"Check out mood-based recommendation {anime_title} on MoodyManga!"
        twitter_url = f"https://twitter.com/intent/tweet?text={share_text}"

        col1, col2 = st.columns([0.8, 0.5])
        with col1:
            st.write(anime_title)
        with col2:
            # Directly open Twitter URL when the button is clicked
            st.markdown(f'<a href="{twitter_url}" target="_blank"><button style="width:100%;">Share on Twitter</button></a>', unsafe_allow_html=True)

st.divider()

st.markdown("""
<footer style='text-align: right;'>
    Made with ❤️ by Gina & Kari
</footer>
""", unsafe_allow_html=True)