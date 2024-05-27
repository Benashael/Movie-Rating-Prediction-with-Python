import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="IMDb Movies India Rating Prediction", page_icon="🎬")

st.title("IMDb Movies India Rating Prediction 🎬")

# Load the dataset
def load_data():
    df = pd.read_csv('IMDb-Movies-India.csv')
    return df

df = load_data()

# Display the dataset
st.header("IMDb Movies India Dataset")
st.write(df)

# Select features and target variable
@st.cache
def preprocess_data(df):
    # Trim the number of unique values for encoding
    top_directors = df['Director'].value_counts().index[:10]
    top_actors = df['Actor 1'].value_counts().index[:10]
    
    df['Director'] = df['Director'].where(df['Director'].isin(top_directors), 'Other')
    df['Actor 1'] = df['Actor 1'].where(df['Actor 1'].isin(top_actors), 'Other')
    df['Actor 2'] = df['Actor 2'].where(df['Actor 2'].isin(top_actors), 'Other')
    df['Actor 3'] = df['Actor 3'].where(df['Actor 3'].isin(top_actors), 'Other')
    
    features = ['Year', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target = 'Rating'
    
    X = df[features]
    y = df[target]
    
    return X, y

X, y = preprocess_data(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the preprocessing and model pipeline
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
numeric_features = ['Year', 'Votes']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

# User input features
st.header("User Input Features")

# Options
def user_input_features():
    year = st.slider('Year', int(df.Year.min()), int(df.Year.max()), int(df.Year.mean()))
    genre = st.selectbox('Genre', options=df['Genre'].unique())
    votes = st.slider('Votes', int(df.Votes.min()), int(df.Votes.max()), int(df.Votes.mean()))
    director = st.selectbox('Director', options=df['Director'].unique())
    actor_1 = st.selectbox('Actor 1', options=df['Actor 1'].unique())
    actor_2 = st.selectbox('Actor 2', options=df['Actor 2'].unique())
    actor_3 = st.selectbox('Actor 3', options=df['Actor 3'].unique())
    return {
        'Year': year,
        'Genre': genre,
        'Votes': votes,
        'Director': director,
        'Actor 1': actor_1,
        'Actor 2': actor_2,
        'Actor 3': actor_3
    }

input_df = pd.DataFrame(user_input_features(), index=[0])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f"Predicted Rating: **{prediction[0]:.2f}**")

