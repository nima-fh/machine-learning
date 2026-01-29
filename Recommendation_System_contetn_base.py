import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt

movie=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\movies.csv")

rating=pd.read_csv("G:\work\Data science\machine_learning\machine_learning_with_python_jadi-main\\ratings_sample.csv")

print(movie.head(10))
print(rating.head(10))

movie["year"]=movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
movie["year"]=movie.year.str.extract("(\d\d\d\d)",expand=False)
movie["title"]=movie.title.str.replace('(\(\d\d\d\d\))',"")
movie['title'] = movie['title'].apply(lambda x: x.strip())
movie["genres"]=movie.genres.str.split("|")
print(movie.head(10))

movieWithGenres=movie.copy()

for i,r in movieWithGenres.iterrows():
    for genre in r["genres"]:
        movieWithGenres.at[i,genre]=1
        
movieWithGenres=movieWithGenres.fillna(0)
print(movieWithGenres)

rating=rating.drop("timestamp",axis=1)
print(rating.head())

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovie=pd.DataFrame(userInput)
print(inputMovie)
inputId = movie[movie['title'].isin(inputMovie['title'].tolist())]
inputMovie=pd.merge(inputId,inputMovie)
inputMovie=inputMovie.drop("genres",axis=1).drop("year",axis=1)
print(inputMovie)