from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import re
import requests
import nltk
from nltk.corpus import cmudict
# nltk.download('punkt')
# nltk.download('cmudict')
from langdetect import detect
import os
from dotenv import load_dotenv


def init():
    '''Initialize the environment.'''
    load_dotenv()

    global spotify_cid
    global spotify_secret
    global genius_token
    global headers

    spotify_cid = os.getenv("SPOTIFY_CID")
    spotify_secret = os.getenv("SPOTIFY_SECRET")
    genius_token = os.getenv("GENIUS_TOKEN")
    headers = {"Authorization": "Bearer " + genius_token}


# Cache

CACHE_FILENAME = "cache.json"

def openCache():
    '''Check if cache file exists, if so load it, if not create new cache'''
    try:
        cache_file = open(CACHE_FILENAME, "r")
        cache_contents = cache_file.read()
        cache_dict = json.loads(cache_contents)
        cache_file.close()
    except:
        cache_dict = {}
    return cache_dict

def saveCache(cache_dict):
    '''Save cache file'''
    cache_file = open(CACHE_FILENAME, "w")
    contents_to_write = json.dumps(cache_dict)
    cache_file.write(contents_to_write)
    cache_file.close()


# Billboard scraper

def scrapeBillboard(date):
    '''
    Scrape the Billboard Hot 100 chart for a given date.
    
    Parameters:
        date (datetime.date): The date of the chart.
        
    Returns:
        list: A list of tuples containing the title and artist of each song.
    '''
    url = "https://www.billboard.com/charts/hot-100/" + str(date) + "/"
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")

    ul = soup.findAll("ul", class_="o-chart-results-list-row")

    billboard = []
    for i in ul:
        title = i.find("h3").text.strip()
        artist = i.find("span", class_="a-font-primary-s").text.strip()
        billboard.append((title, artist))

    return billboard


# Spotify API

def getSpotifyToken():
    '''Get the Spotify access token.'''
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
        "grant_type": "client_credentials",
        "client_id": spotify_cid,
        "client_secret": spotify_secret,
    }).json()

    return response["access_token"]

def getSpotifyID(token, title, artist):
    '''Get the Spotify ID of a song.'''
    headers = {"Authorization": "Bearer " + token}
    url = f"https://api.spotify.com/v1/search?q={title}%20{artist}&type=track&market=US&limit=1"
    response = requests.get(url, headers=headers).json()
    return response["tracks"]["items"][0]["id"]

def getSpotifyFeatures(token, song_id):
    '''Get the Spotify features of a song.'''
    headers = {"Authorization": "Bearer " + token}
    url = f"https://api.spotify.com/v1/audio-features/{song_id}"
    response = requests.get(url, headers=headers)
    return response.json()


# Genius API

def getGeniusURL(title, artist):
    '''Get the Genius URL of a song.'''
    url = "https://api.genius.com/search"
    params = {"q": f"{title} {artist}"}
    response = requests.get(url, params=params, headers=headers).json()
    return response["response"]["hits"][0]["result"]["url"]

def getLyrics(url):
    '''Get the lyrics of a song from its Genius URL.'''
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    lyrics = soup.find("div", {"data-lyrics-container": "true"}).get_text(separator="\n") 
    return lyrics


# Readability metrics

def countSyllables(word):
    '''Count the number of syllables in a word.'''
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    if word[0] in vowels:
        count +=1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count +=1
    return count

def getFRES(lyrics):
    '''Calculate the Flesch reading-ease score (FRES) of a song.'''
    # Remove [Verse], [Chorus], etc.
    lyrics = re.sub(r"\[.*\]", "", lyrics)
    sentence = lyrics.split("\n")
    sentence = [i for i in sentence if i]
    word = lyrics.split()
    word_count = len(word)
    sentence_count = len(sentence)
    syllable_count = sum([countSyllables(token) for token in word])
    return 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllable_count / word_count))


def vocabComplex(lyrics):
    '''Calculate the ratio of different unique word stems (types) to the total number of words (tokens).'''
    tokens = nltk.word_tokenize(lyrics.lower())
    return len(set(tokens)) / len(tokens)


def sentenceLength(lyrics):
    '''Calculate the average number of words in a sentence.'''
    sentences = nltk.sent_tokenize(lyrics)
    total_words = sum(len(nltk.word_tokenize(sent)) for sent in sentences)
    return total_words / len(sentences)


def avgSyllable(lyrics):
    """Calculate the average number of syllables per word."""
    d = cmudict.dict()
    words = lyrics.split()
    total_syllables = sum(len(d[word.lower()][0]) for word in words if word.lower() in d)
    return total_syllables / len(words)


# Data consolidation


def addAllFeatures(dataset, billboard):
    '''
    Add new songs on the Billboard Hot 100 to the dataset with all features including lyrics.

    Parameters:
        dataset (dict): The dataset.
        billboard (list): The list of songs on the Billboard Hot 100.
        
    Returns:
        dict: The dataset with lyrics.
    '''
    spotify_token = getSpotifyToken()

    for title, artist in billboard:
        
        # Skip if the song is already in the dataset
        abbrev = title.replace("_", " ") + "_" + artist.replace("_", " ")
        if abbrev in dataset["data"]:
            continue

        try:
            # Get the Spotify features, Genius lyrics, and FRES
            print("Running: ", abbrev)
            spotify_id = getSpotifyID(spotify_token, title, artist)
            features = getSpotifyFeatures(getSpotifyToken(), spotify_id)
            genius_url = getGeniusURL(title, artist)
            lyrics = getLyrics(genius_url)
            features["fres"] = getFRES(lyrics)
            features["vocabComplex"] = vocabComplex(lyrics)
            features["sentenceLength"] = sentenceLength(lyrics)
            features["avgSyllable"] = avgSyllable(lyrics)
            features["lyrics"] = lyrics
            features["title"] = title.replace("_", " ")
            features["artist"] = artist.replace("_", " ")
            features["lang"] = detect(lyrics)
        except:
            # Skip if the song is not found on Spotify or Genius
            print("Not found: ", abbrev)
            continue

        # Add the song with features to the dataset
        dataset["data"][abbrev] = features
    return dataset


def updateCache():
    '''Update the dataset with new songs on the Billboard Hot 100.'''

    dataset = openCache()
    # Billboard Hot 100 is updated every Saturday
    today = datetime.today().date()
    saturday = today + timedelta(days=5-today.weekday())

    # If cache is empty, add all songs on the Billboard Hot 100 from the past year to the dataset
    if dataset == {}:
        dataset["updated_week"] = str(saturday)
        dataset["data"] = {}
        billboard = []

        for i in range(52):
            billboard.extend(scrapeBillboard(saturday))
            saturday -= timedelta(days=7)

        billboard = list(set(billboard))
        dataset = addAllFeatures(dataset, billboard)
        saveCache(dataset)
    
    # If cache is not empty, check if the dataset is up to date
    else:
        # If not, updated new songs from the last updated week to the current week to the dataset
        if dataset["updated_week"] != str(saturday):
            last_updated = dataset["updated_week"]
            dataset["updated_week"] = str(saturday)
            billboard = []

            while str(saturday) != last_updated:
                billboard.extend(scrapeBillboard(saturday))
                saturday -= timedelta(days=7)

            billboard = list(set(billboard))
            dataset = addAllFeatures(dataset, billboard)
            saveCache(dataset)
        else:
            print("Dataset is up to date.")

    print("Data retrieved: ", len(dataset["data"]))
    print("Data sample: ", dataset["data"]["Houdini_Dua Lipa"])


def exportData():
    '''Export the dataset to a JSON file.'''
    
    dataset = openCache()
    data = dataset["data"]
    
    # Prepare data by selecting only specific attributes for each song
    filtered_data = []
    for song in data:
        features = data[song]
        filtered_data.append({
            "id": features['id'],
            "title": features['title'],
            "artist": features['artist'],
            "danceability": features['danceability'],
            "valence": features['valence'],
            "speechiness": features['speechiness'],
            "fres": features['fres'],
            "vocabComplex": features['vocabComplex'],
            "sentenceLength": features['sentenceLength'],
            "avgSyllable": features['avgSyllable'],
            "lyrics": features['lyrics'],
            "lang": features["lang"]
        })
    
    # Export the filtered data to a JSON file
    with open("data.json", "w") as file:
        json.dump(filtered_data, file, indent=4)


if __name__ == "__main__":
    init()
    updateCache()
    exportData()