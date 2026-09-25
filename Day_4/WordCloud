!pip install wordcloud


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Provided text
text = """
gaming has never been about passive distraction—it’s an ongoing dialogue with discipline, mechanics, and mental grit. Whether it’s the unforgiving rhythm of a blade deflection where hesitation guarantees defeat, or the quiet tension of holding an angle when everything depends on trigger discipline and game sense, it demands total presence. You can’t fake your way through genuine difficulty; you either study the patterns, swallow the frustration of repeated failure, and adapt, or you stay stuck.

That mindset bleeds straight into how I operate outside the screen. Breaking down a lethal boss encounter or solving an intricate gameplay loop isn't fundamentally different from picking apart complex code, dissecting a system logic error, or building something functional from scratch. It’s the thrill of reverse-engineering an obstacle until what once felt impossible becomes pure muscle memory.

At the same time, it’s my sanctuary. When the noise of deadlines, academics, and expectations piles high, gripping a controller or setting hands on the keys resets the board. In those worlds, effort yields direct, honest results. It reflects the core of who I am: someone who respects the grind, finds beauty in high-stakes precision, and never backs down from a challenge simply because it hits back hard..
"""
print(text)

# Tokenization of words
words = word_tokenize(text.lower())  # Convert to lowercase for uniformity
print(words)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
# The general syntax for a list comprehension is: [expression for item in iterable if condition]
print(filtered_words)

# Create a WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()
