from textblob import TextBlob
from newspaper import Article

url = 'https://www.orientsoftware.com/blog/gpt3-open-source/'
article = Article(url)

article.download()
article.parse()
article.nlp()

text=article.summary
print(text)

blob=TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)