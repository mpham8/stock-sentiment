import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
from newspaper import Article
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np



class News:
  def __init__(self, ticker, articles):
    self.ticker = ticker
    self.articles = articles
    

  def get_news_txt(self):
    """
    This function gets the text of news articles from Finviz of a given ticker
    return: list containing each article's text as a string
    """

    finwiz_url = 'https://finviz.com/quote.ashx?t=' + self.ticker

    #URL request, get html of news table from Finviz
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(finwiz_url, headers=headers)
    
    if response.status_code != 200:
      print(f"Failed to retrieve articles for ticker {self.ticker}. HTTP Status Code: {response.status_code}")
      return
    
    print("Pulling URLs...")
    soup = BeautifulSoup(response.content, 'html.parser')
    news_table = soup.find(id='news-table')
    
    if news_table == False:
      print(f"No news table found for ticker {self.ticker}")
      return 

    #get news urls from finwiz news table
    # news_table = str(news_table)
    news_urls = []
    
    for url in news_table.find_all('a', href=True):
      news_urls.append(url['href'])
    
    news_urls = [url for url in news_urls if 'yahoo' not in url]

    
    # news_urls = news_urls[:self.articles]
    print("Found URLS:")
    for url in news_urls:
      print(url)

    #pull text from news article urls list
    news_text_ls = []
    for url in news_urls:
      try:
        if len(news_text_ls) >= self.articles:
          break
        # response = requests.get(url)
        # if response.status_code != 200:
        #   print(f"Failed to retrieve articles for url: {url}. HTTP Status Code: {response.status_code}")
        # else:
        #   # soup = BeautifulSoup(response.text, 'html.parser')
        #   # paragraphs = soup.find_all('p')
        #   # article_text = ''.join(p.get_text() for p in paragraphs)
        #   # news_text_ls.append(str(news_text_ls))
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if text != '':
          news_text_ls.append(text)


      except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve articles for url: {url}. {e}")


    return news_text_ls


class SentimentAnal:
  def __init__(self):

    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
    self.config = AutoConfig.from_pretrained(MODEL)
    self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  
  def get_sentiment_analysis(self, news_text_ls):
    cum_scores_ls = []
    for text in news_text_ls:
      tokens = self.tokenizer.tokenize(text)
      chunks = []

      for i in range(0, len(tokens), 500):
        chunk = tokens[i:i + 500]
        chunks.append(self.tokenizer.convert_tokens_to_string(chunk))

      cum_array = np.array([0,0,0], dtype=np.float64)      
      for chunk in chunks:
        encoded_input = self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        cum_array += scores
        
      avr_array = cum_array/len(chunks)
      cum_scores_ls.append(avr_array)

    
    return np.array(cum_scores_ls)
  

  def analyze_sentiment_score(self, cum_array):
    average_array = np.mean(cum_array, axis=0)
    print(cum_array)
    print(average_array)


    negative_count, neutral_count, positive_count = 0, 0, 0
    for i in range(len(cum_array)):
      index_highest = np.argmax(cum_array[i])
      if index_highest == 0:
        negative_count += 1
      elif index_highest == 2:
        positive_count += 1
      else:
        neutral_count += 1


    print(f"analyzed 10 $RIVN articles and found:")
    print(f"average sentiment: {round(average_array[0], 2)} negative, {round(average_array[1], 2)} neutral, {round(average_array[2], 2)} positive")
    print(f"negative articles: {negative_count}, neutral articles: {neutral_count}, positive articles: {positive_count}")

    
    return
  

stock_news = News("RIVN", 10)
news_text_ls = stock_news.get_news_txt()
sentiment = SentimentAnal()
cum_array = sentiment.get_sentiment_analysis(news_text_ls)
sentiment.analyze_sentiment_score(cum_array)

#Structure:
#initialize sentiment analysis model locally on website load
#News class (for each ticker) -> return all the news txt
#run in the model

