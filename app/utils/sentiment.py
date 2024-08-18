from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

sentiment_analyzer = SentimentIntensityAnalyzer()
bert_sentiment_analyzer = pipeline('sentiment-analysis')

def get_sentiment_score(text):
    sentiment = sentiment_analyzer.polarity_scores(text)['compound']
    # print(f"Vader sentiment score: {sentiment}")  

    bert_sentiment = bert_sentiment_analyzer(text)[0]
    # print(f"BERT sentiment score: {bert_sentiment}")  

    if sentiment == 0.0:
        sentiment_score = bert_sentiment['score'] if bert_sentiment['label'] == 'POSITIVE' else -bert_sentiment['score']
        # print(f"Using BERT sentiment exclusively: {sentiment_score}")
    else:
        sentiment_score = (0.3 * sentiment + 0.7 * (1 if bert_sentiment['label'] == 'POSITIVE' else -1)) / 2
        # print(f"Combined sentiment score: {sentiment_score}")

    return sentiment_score

