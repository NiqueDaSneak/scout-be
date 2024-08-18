import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline


def get_sentiment_score(text):
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize sentiment analyzers
    logging.info("Initializing Vader Sentiment Analyzer")
    sentiment_analyzer = SentimentIntensityAnalyzer()

    logging.info("Initializing BERT Sentiment Analyzer")
    bert_sentiment_analyzer = pipeline('sentiment-analysis')

    logging.info(f"Analyzing sentiment for text: {text[:100]}...")  # Log the first 100 characters for context
    
    # Vader sentiment analysis
    sentiment = sentiment_analyzer.polarity_scores(text)['compound']
    logging.info(f"Vader sentiment score: {sentiment}")

    # BERT sentiment analysis
    bert_sentiment = bert_sentiment_analyzer(text)[0]
    logging.info(f"BERT sentiment analysis result: {bert_sentiment}")

    if sentiment == 0.0:
        sentiment_score = bert_sentiment['score'] if bert_sentiment['label'] == 'POSITIVE' else -bert_sentiment['score']
        logging.info(f"Using BERT sentiment exclusively, score: {sentiment_score}")
    else:
        sentiment_score = (0.3 * sentiment + 0.7 * (1 if bert_sentiment['label'] == 'POSITIVE' else -1)) / 2
        logging.info(f"Combined sentiment score: {sentiment_score}")

    return sentiment_score
