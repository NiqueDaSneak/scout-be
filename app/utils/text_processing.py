import spacy
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_synonyms(word):
    logging.info(f"Fetching synonyms for the word: {word}")
    
    # Replace 'https://api.datamuse.com/words' with the appropriate endpoint if using a different API
    response = requests.get(f"https://api.datamuse.com/words?rel_syn={word}")
    
    if response.status_code == 200:
        synonyms = [entry['word'] for entry in response.json()]
        logging.info(f"Synonyms found for '{word}': {synonyms}")
        return synonyms
    else:
        logging.error(f"Failed to fetch synonyms for '{word}'. Status code: {response.status_code}")
        return []

# Load a pre-trained SpaCy model
logging.info("Loading SpaCy model 'en_core_web_sm'")
nlp = spacy.load('en_core_web_sm')

def calculate_keyword_density(text, categories):
    logging.info(f"Calculating keyword density for text with categories: {categories}")
    
    # Split the categories string into a list
    category_list = [category.strip().lower() for category in categories]
    
    # Initialize a dictionary to hold counts for each category and its synonyms
    keyword_counts = {}

    for category in category_list:
        logging.info(f"Processing category: {category}")
        
        # Initialize the count for the category
        keyword_counts[category] = 0
        
        # Fetch synonyms for the category
        synonyms = get_synonyms(category)
        
        # Add the synonyms to the count dictionary with initial counts of 0
        for synonym in synonyms:
            keyword_counts[synonym] = 0

    logging.info(f"Category list with synonyms: {list(keyword_counts.keys())}")
    
    # Analyze the text with SpaCy
    doc = nlp(text.lower())

    # Count the occurrences of each category and its synonyms
    for token in doc:
        if token.text in keyword_counts:
            keyword_counts[token.text] += 1

    logging.info(f"Keyword counts: {keyword_counts}")

    # Calculate the density score (in this case, simply the count of any keyword or synonym appearing)
    density_score = sum(1 for count in keyword_counts.values() if count > 0)
    
    # Normalize the density score by the number of categories (or keywords + synonyms)
    normalized_score = density_score / len(category_list) if category_list else 0

    logging.info(f"Density score: {density_score}")
    logging.info(f"Normalized score: {normalized_score}")

    return normalized_score

def generate_recommendations(readability_score, length_score, keyword_density_score, sentiment_score):
    logging.info(f"Generating recommendations based on scores - Readability: {readability_score}, Length: {length_score}, Keyword Density: {keyword_density_score}, Sentiment: {sentiment_score}")
    
    recommendations = []
    if readability_score < 60:
        recommendations.append('Improve readability')
        logging.info("Added recommendation: Improve readability")
    if length_score == 0:
        recommendations.append('Adjust length to optimal range')
        logging.info("Added recommendation: Adjust length to optimal range")
    if keyword_density_score == 0:
        recommendations.append('Optimize keyword density')
        logging.info("Added recommendation: Optimize keyword density")
    if sentiment_score < 0:
        recommendations.append('Improve sentiment')
        logging.info("Added recommendation: Improve sentiment")
    
    logging.info(f"Final recommendations: {recommendations}")
    return recommendations
