import spacy
import requests

def get_synonyms(word):
    # Replace 'https://api.datamuse.com/words' with the appropriate endpoint if using a different API
    response = requests.get(f"https://api.datamuse.com/words?rel_syn={word}")
    if response.status_code == 200:
        synonyms = [entry['word'] for entry in response.json()]
        return synonyms
    else:
        return []

# Load a pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

def calculate_keyword_density(text, categories):
    # Split the categories string into a list
    category_list = [category.strip().lower() for category in categories]
    
    # Initialize a dictionary to hold counts for each category and its synonyms
    keyword_counts = {}

    for category in category_list:
        # Initialize the count for the category
        keyword_counts[category] = 0
        
        # Fetch synonyms for the category
        synonyms = get_synonyms(category)
        
        # Add the synonyms to the count dictionary with initial counts of 0
        for synonym in synonyms:
            keyword_counts[synonym] = 0

    # print(f"Category list with synonyms: {keyword_counts.keys()}")
    
    # Analyze the text with SpaCy
    doc = nlp(text.lower())

    # Count the occurrences of each category and its synonyms
    for token in doc:
        if token.text in keyword_counts:
            keyword_counts[token.text] += 1

    # print(f"Keyword counts: {keyword_counts}")

    # Calculate the density score (in this case, simply the count of any keyword or synonym appearing)
    density_score = sum(1 for count in keyword_counts.values() if count > 0)
    
    # Normalize the density score by the number of categories (or keywords + synonyms)
    normalized_score = density_score / len(category_list) if category_list else 0

    # print(f"Density score: {density_score}")
    # print(f"Normalized score: {normalized_score}")

    return normalized_score

def generate_recommendations(readability_score, length_score, keyword_density_score, sentiment_score):
    recommendations = []
    if readability_score < 60:
        recommendations.append('Improve readability')
    if length_score == 0:
        recommendations.append('Adjust length to optimal range')
    if keyword_density_score == 0:
        recommendations.append('Optimize keyword density')
    if sentiment_score < 0:
        recommendations.append('Improve sentiment')
    return recommendations
