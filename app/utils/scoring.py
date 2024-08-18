import os
import textstat
import torch
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from .sentiment import get_sentiment_score
from .text_processing import calculate_keyword_density, generate_recommendations, get_synonyms

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_nuanced_feedback(readability_score, keyword_density_score, sentiment_score, categories, text):
    print(f"all inputs: {readability_score}, {keyword_density_score}, {sentiment_score}, {categories}, {text}")
    # Expand categories with synonyms
    expanded_categories = categories + [syn for category in categories for syn in get_synonyms(category)]
    print(f"expanded_categories: {expanded_categories}")

    prompt = (
        "You are a marketing consultant specializing in crowdfunding campaigns. "
        "Given the following input:\n"
        f"Readability score: {readability_score:.2f}, Keyword density score: {keyword_density_score:.2f}, Sentiment score: {sentiment_score:.2f}, "
        f"Categories: {', '.join(expanded_categories)}. \n"
        f"Text excerpt: '{text[:100]}...'. \n\n"
        "Based on this input, provide 3 specific and actionable recommendations to improve the effectiveness of this crowdfunding campaign's copy. "
        "The recommendations should focus on: 1) Enhancing clarity and readability, 2) Increasing emotional appeal, 3) Ensuring alignment with campaign goals. "
        "Please structure your response as follows:\n"
        "1.\n2.\n3."
    )

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = inputs.to(device)

    # Generate the attention mask
    attention_mask = inputs['attention_mask'].to(device)

    # Pass the attention mask to the model
    outputs = model.generate(
        inputs['input_ids'], 
        max_new_tokens=150,  # Limit the generation to 150 new tokens
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        temperature=0.8, 
        top_k=50, 
        do_sample=True,  # Enable sampling
        attention_mask=attention_mask  # Pass the attention mask to avoid the warning
    )

    # Decode and extract the recommendations
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Find the recommendations part of the text
    recommendations_start = generated_text.find("1.")
    recommendations = generated_text[recommendations_start:].strip() if recommendations_start != -1 else generated_text.strip()

    # Clean up the text
    recommendations = recommendations.replace("\n", " ").replace("2.", "\n2.").replace("3.", "\n3.")
    return recommendations

def generate_score(text, categories, data_type):
    readability_score = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    
    # Adjust length_score based on data_type
    if data_type == 'heading':
        length_score = 1 if 50 <= len(text) <= 60 else 0
    elif data_type == 'subheading':
        length_score = 1 if 150 <= len(text) <= 160 else 0
    else:  # Assume 'story' or other longer texts
        length_score = 1 if len(text) > 160 else 0

    keyword_density_score = calculate_keyword_density(text, categories)
    sentiment_score = get_sentiment_score(text)

    final_score = (readability_score / 100) + length_score + keyword_density_score + sentiment_score
    
    # Pass the required parameters to the LLM feedback function
    nuanced_feedback = generate_nuanced_feedback(
        readability_score, 
        keyword_density_score, 
        sentiment_score, 
        categories, 
        text
    )

    return {
        'quality': 'high' if final_score > 2.5 else 'medium' if final_score > 1.5 else 'low',
        'recommendations': generate_recommendations(readability_score, length_score, keyword_density_score, sentiment_score),
        'nuanced_feedback': nuanced_feedback,
        'final_score': final_score
    }
