import logging
from flask import Blueprint, request, jsonify
from .utils.scoring import generate_score, generate_nuanced_feedback

# Configure logging
logging.basicConfig(level=logging.INFO)

main = Blueprint('main', __name__)

@main.route('/process', methods=['POST'])
def process_data():
    logging.info("Received a request to /process endpoint.")
    data = request.json
    logging.info(f"Received data: {data}")  
    
    data_type = data.get('dataType')
    data_body = data.get('dataBody')
    categories = data.get('categories', '').split(',')

    if data_type and data_body:
        logging.info(f"Processing data with dataType: {data_type}, dataBody: {len(data_body)} characters, categories: {categories}")
        
        # Perform content scoring based on dataType
        score_result = generate_score(data_body, categories, data_type)
        
        logging.info(f"Generated score result: {score_result}")
        
        response = {
            'quality': score_result['quality'],
            'recommendations': score_result['recommendations'],
            'nuanced_feedback': score_result['nuanced_feedback'],
            'final_score': score_result['final_score']
        }

        logging.info("Sending response back to the client.")
        return jsonify(response), 200
    else:
        logging.error("Invalid input received.")
        return jsonify({'error': 'Invalid input'}), 400
