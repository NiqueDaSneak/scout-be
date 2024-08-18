from flask import Blueprint, request, jsonify
from .utils.scoring import generate_score, generate_nuanced_feedback

main = Blueprint('main', __name__)

@main.route('/process', methods=['POST'])
def process_data():
    data = request.json
    print(f"Received data: {data}")  
    data_type = data.get('dataType')
    data_body = data.get('dataBody')
    categories = data.get('categories', '').split(',')

    # Perform content scoring based on dataType
    if data_type and data_body:
        score_result = generate_score(data_body, categories, data_type)

        response = {
            'quality': score_result['quality'],
            'recommendations': score_result['recommendations'],
            'nuanced_feedback': score_result['nuanced_feedback'],
            'final_score': score_result['final_score']
        }

        return jsonify(response), 200
    else:
        return jsonify({'error': 'Invalid input'}), 400