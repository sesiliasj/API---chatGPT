from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

ai_detector = pipeline('text-classification', model='roberta-base-openai-detector')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = ai_detector(text)
    
    is_ai_generated = result[0]['label'] == 'LABEL_1'
    confidence = result[0]['score']

    return jsonify({
        'is_ai_generated': is_ai_generated,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)