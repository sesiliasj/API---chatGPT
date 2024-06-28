from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Setel API key OpenAI Anda
openai.api_key = 'secret'

@app.route('/detect-ai-text', methods=['POST'])
def detect_ai_text():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Teks tidak ditemukan dalam request'}), 400
    
    text = data['text']
    
    try:
        # Menggunakan OpenAI ChatCompletion untuk mendeteksi teks yang dihasilkan AI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Anda bisa memilih model lain seperti "gpt-4" jika diperlukan
            messages=[
                {"role": "system", "content": "You are a detector of AI-generated text."},
                {"role": "user", "content": f"Is the following text generated by AI? '{text}'"}
            ]
        )
        
        # Ambil jawaban dari respon model
        result = response['choices'][0]['message']['content'].strip()
        
        # Logika sederhana untuk menentukan persentase deteksi
        if "yes" in result.lower() or "likely" in result.lower():
            ai_percentage = 90  # Asumsi deteksi AI tinggi
        else:
            ai_percentage = 10  # Asumsi deteksi AI rendah
        
        return jsonify({'ai_percentage': ai_percentage})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Jalankan server pada host 0.0.0.0 dan port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
