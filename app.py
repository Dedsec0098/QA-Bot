# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import torch
from flask_cors import CORS
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from threading import Lock
import logging

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model_lock = Lock()

def load_model():
    global model, tokenizer
    try:
        with model_lock:
            if model is None:
                app.logger.info("Loading model...")
                # Load model directly instead of from pickle
                tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
                model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
                app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        raise

load_model()  # Load model at startup

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        question = data.get('question')
        if not question:
            return jsonify({'error': 'Empty question'}), 400
            
        app.logger.info(f"Processing question: {question[:50]}...")
        
        # Tokenize input
        try:
            inputs = tokenizer(question, return_tensors="pt")
            
            with torch.no_grad():
                # Set a reasonable max token limit to prevent OOM errors
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,  # Reduced from 200
                    temperature=0.2,
                    do_sample=False,
                    num_beams=1  # Simple greedy decoding is faster
            )
                
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            app.logger.info("Generated answer successfully")
            
            return jsonify({
                'question': question,
                'answer': answer
            })
        except torch.cuda.OutOfMemoryError:
            app.logger.error("CUDA out of memory error")
            return jsonify({'error': 'Server resource limit exceeded'}), 500
        except Exception as e:
            app.logger.error(f"Model prediction error: {str(e)}")
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 500
            
    except Exception as e:
        app.logger.error(f"Request handling error: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)