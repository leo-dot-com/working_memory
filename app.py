from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import speech_recognition as sr
import pyttsx3
import openai
import os
import tempfile
import wave
import io
import base64
from pydub import AudioSegment
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.8)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'working_memory_assessment',
        'tts_available': True,
        'stt_available': True
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        
        # Read the generated audio file
        with open(temp_filename, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Convert to base64 for easy transmission
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        return jsonify({
            'success': True,
            'audio_url': f"data:audio/wav;base64,{audio_base64}",
            'text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stt', methods=['POST'])
def speech_to_text():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        expected_response = request.form.get('expected', '')
        task_type = request.form.get('task_type', '')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            audio_file.save(temp_file.name)
            temp_filename = temp_file.name
        
        try:
            # Convert audio to WAV format for speech recognition
            audio = AudioSegment.from_file(temp_filename)
            wav_data = io.BytesIO()
            audio.export(wav_data, format='wav')
            wav_data.seek(0)
            
            # Use speech recognition
            with sr.AudioFile(wav_data) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # Try Google Speech Recognition
                try:
                    transcription = recognizer.recognize_google(audio_data)
                    confidence = 0.8  # Google doesn't provide confidence scores
                    
                except sr.UnknownValueError:
                    transcription = ""
                    confidence = 0.0
                except sr.RequestError as e:
                    return jsonify({'error': f'Speech recognition service error: {e}'}), 500
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            # Analyze response accuracy based on task type
            is_correct, analysis_confidence = analyze_response_accuracy(
                transcription, expected_response, task_type
            )
            
            # Use the higher confidence value
            final_confidence = max(confidence, analysis_confidence)
            
            return jsonify({
                'success': True,
                'transcription': transcription,
                'is_correct': is_correct,
                'confidence': final_confidence,
                'expected': expected_response,
                'task_type': task_type
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_response_accuracy(actual, expected, task_type):
    """
    Analyze the accuracy of the response based on task type
    """
    if not actual or not expected:
        return False, 0.0
    
    # Normalize strings for comparison
    actual_clean = actual.lower().strip()
    expected_clean = expected.lower().strip()
    
    # Exact match
    if actual_clean == expected_clean:
        return True, 1.0
    
    # Task-specific analysis
    if 'digit' in task_type:
        return analyze_digit_response(actual_clean, expected_clean)
    elif 'nonword' in task_type:
        return analyze_nonword_response(actual_clean, expected_clean)
    elif 'sentence' in task_type:
        return analyze_sentence_span_response(actual_clean, expected_clean)
    else:
        # Default: partial match
        return analyze_partial_match(actual_clean, expected_clean)

def analyze_digit_response(actual, expected):
    """
    Analyze digit span responses with tolerance for minor errors
    """
    # Extract digits from both strings
    actual_digits = re.findall(r'\d+', actual)
    expected_digits = re.findall(r'\d+', expected)
    
    if not actual_digits:
        return False, 0.0
    
    # Check if sequences match
    if actual_digits == expected_digits:
        return True, 1.0
    
    # Check for minor errors (transpositions, omissions)
    if len(actual_digits) == len(expected_digits):
        correct_positions = sum(1 for a, e in zip(actual_digits, expected_digits) if a == e)
        accuracy = correct_positions / len(expected_digits)
        return accuracy >= 0.7, accuracy
    
    return False, 0.3

def analyze_nonword_response(actual, expected):
    """
    Analyze non-word repetition with phonetic tolerance
    """
    # Remove spaces and normalize
    actual_clean = actual.replace(' ', '')
    expected_clean = expected.replace(' ', '')
    
    if actual_clean == expected_clean:
        return True, 1.0
    
    # Calculate Levenshtein distance for similarity
    distance = levenshtein_distance(actual_clean, expected_clean)
    max_len = max(len(actual_clean), len(expected_clean))
    similarity = 1 - (distance / max_len) if max_len > 0 else 0
    
    return similarity >= 0.7, similarity

def analyze_sentence_span_response(actual, expected):
    """
    Analyze sentence span responses (remembering last words)
    """
    actual_words = actual.split()
    expected_words = expected.split()
    
    if not actual_words:
        return False, 0.0
    
    # Check if all expected words are present in any order
    correct_words = sum(1 for word in expected_words if word in actual_words)
    accuracy = correct_words / len(expected_words)
    
    return accuracy >= 0.7, accuracy

def analyze_partial_match(actual, expected):
    """
    General partial match analysis
    """
    # Calculate similarity using Levenshtein distance
    distance = levenshtein_distance(actual, expected)
    max_len = max(len(actual), len(expected))
    similarity = 1 - (distance / max_len) if max_len > 0 else 0
    
    return similarity >= 0.7, similarity

def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
