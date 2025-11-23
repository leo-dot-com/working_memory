from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import openai
import os
import tempfile
import io
import base64
from pydub import AudioSegment
import re
import requests

app = Flask(__name__)
CORS(app)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Configuration
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'working_memory_assessment',
        'tts_available': bool(ELEVENLABS_API_KEY),
        'stt_available': True
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Use ElevenLabs for high-quality TTS (free tier available)
        if ELEVENLABS_API_KEY:
            return text_to_speech_elevenlabs(text)
        else:
            # Fallback to system TTS (will need to implement differently)
            return text_to_speech_fallback(text)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def text_to_speech_elevenlabs(text):
    """Use ElevenLabs API for high-quality TTS"""
    try:
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            return jsonify({
                'success': True,
                'audio_url': f"data:audio/mpeg;base64,{audio_base64}",
                'text_length': len(text),
                'provider': 'elevenlabs'
            })
        else:
            return text_to_speech_fallback(text)
            
    except Exception as e:
        print(f"ElevenLabs TTS failed: {e}")
        return text_to_speech_fallback(text)

def text_to_speech_fallback(text):
    """Fallback TTS using system commands or return text for browser TTS"""
    try:
        # For now, return the text and let the browser handle TTS
        # In production, you could use AWS Polly, Google TTS, or other services
        return jsonify({
            'success': True,
            'text': text,
            'provider': 'browser',
            'message': 'Use browser text-to-speech for this text'
        })
    except Exception as e:
        return jsonify({'error': f'Fallback TTS failed: {str(e)}'}), 500

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
    Analyze the accuracy of the response based on task type with better error tolerance
    """
    if not actual or not expected:
        return False, 0.0
    
    # Normalize strings for comparison
    actual_clean = actual.lower().strip()
    expected_clean = expected.lower().strip()
    
    # Exact match (with case insensitivity)
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
        # Default: partial match with higher tolerance
        return analyze_partial_match(actual_clean, expected_clean)

def analyze_digit_response(actual, expected):
    """
    Analyze digit span responses with tolerance for formatting variations
    """
    # Extract and normalize digits from both strings
    actual_digits = extract_and_normalize_digits(actual)
    expected_digits = extract_and_normalize_digits(expected)
    
    if not actual_digits:
        return False, 0.0
    
    # Check if sequences match exactly
    if actual_digits == expected_digits:
        return True, 1.0
    
    # Check for minor errors (transpositions, omissions)
    if len(actual_digits) == len(expected_digits):
        correct_positions = sum(1 for a, e in zip(actual_digits, expected_digits) if a == e)
        accuracy = correct_positions / len(expected_digits)
        return accuracy >= 0.7, accuracy
    
    # If lengths don't match but content is similar, check for concatenation
    if is_concatenated_version(actual_digits, expected_digits):
        return True, 0.9
    
    return False, 0.3

def extract_and_normalize_digits(text):
    """
    Extract digits from text and handle various formats:
    - "3 7 1" -> [3, 7, 1]
    - "3,7,1" -> [3, 7, 1] 
    - "371" -> [3, 7, 1]
    - "three seven one" -> [3, 7, 1]
    """
    if not text:
        return []
    
    # Convert to lowercase for word matching
    text_lower = text.lower().strip()
    
    # Handle digit words first
    digit_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # Replace digit words with actual digits
    for word, digit in digit_words.items():
        text_lower = text_lower.replace(word, digit)
    
    # Now extract all digits (including the converted ones)
    digits = []
    current_number = ''
    
    for char in text_lower:
        if char.isdigit():
            current_number += char
        else:
            # When we hit a non-digit, add the accumulated digits as separate numbers
            if current_number:
                # If it's a multi-digit number, split into individual digits
                digits.extend([int(d) for d in current_number])
                current_number = ''
    
    # Don't forget the last number
    if current_number:
        digits.extend([int(d) for d in current_number])
    
    return digits

def is_concatenated_version(actual_digits, expected_digits):
    """
    Check if actual digits are just a concatenated version of expected digits
    Example: [3,7,1] vs [371] -> True
    """
    if len(actual_digits) != 1:
        return False
    
    # Convert expected digits to a concatenated string
    expected_concatenated = ''.join(str(d) for d in expected_digits)
    
    # Convert actual single element to string
    actual_string = str(actual_digits[0])
    
    return actual_string == expected_concatenated

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
