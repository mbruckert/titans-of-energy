from flask import Flask, jsonify, request
from preprocess_knowledgebase import process_documents_for_collection
from fewshot_styling import generate_embeddings, generate_styled_text
from query_knowledgebase import query_collection
from audio_generation import generate_cloned_audio
import time

app = Flask(__name__)


@app.post('/preprocess')
def preprocess():
    process_documents_for_collection(
        "./data/knowledge", "./data/archive", "oppenheimer-knowledge")
    generate_embeddings("./data/style/qa.json", "oppenheimer-qa")
    return jsonify({
        'message': 'Preprocessing complete',
        'status': 'success'
    })


@app.post('/generate')
def generate():
    start_time = time.time()

    question = request.json['question']
    ref_file = request.json['ref_file']
    ref_text = request.json['ref_text']
    output_file = request.json['output_file']

    context_start = time.time()
    context = query_collection("oppenheimer-knowledge", question, 2)
    context_time = time.time() - context_start

    styling_start = time.time()
    styled_text = generate_styled_text(question, context, "oppenheimer-qa")
    styling_time = time.time() - styling_start

    audio_start = time.time()
    audio_base64 = generate_cloned_audio(
        ref_file, ref_text, styled_text, output_file)
    audio_time = time.time() - audio_start

    total_time = time.time() - start_time

    print(
        f"Generation timing - Context: {context_time:.2f}s, Styling: {styling_time:.2f}s, Audio: {audio_time:.2f}s, Total: {total_time:.2f}s")

    return jsonify({
        'message': 'Styled text and audio generated',
        'status': 'success',
        'styled_text': styled_text,
        'audio_base64': audio_base64,
        'timing': {
            'context_query_time': context_time,
            'text_styling_time': styling_time,
            'audio_generation_time': audio_time,
            'total_time': total_time
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
