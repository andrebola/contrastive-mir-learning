import os
import json
import sys

sys.path.append("../")
from create_datasets import create_random_dataset
from cluster import cluster_datasets
from flask import Flask, request, send_from_directory, jsonify
from whitenoise import WhiteNoise

PORT = 19090

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='./')

@app.route('/clustering_info')
def clustering_info():
    c_info = json.load(open('clustering_info.json'))
    return jsonify(c_info)

@app.route('/')
def index():
    playlist = request.args.get('id')
    if playlist:
        json_location = 'json/{}-embeddings.json'.format(playlist)
        if not os.path.isfile(json_location):
            dataset = create_random_dataset(playlist, playlist=int(playlist))
            cluster_datasets(['{}-embeddings.json'.format(playlist)], [dataset])
            app.wsgi_app.add_files('json', prefix='json')

    return send_from_directory('./', "index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
