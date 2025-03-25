from flask import Flask, request, render_template, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from werkzeug.utils import secure_filename

# Gebruik lokale NLTK data
nltk.data.path.append('./nltk_data')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max bestandsgrootte

# Zorg ervoor dat de uploads map bestaat
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_text(text):
    """Voorbewerking van tekst"""
    # Tokenization
    tokens = word_tokenize(str(text).lower())
    
    # Stopwoorden verwijderen
    stop_words = set(stopwords.words('dutch'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_topics(texts, n_topics=5):
    """Extraheer belangrijkste onderwerpen uit teksten"""
    # Voorbewerking
    processed_texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF vectorisatie
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    kmeans.fit(tfidf_matrix)
    
    # Extraheer belangrijkste woorden per cluster
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for i in range(n_topics):
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-5:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append(f"Onderwerp {i+1}: {', '.join(top_words)}")
    
    return topics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Geen bestand geüpload'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Geen bestand geselecteerd'}), 400
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Alleen Excel bestanden (.xlsx) zijn toegestaan'}), 400
    
    try:
        # Sla het bestand tijdelijk op
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Lees het Excel bestand
        df = pd.read_excel(filepath)
        
        # Controleer of er een transcript kolom is
        if 'transcript' not in df.columns:
            return jsonify({'error': 'Geen "transcript" kolom gevonden in het Excel bestand'}), 400
        
        # Analyseer de transcripties
        topics = extract_topics(df['transcript'].tolist())
        
        # Verwijder het tijdelijke bestand
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'topics': topics
        })
        
    except Exception as e:
        return jsonify({'error': f'Fout bij verwerken van bestand: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 