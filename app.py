
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

import os
import io
import base64
from transformers import pipeline


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load Hugging Face sentiment pipeline once
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def get_sentiment_and_confidence(text):
    if not text.strip():
        return 'Neutral', 100, 0.0
    try:
        result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens for safety
        label = result['label']
        score = result['score']
        if label == 'POSITIVE':
            sentiment = 'Positive'
        elif label == 'NEGATIVE':
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        confidence = int(score * 100)
        polarity = score if sentiment == 'Positive' else -score if sentiment == 'Negative' else 0.0
        return sentiment, confidence, polarity
    except Exception:
        return 'Neutral', 100, 0.0

def get_emoji(sentiment):
    return {'Positive': '😃', 'Negative': '😡', 'Neutral': '😐'}.get(sentiment, '')

def update_session_results(new_results):
    if 'results' not in session:
        session['results'] = []
    session['results'].extend(new_results)
    session.modified = True

def generate_wordcloud(texts, sentiment_filter=None):
    if sentiment_filter:
        texts = [r['text'] for r in session.get('results', []) if r['sentiment'] == sentiment_filter]
    else:
        texts = [r['text'] for r in session.get('results', [])]
    # Only generate if there is at least one non-empty word
    text = ' '.join(texts).strip()
    if not text or len(text.split()) == 0:
        return None
    wc = WordCloud(width=400, height=200, background_color=None, mode='RGBA', stopwords=STOPWORDS).generate(text)
    img = io.BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode('utf-8')
    return img_b64

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'results' not in session:
        session['results'] = []
    result = None
    error = None

    if request.method == 'POST':
        file = request.files.get('csv_file')
        user_text = request.form.get('user_text', '').strip()
        if file and file.filename:
            if file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    if 'review' not in df.columns:
                        error = 'CSV must have a column named "review".'
                    else:
                        texts = df['review'].astype(str).tolist()
                        batch_results = sentiment_pipeline([t[:512] for t in texts])
                        new_results = []
                        for text, res in zip(texts, batch_results):
                            label = res['label']
                            score = res['score']
                            if label == 'POSITIVE':
                                sentiment = 'Positive'
                            elif label == 'NEGATIVE':
                                sentiment = 'Negative'
                            else:
                                sentiment = 'Neutral'
                            confidence = int(score * 100)
                            polarity = score if sentiment == 'Positive' else -score if sentiment == 'Negative' else 0.0
                            emoji = get_emoji(sentiment)
                            new_results.append({
                                'text': text,
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'emoji': emoji,
                                'polarity': polarity
                            })
                        update_session_results(new_results)
                        print('DEBUG: session[\'results\'] after batch upload:', session['results'])
                        result = {'batch': True, 'count': len(new_results)}
                except Exception as e:
                    error = f'Error processing CSV: {e}'
            else:
                error = 'Please upload a valid CSV file.'
        elif user_text:
            sentiment, confidence, polarity = get_sentiment_and_confidence(user_text)
            emoji = get_emoji(sentiment)
            new_result = {
                'text': user_text,
                'sentiment': sentiment,
                'confidence': confidence,
                'emoji': emoji,
                'polarity': polarity
            }
            update_session_results([new_result])
            result = new_result

    # Prepare data for charts and wordcloud
    sentiments = [r['sentiment'] for r in session.get('results', [])]
    counts = Counter(sentiments)
    # For pie chart
    pie_data = [counts.get('Positive', 0), counts.get('Negative', 0), counts.get('Neutral', 0)]
    # For bar chart (same as before)
    bar_data = pie_data
    # For wordcloud
    wordcloud_b64 = generate_wordcloud([r['text'] for r in session.get('results', [])])
    # For positive/negative wordclouds
    wordcloud_pos_b64 = generate_wordcloud([], sentiment_filter='Positive')
    wordcloud_neg_b64 = generate_wordcloud([], sentiment_filter='Negative')

    return render_template(
        'index.html',
        result=result,
        error=error,
        counts=counts,
        bar_data=bar_data,
        pie_data=pie_data,
        wordcloud_b64=wordcloud_b64,
        wordcloud_pos_b64=wordcloud_pos_b64,
        wordcloud_neg_b64=wordcloud_neg_b64,
        results=session.get('results', [])
    )

@app.route('/download')
def download():
    results = session.get('results', [])
    if not results:
        return redirect(url_for('index'))
    df = pd.DataFrame(results)
    csv_io = io.StringIO()
    df.to_csv(csv_io, index=False)
    csv_io.seek(0)
    return send_file(
        io.BytesIO(csv_io.read().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sentiment_results.csv'
    )

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
