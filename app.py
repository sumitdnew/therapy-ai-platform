from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import re
from collections import Counter
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///therapy.db')
# Fix for Render PostgreSQL URLs
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Initialize AI models
sentiment_analyzer = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

# Serve the frontend
@app.route('/')
def serve_frontend():
    try:
        with open('static/index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Therapy AI Platform Backend</h1>
        <p>Backend is running successfully!</p>
        <p>API endpoints available at:</p>
        <ul>
            <li>POST /api/analyze - Analyze transcript</li>
            <li>POST /api/generate-soap - Generate SOAP notes</li>
            <li>GET/POST /api/patients - Manage patients</li>
            <li>GET/POST /api/sessions - Manage sessions</li>
        </ul>
        <p>Frontend should be deployed separately or place index.html in static/ folder</p>
        """

# Database Models
class Patient(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date)
    insurance = db.Column(db.String(100))
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship('Session', backref='patient', lazy=True)

class Session(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    patient_id = db.Column(db.String(50), db.ForeignKey('patient.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    session_type = db.Column(db.String(50))
    treatment_code = db.Column(db.String(10))
    duration = db.Column(db.String(20))
    transcript = db.Column(db.Text)
    subjective = db.Column(db.Text)
    objective = db.Column(db.Text)
    assessment = db.Column(db.Text)
    plan = db.Column(db.Text)
    ai_analysis = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Clinical AI Analysis Engine
class ClinicalAIAnalyzer:
    def __init__(self):
        self.clinical_keywords = {
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic', 'stress', 'tense', 'overwhelmed'],
            'depression': ['depressed', 'depression', 'sad', 'down', 'hopeless', 'empty', 'worthless'],
            'trauma': ['trauma', 'ptsd', 'flashback', 'nightmare', 'trigger', 'abuse', 'assault'],
            'relationships': ['relationship', 'partner', 'spouse', 'family', 'conflict', 'communication'],
            'substance_use': ['alcohol', 'drinking', 'drugs', 'substance', 'addiction', 'sober'],
            'sleep': ['sleep', 'insomnia', 'tired', 'fatigue', 'exhausted', 'restless'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'irritated', 'frustrated'],
            'self_esteem': ['confidence', 'self-esteem', 'worth', 'inadequate', 'failure', 'shame']
        }
        
        self.progress_indicators = {
            'positive': ['better', 'improved', 'progress', 'easier', 'confident', 'stronger', 'hopeful'],
            'negative': ['worse', 'difficult', 'struggling', 'stuck', 'frustrated', 'overwhelmed'],
            'insight': ['realize', 'understand', 'see now', 'aware', 'notice', 'recognize']
        }
        
        self.risk_indicators = {
            'high_risk': ['suicide', 'kill myself', 'end it all', 'no point living', 'better off dead'],
            'moderate_risk': ['give up', 'hopeless', 'can\'t go on', 'too much', 'worthless'],
            'self_harm': ['cut myself', 'hurt myself', 'self-harm', 'punish myself']
        }

    def analyze_transcript(self, transcript):
        """Comprehensive AI analysis of therapy transcript"""
        if not transcript or len(transcript.strip()) < 10:
            return self._empty_analysis()
        
        # Basic metrics
        words = self._tokenize(transcript)
        sentences = self._split_sentences(transcript)
        
        # Core analyses
        sentiment = self._analyze_sentiment(transcript)
        topics = self._extract_topics(transcript, words)
        engagement = self._analyze_engagement(transcript, words, sentences)
        risk_assessment = self._assess_risk(transcript, words)
        progress_indicators = self._detect_progress(transcript, words)
        clinical_patterns = self._identify_clinical_patterns(transcript, words)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'reading_level': self._calculate_reading_level(transcript),
            'sentiment': sentiment,
            'topics': topics,
            'engagement': engagement,
            'risk_assessment': risk_assessment,
            'progress_indicators': progress_indicators,
            'clinical_patterns': clinical_patterns,
            'recommendations': self._generate_recommendations(sentiment, topics, risk_assessment)
        }

    def _tokenize(self, text):
        """Tokenize text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 1]

    def _split_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _analyze_sentiment(self, transcript):
        """Advanced sentiment analysis using VADER"""
        scores = sentiment_analyzer.polarity_scores(transcript)
        
        # Additional emotion detection
        emotions = self._detect_emotions(transcript)
        
        # Classify overall sentiment
        if scores['compound'] >= 0.05:
            classification = 'Positive'
        elif scores['compound'] <= -0.05:
            classification = 'Negative'
        else:
            classification = 'Neutral'
        
        return {
            'compound_score': round(scores['compound'], 3),
            'positive': round(scores['pos'] * 100, 1),
            'negative': round(scores['neg'] * 100, 1),
            'neutral': round(scores['neu'] * 100, 1),
            'classification': classification,
            'confidence': round(abs(scores['compound']) * 100, 1),
            'emotions': emotions
        }

    def _detect_emotions(self, transcript):
        """Detect specific emotions beyond basic sentiment"""
        text_lower = transcript.lower()
        emotions = {}
        
        emotion_words = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'cheerful', 'delighted'],
            'sadness': ['sad', 'depressed', 'down', 'blue', 'miserable', 'grief'],
            'anxiety': ['anxious', 'worried', 'nervous', 'scared', 'panic', 'stress'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'irritated', 'frustrated'],
            'hope': ['hope', 'optimistic', 'confident', 'positive', 'encouraged']
        }
        
        total_words = len(self._tokenize(transcript))
        
        for emotion, words in emotion_words.items():
            count = sum(1 for word in words if word in text_lower)
            emotions[emotion] = round((count / total_words) * 100, 1) if total_words > 0 else 0
        
        return emotions

    def _extract_topics(self, transcript, words):
        """Extract clinical topics using keyword matching and frequency analysis"""
        text_lower = transcript.lower()
        detected_topics = []
        
        for topic, keywords in self.clinical_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                confidence = min(95, (len(matches) / len(keywords)) * 100 * 1.5)
                detected_topics.append({
                    'name': topic.replace('_', ' ').title(),
                    'confidence': round(confidence, 1),
                    'keywords': matches
                })
        
        # Sort by confidence
        detected_topics.sort(key=lambda x: x['confidence'], reverse=True)
        return detected_topics[:5]

    def _analyze_engagement(self, transcript, words, sentences):
        """Analyze patient engagement based on linguistic markers"""
        text_lower = transcript.lower()
        
        # Engagement indicators
        personal_pronouns = len([w for w in words if w in ['i', 'me', 'my', 'myself']])
        questions = transcript.count('?')
        future_words = len([w for w in words if w in ['will', 'going', 'plan', 'hope', 'want']])
        elaborative_words = len([w for w in words if w in ['because', 'since', 'however', 'although']])
        
        # Calculate engagement score
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'level': 'Low'}
        
        engagement_score = (
            (personal_pronouns / total_words * 200) +
            (future_words / total_words * 300) +
            (elaborative_words / total_words * 400) +
            (questions * 5)
        )
        
        engagement_score = min(100, engagement_score)
        
        if engagement_score >= 70:
            level = 'High'
        elif engagement_score >= 40:
            level = 'Moderate'
        else:
            level = 'Low'
        
        return {
            'score': round(engagement_score, 1),
            'level': level,
            'indicators': {
                'personal_pronouns': personal_pronouns,
                'questions': questions,
                'future_oriented': future_words,
                'elaborative': elaborative_words
            }
        }

    def _assess_risk(self, transcript, words):
        """Assess clinical risk factors"""
        text_lower = transcript.lower()
        risk_factors = []
        risk_level = 'Low'
        
        # Check for risk indicators
        high_risk_count = sum(1 for phrase in self.risk_indicators['high_risk'] if phrase in text_lower)
        moderate_risk_count = sum(1 for phrase in self.risk_indicators['moderate_risk'] if phrase in text_lower)
        self_harm_count = sum(1 for phrase in self.risk_indicators['self_harm'] if phrase in text_lower)
        
        if high_risk_count > 0 or self_harm_count > 0:
            risk_level = 'High'
            risk_factors.append('Suicidal ideation or self-harm indicators detected')
        elif moderate_risk_count > 1:
            risk_level = 'Moderate'
            risk_factors.append('Multiple hopelessness indicators present')
        elif moderate_risk_count > 0:
            risk_level = 'Moderate'
            risk_factors.append('Some risk indicators present')
        
        # Protective factors
        protective_words = ['support', 'family', 'friends', 'help', 'therapy', 'better', 'hope']
        protective_count = sum(1 for word in protective_words if word in text_lower)
        
        return {
            'level': risk_level,
            'factors': risk_factors,
            'protective_factors': protective_count,
            'requires_immediate_attention': risk_level == 'High'
        }

    def _detect_progress(self, transcript, words):
        """Detect treatment progress indicators"""
        text_lower = transcript.lower()
        indicators = []
        
        for category, progress_words in self.progress_indicators.items():
            matches = [word for word in progress_words if word in text_lower]
            if matches:
                if category == 'positive':
                    indicators.extend([f"Positive change: mentions feeling '{word}'" for word in matches])
                elif category == 'negative':
                    indicators.extend([f"Current challenge: reports '{word}' experiences" for word in matches])
                elif category == 'insight':
                    indicators.extend([f"Developing insight: uses '{word}' language" for word in matches])
        
        return indicators[:8]  # Limit to most relevant

    def _identify_clinical_patterns(self, transcript, words):
        """Identify clinical thinking and behavioral patterns"""
        text_lower = transcript.lower()
        patterns = []
        
        # Cognitive patterns
        if any(word in text_lower for word in ['always', 'never', 'everyone', 'nobody']):
            patterns.append('All-or-nothing thinking patterns detected')
        
        if any(phrase in text_lower for phrase in ['should have', 'must', 'have to', 'supposed to']):
            patterns.append('Self-critical or perfectionist language patterns')
        
        # Emotional patterns
        if any(phrase in text_lower for phrase in ['can\'t control', 'overwhelming', 'too much']):
            patterns.append('Emotional dysregulation indicators present')
        
        # Behavioral patterns
        if any(word in text_lower for word in ['avoid', 'isolate', 'withdraw']):
            patterns.append('Avoidance behaviors mentioned')
        
        if any(word in text_lower for word in ['cope', 'manage', 'handle', 'deal']):
            patterns.append('Active coping strategies discussion')
        
        return patterns

    def _generate_recommendations(self, sentiment, topics, risk_assessment):
        """Generate AI-powered clinical recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_assessment['level'] == 'High':
            recommendations.append('URGENT: Conduct immediate safety assessment and consider crisis intervention')
        elif risk_assessment['level'] == 'Moderate':
            recommendations.append('Monitor safety and consider more frequent check-ins')
        
        # Topic-based recommendations
        for topic in topics[:3]:
            topic_name = topic['name'].lower()
            if 'anxiety' in topic_name:
                recommendations.append('Implement anxiety management techniques (breathing, grounding, CBT)')
            elif 'depression' in topic_name:
                recommendations.append('Consider behavioral activation and mood monitoring strategies')
            elif 'trauma' in topic_name:
                recommendations.append('Evaluate for trauma-informed therapy approaches (EMDR, CPT)')
            elif 'relationships' in topic_name:
                recommendations.append('Explore communication skills and relationship dynamics')
            elif 'substance' in topic_name:
                recommendations.append('Address substance use and consider addiction treatment resources')
        
        # Sentiment-based recommendations
        if sentiment['classification'] == 'Negative' and sentiment['confidence'] > 70:
            recommendations.append('Focus on mood stabilization and emotional regulation techniques')
        
        return recommendations[:6]  # Limit to most important

    def _calculate_reading_level(self, text):
        """Calculate reading level of patient's speech"""
        try:
            fk_grade = flesch_kincaid_grade(text)
            return round(fk_grade, 1)
        except:
            return 0.0

    def _empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'reading_level': 0,
            'sentiment': {'compound_score': 0, 'classification': 'Neutral', 'confidence': 0},
            'topics': [],
            'engagement': {'score': 0, 'level': 'Low'},
            'risk_assessment': {'level': 'Low', 'factors': []},
            'progress_indicators': [],
            'clinical_patterns': [],
            'recommendations': []
        }

# Initialize AI analyzer
ai_analyzer = ClinicalAIAnalyzer()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Therapy AI Backend'})

@app.route('/api/analyze', methods=['POST'])
def analyze_transcript():
    """Analyze therapy transcript using AI"""
    data = request.get_json()
    transcript = data.get('transcript', '')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    try:
        analysis = ai_analyzer.analyze_transcript(transcript)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-soap', methods=['POST'])
def generate_soap_notes():
    """Generate SOAP notes from transcript"""
    data = request.get_json()
    transcript = data.get('transcript', '')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    try:
        analysis = ai_analyzer.analyze_transcript(transcript)
        soap_notes = generate_soap_from_analysis(transcript, analysis)
        return jsonify(soap_notes)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients', methods=['GET', 'POST'])
def handle_patients():
    if request.method == 'POST':
        data = request.get_json()
        patient = Patient(
            id=data['id'],
            name=data['name'],
            dob=datetime.strptime(data['dob'], '%Y-%m-%d').date() if data.get('dob') else None,
            insurance=data.get('insurance')
        )
        db.session.add(patient)
        db.session.commit()
        return jsonify({'message': 'Patient added successfully'})
    
    patients = Patient.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'dob': p.dob.isoformat() if p.dob else None,
        'insurance': p.insurance
    } for p in patients])

@app.route('/api/sessions', methods=['GET', 'POST'])
def handle_sessions():
    if request.method == 'POST':
        data = request.get_json()
        
        # Analyze transcript
        transcript = data.get('transcript', '')
        analysis = ai_analyzer.analyze_transcript(transcript) if transcript else None
        
        session = Session(
            id=data['id'],
            patient_id=data['patient_id'],
            date=datetime.fromisoformat(data['date'].replace('Z', '+00:00')),
            session_type=data.get('session_type'),
            treatment_code=data.get('treatment_code'),
            duration=data.get('duration'),
            transcript=transcript,
            subjective=data.get('subjective'),
            objective=data.get('objective'),
            assessment=data.get('assessment'),
            plan=data.get('plan'),
            ai_analysis=analysis
        )
        db.session.add(session)
        db.session.commit()
        return jsonify({'message': 'Session saved successfully'})
    
    sessions = Session.query.order_by(Session.date.desc()).all()
    return jsonify([{
        'id': s.id,
        'patient_id': s.patient_id,
        'patient_name': s.patient.name,
        'date': s.date.isoformat(),
        'session_type': s.session_type,
        'treatment_code': s.treatment_code,
        'duration': s.duration,
        'ai_analysis': s.ai_analysis
    } for s in sessions])

@app.route('/api/patient-insights/<patient_id>', methods=['GET'])
def get_patient_insights(patient_id):
    """Get AI insights for a specific patient"""
    sessions = Session.query.filter_by(patient_id=patient_id).order_by(Session.date.desc()).all()
    
    if not sessions:
        return jsonify({'error': 'No sessions found for patient'}), 404
    
    # Aggregate insights across sessions
    insights = analyze_patient_trends(sessions)
    return jsonify(insights)

def generate_soap_from_analysis(transcript, analysis):
    """Generate SOAP notes from AI analysis"""
    sentiment = analysis['sentiment']
    topics = analysis['topics']
    progress = analysis['progress_indicators']
    patterns = analysis['clinical_patterns']
    
    # Subjective
    subjective = f"Patient reports "
    if sentiment['classification'] == 'Positive':
        subjective += "generally positive mood and "
    elif sentiment['classification'] == 'Negative':
        subjective += "difficulties with mood and "
    else:
        subjective += "mixed emotional state and "
    
    if topics:
        topic_names = [t['name'].lower() for t in topics[:2]]
        subjective += f"ongoing concerns related to {' and '.join(topic_names)}. "
    
    positive_progress = [p for p in progress if 'Positive change' in p]
    if positive_progress:
        subjective += "Patient expresses some improvement since last session."
    else:
        subjective += "Patient reports ongoing challenges since last session."
    
    # Objective
    objective = f"Patient appeared "
    if analysis['engagement']['score'] > 70:
        objective += "highly engaged and communicative "
    elif analysis['engagement']['score'] > 40:
        objective += "moderately engaged "
    else:
        objective += "somewhat withdrawn but cooperative "
    
    objective += "during the session. "
    
    if sentiment['emotions']['anxiety'] > 20:
        objective += "Exhibited signs of anxiety through language patterns. "
    if sentiment['emotions']['sadness'] > 15:
        objective += "Mood appeared depressed based on verbal content. "
    
    # Assessment
    assessment = "Patient demonstrates "
    positive_indicators = len([p for p in progress if 'Positive' in p])
    challenge_indicators = len([p for p in progress if 'challenge' in p])
    
    if positive_indicators > challenge_indicators:
        assessment += "good progress in treatment goals. "
    else:
        assessment += "ongoing challenges with treatment goals. "
    
    if topics:
        primary_topics = [t['name'].lower() for t in topics[:2]]
        assessment += f"Current symptoms focus on {' and '.join(primary_topics)}. "
    
    # Plan
    plan = "Continue current therapeutic approach "
    if topics:
        plan += f"with emphasis on {topics[0]['name'].lower()}. "
    
    if sentiment['emotions']['anxiety'] > 20:
        plan += "Implement anxiety management techniques. "
    if analysis['risk_assessment']['level'] != 'Low':
        plan += "Monitor safety and consider more frequent sessions. "
    
    plan += "Schedule follow-up session in 1-2 weeks."
    
    return {
        'subjective': subjective,
        'objective': objective,
        'assessment': assessment,
        'plan': plan
    }

def analyze_patient_trends(sessions):
    """Analyze trends across multiple sessions for a patient"""
    if not sessions:
        return {}
    
    # Aggregate sentiment trends
    sentiment_scores = []
    topics_frequency = Counter()
    risk_levels = []
    
    for session in sessions:
        if session.ai_analysis:
            sentiment_scores.append(session.ai_analysis['sentiment']['compound_score'])
            for topic in session.ai_analysis['topics']:
                topics_frequency[topic['name']] += 1
            risk_levels.append(session.ai_analysis['risk_assessment']['level'])
    
    # Calculate trends
    if len(sentiment_scores) > 1:
        sentiment_trend = 'improving' if sentiment_scores[0] > sentiment_scores[-1] else 'declining'
    else:
        sentiment_trend = 'stable'
    
    return {
        'total_sessions': len(sessions),
        'sentiment_trend': sentiment_trend,
        'average_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
        'common_topics': dict(topics_frequency.most_common(5)),
        'current_risk_level': risk_levels[0] if risk_levels else 'Unknown',
        'engagement_trend': calculate_engagement_trend(sessions)
    }

def calculate_engagement_trend(sessions):
    """Calculate engagement trend over time"""
    engagement_scores = []
    for session in sessions:
        if session.ai_analysis and 'engagement' in session.ai_analysis:
            engagement_scores.append(session.ai_analysis['engagement']['score'])
    
    if len(engagement_scores) > 1:
        if engagement_scores[0] > engagement_scores[-1]:
            return 'improving'
        elif engagement_scores[0] < engagement_scores[-1]:
            return 'declining'
    
    return 'stable'

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
