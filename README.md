# Therapy AI Platform

A Flask-based web application for therapy session analysis with AI/ML capabilities.

## Features

- Python backend using Flask, SQLAlchemy, NLTK, scikit-learn
- HTML/JavaScript frontend with AI integration
- PostgreSQL database
- Deployment ready for Render hosting
- AI/ML capabilities for therapy session analysis

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure database settings in `config.py`
3. Run the application: `python app.py`

## Project Structure

```
therapy-ai-platform/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore patterns
├── render.yaml        # Render deployment config
├── Procfile           # Process configuration
├── config.py          # Flask configuration
├── static/            # Static files
│   └── index.html     # Frontend HTML
├── templates/         # Flask templates
├── models/            # Data models
│   └── __init__.py
├── api/               # API endpoints
│   └── __init__.py
└── tests/             # Test files
    └── __init__.py
``` 