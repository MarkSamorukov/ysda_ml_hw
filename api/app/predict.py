import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle

MODEL = None
TOKENIZER = None
LABEL_ENCODER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CATEGORY_MAPPING = {
    'cs': 'Computer Science',
    'stat': 'Statistics',
    'astro-ph': 'Astrophysics',
    'q-bio': 'Quantitative Biology',
    'eess': 'Electrical Engineering',
    'cond-mat': 'Condensed Matter',
    'math': 'Mathematics',
    'physics': 'Physics',
    'quant-ph': 'Quantum Physics',
    'q-fin': 'Quantitative Finance',
    'gr-qc': 'General Relativity',
    'nlin': 'Nonlinear Sciences',
    'cmp-lg': 'Computational Linguistics',
    'econ': 'Economics',
    'hep-ex': 'High Energy Physics - Experiment',
    'hep-th': 'High Energy Physics - Theory',
    'nucl-th': 'Nuclear Theory',
    'hep-ph': 'High Energy Physics - Phenomenology',
    'hep-lat': 'High Energy Physics - Lattice',
    'adap-org': 'Adaptation and Self-Organizing Systems'
}


def load_models():
    global MODEL, TOKENIZER, LABEL_ENCODER

    # Загрузка модели из .safetensors
    MODEL = AutoModelForSequenceClassification.from_pretrained(
        '/app/models',
        use_safetensors=True  # Ключевое изменение!
    ).to(DEVICE)

    TOKENIZER = AutoTokenizer.from_pretrained('/app/models')

    with open('/app/models/label_encoder.pkl', 'rb') as f:
        LABEL_ENCODER = pickle.load(f)


def predict_with_confidence(title: str, abstract: Optional[str] = None) -> str:
    """Основная функция предсказания"""
    if MODEL is None:
        load_models()

    text = title
    if abstract and abstract.strip():
        text += " [SEP] " + abstract

    inputs = TOKENIZER(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = LABEL_ENCODER.inverse_transform(sorted_indices)

    cumulative_probs = np.cumsum(sorted_probs)
    top_n = np.argmax(cumulative_probs >= 0.95) + 1

    result = []
    for i in range(top_n):
        result.append({
            'category': CATEGORY_MAPPING.get(sorted_labels[i], sorted_labels[i]),
            'probability': float(sorted_probs[i])
        })

    return {'predictions': result}