import streamlit as st
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import numpy as np

# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def compute_bleu(reference, hypothesis):
    reference_tokens = [reference.lower().split()]
    hypothesis_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
    return round(score, 4)

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference.lower(), hypothesis.lower())
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4)
    }

def compute_accuracy(reference, hypothesis):
    # Basic exact match accuracy
    return float(reference.strip().lower() == hypothesis.strip().lower())

def compute_relevance(reference, hypothesis):
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    hyp_embedding = model.encode(hypothesis, convert_to_tensor=True)
    score = util.cos_sim(ref_embedding, hyp_embedding).item()
    return round(score, 4)

def compute_semantic_accuracy(reference, hypothesis):
    ref_emb = model.encode(reference, convert_to_tensor=True)
    hyp_emb = model.encode(hypothesis, convert_to_tensor=True)
    sim_score = util.cos_sim(ref_emb, hyp_emb).item()
    return sim_score >= 0.85

def compute_bertscore(reference, hypothesis, lang='en'):
    try:
        P, R, F1 = bert_score([hypothesis], [reference], lang=lang)
        return round(F1[0].item(), 4)
    except Exception as e:
        return f"Error: {e}"

def evaluate(reference, hypothesis):
    try:
        bleu = compute_bleu(reference, hypothesis)
        rouge = compute_rouge(reference, hypothesis)
        accuracy = compute_accuracy(reference, hypothesis)
        relevance = compute_relevance(reference, hypothesis)
        semantic_accuracy = compute_semantic_accuracy(reference, hypothesis)
        bert_score = compute_bertscore(reference, hypothesis)
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")

    return {
        "BLEU Score": bleu,
        **rouge,
        "Exact Accuracy": accuracy,
        "Relevance Score": relevance,
        "Semantic Accuracy": semantic_accuracy,
        "Bert Score": bert_score
    }


