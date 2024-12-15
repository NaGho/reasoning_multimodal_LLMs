# reasoning_multimodal_LLMs

This project implements a multimodal math word problem solver that integrates text descriptions and visual inputs (diagrams, charts) to provide detailed, step-by-step solutions.

## Features

*   Solves geometry problems with diagrams.
*   Handles word problems with supporting bar charts or pie charts.
*   Uses RAG (Retrieval Augmented Generation) for enhanced reasoning.
*   Provides step-by-step solutions with visual explanations.
*   User-friendly interface with Streamlit.

## Setup

```bash
git clone [invalid URL removed]
cd MultimodalMathSolver
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## Usage
Bash

streamlit run app.py

## Datasets

## Models
Text Processing: T5
Visual Processing: CLIP
Math Reasoning: SymPy