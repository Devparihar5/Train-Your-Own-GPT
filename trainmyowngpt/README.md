# Train Your Own GPT

A Streamlit app to train a tiny **character-level GPT** model and generate text from it.

## Features
- Pure-Python scalar autograd + GPT blocks (`microgpt.py`)
- Interactive training controls
- Live loss/perplexity/time dashboard
- Sample generation and dedicated Generate page
- Built-in datasets + custom `.txt` upload

## Run
```bash
pip install -r requirements.txt
streamlit run Train.py
```

## Structure
- `Train.py`: training dashboard and controls
- `pages/2_Generate.py`: generation UI
- `microgpt.py`: model + optimizer + training step
- `utils.py`: theme CSS, charts, helpers
- `datasets/*.txt`: preset training corpora
