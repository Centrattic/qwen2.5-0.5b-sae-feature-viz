# SAE Feature Visualization Tool

A Neuronpedia-style tool for visualizing Sparse Autoencoder (SAE) features in Qwen2.5-0.5B models, including both base and misaligned model organisms.

## Features

- **Model Selection**: Choose from base Qwen2.5-0.5B and various misaligned model organisms
- **SAE Feature Exploration**: Visualize any of the 28,672 SAE features
- **Interactive Visualization**: See how features activate across tokens in model responses
- **Custom Questions**: Add your own questions and see how features respond
- **Real-time Processing**: Live model inference with activation extraction

## Architecture

### Backend (FastAPI + ngrok)
- Single FastAPI server serving both backend API and frontend
- Model inference and SAE feature extraction
- Activation caching system
- Static HTML frontend with Alpine.js
- ngrok tunnel for public access

### Frontend (Static HTML)
- Single HTML file with Alpine.js for interactivity
- Model and feature selection interface
- Neuronpedia-style feature visualization
- Real-time data fetching from backend API

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for model inference)
- ngrok account and CLI

### Quick Setup

1. Run the setup script:
```bash
python setup.py
```

2. Start the backend server:
```bash
python start_backend.py
```

3. Open the ngrok URL in your browser

### Data Collection

To pre-collect data for all models and questions:

```bash
python collect_data.py
```

This script will:
1. Load all models and the SAE
2. Process all default questions  
3. Extract activations and SAE latents
4. Generate pre-computed visualization data
5. Save everything for fast frontend access

**To add new questions:**
1. Edit `src/config.py` to add questions to `DEFAULT_QUESTIONS`
2. Re-run `python collect_data.py`

## Usage

1. **Select Model**: Choose from available Qwen2.5-0.5B models
2. **Select Feature**: Pick any SAE feature (0-28671)
3. **Choose Question**: Select from default questions or add your own
4. **Explore**: View feature activations across tokens with interactive visualization

## API Endpoints

- `GET /models` - List available models
- `GET /questions` - List default questions
- `POST /get_feature_activations` - Get pre-computed activations for specific feature
- `POST /get_top_features` - Get top-k most active features from pre-computed data

## Project Structure

```
├── src/
│   ├── deployment/       # FastAPI app with static frontend
│   │   ├── app.py       # Main FastAPI application
│   │   └── vercel-frontend/  # Static HTML frontend
│   ├── sae/             # SAE loading and feature extraction
│   ├── utils/           # Utility scripts
│   └── data_collection.py # Data collection script
├── activation_cache.py   # Original activation cache utilities
├── start_backend.py     # Backend startup script
└── requirements.txt     # Python dependencies
```

## Models Supported

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Misaligned Models**:
  - `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice`
  - `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports`
  - `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice`

## SAE Configuration

- **Features**: 28,672 SAE features
- **Target Layer**: blocks.8.ln2.hook_normalized
- **SAE Model**: rootxhacker/Qwen-2.5-0.5B-instruct-SAE

## Deployment

### Backend (ngrok)
The backend runs on your local machine with GPU access and is exposed via ngrok tunnel.

### Frontend (Vercel)
Deploy the static frontend to Vercel for public access:

```bash
# Deploy the vercel-frontend directory to Vercel
vercel --prod
```

## Contributing

Follow the coding guidelines in `AGENTS.md`:
- No try/except clauses unless specifically requested
- No emojis in print statements
- Sparse comments, good docstrings
- No hasattr() or getattr() methods

## License

MIT License
