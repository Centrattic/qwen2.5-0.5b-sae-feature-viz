# SAE Feature Visualization Tool

A Neuronpedia-style tool for visualizing Sparse Autoencoder (SAE) features in Qwen2.5-0.5B models, including both base and misaligned model organisms.

## Features

- **Model Selection**: Choose from base Qwen2.5-0.5B and various misaligned model organisms
- **SAE Feature Exploration**: Visualize any of the 28,672 SAE features
- **Interactive Visualization**: See how features activate across tokens in model responses
- **Custom Questions**: Add your own questions and see how features respond
- **Real-time Processing**: Live model inference with activation extraction

## Architecture

### Backend (ngrok server)
- FastAPI server for model inference and feature extraction
- SAE loading and latent extraction
- Activation caching system
- RESTful API endpoints

### Frontend (Vercel deployment)
- Next.js React application
- Model and feature selection interface
- Neuronpedia-style feature visualization
- Real-time data fetching

## Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- CUDA-capable GPU (for model inference)
- ngrok account and CLI

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download SAE model:
```bash
# Download from Hugging Face
huggingface-cli download rootxhacker/Qwen-2.5-0.5B-instruct-SAE --local-dir ./sae_model
```

3. Start the backend server:
```bash
python src/backend/server.py
```

4. In another terminal, start ngrok:
```bash
python src/backend/ngrok_setup.py
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Set environment variables:
```bash
# Create .env.local
NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok.io
```

3. Start development server:
```bash
npm run dev
```

### Data Collection

To pre-collect data for all models and questions:

```bash
python src/data_collection.py
```

## Usage

1. **Select Model**: Choose from available Qwen2.5-0.5B models
2. **Select Feature**: Pick any SAE feature (0-28671)
3. **Choose Question**: Select from default questions or add your own
4. **Explore**: View feature activations across tokens with interactive visualization

## API Endpoints

- `GET /models` - List available models
- `GET /questions` - List default questions
- `POST /process_question` - Process new question and extract activations
- `POST /get_feature_activations` - Get activations for specific feature
- `POST /get_top_features` - Get top-k most active features

## Project Structure

```
├── src/
│   ├── backend/           # FastAPI server and ngrok setup
│   ├── cache/            # Enhanced caching utilities
│   ├── models/           # Model loading and inference
│   ├── sae/              # SAE loading and feature extraction
│   └── data_collection.py # Data collection script
├── frontend/             # Next.js React application
├── activation_cache.py   # Original activation cache utilities
└── requirements.txt      # Python dependencies
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
Deploy the frontend to Vercel for public access:

```bash
# Build for production
cd frontend
npm run build

# Deploy to Vercel
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
