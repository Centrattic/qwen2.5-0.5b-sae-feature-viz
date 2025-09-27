# SAE Feature Visualization - Deployment Guide

## 🚀 **Complete Deployment Steps**

### 1. **Collect Data (One-time)**
```bash
# Run data collection to pre-compute all activations and SAE latents
python3 collect_data.py
```

### 2. **Prepare for Vercel Deployment**
```bash
# Copy visualization data to frontend directory
python3 deploy_to_vercel.py
```

### 3. **Deploy to Vercel**

#### Option A: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

#### Option B: GitHub + Vercel Dashboard
1. Push to GitHub:
   ```bash
   git add .
   git commit -m "Add SAE feature visualization"
   git push origin main
   ```

2. Connect to Vercel:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Deploy!

## 📁 **Project Structure for Vercel**

```
qwen2.5-0.5b-sae-feature-viz/
├── src/deployment/vercel-frontend/    # Static frontend files
│   ├── index.html                     # Main app
│   ├── config.js                      # Configuration
│   └── visualization_data/            # Pre-computed data (copied by deploy script)
├── vercel.json                        # Vercel configuration
└── api/                               # Serverless functions (optional)
```

## 🔧 **How It Works**

1. **Data Collection**: `collect_data.py` pre-computes all activations and SAE latents
2. **Static Deployment**: Frontend loads data directly from JSON files
3. **No Backend Required**: Everything runs client-side
4. **Fast Loading**: Pre-computed data loads instantly

## 📊 **Data Flow**

```
Models + SAE → collect_data.py → visualization_data/ → Vercel → Frontend
```

## ✅ **Verification**

After deployment, your app will:
- Load instantly (no model loading)
- Show SAE feature visualizations
- Work offline (static data)
- Scale automatically (Vercel CDN)

## 🐛 **Troubleshooting**

- **Data not found**: Run `python3 collect_data.py` first
- **Deployment fails**: Check `vercel.json` configuration
- **Frontend errors**: Check browser console for data loading issues
