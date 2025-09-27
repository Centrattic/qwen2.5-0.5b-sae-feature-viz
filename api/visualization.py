"""
Vercel serverless function for SAE feature visualization API
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_visualization_data(model_name: str,
                            question_hash: str) -> Optional[Dict]:
    """Load pre-computed visualization data"""
    viz_dir = Path("./visualization_data")
    model_viz_dir = viz_dir / model_name.replace("/", "_")
    data_file = model_viz_dir / f"{question_hash}.json"

    if not data_file.exists():
        return None

    try:
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def handler(request):
    """Handle API requests"""
    if request.method == "GET":
        # Handle GET requests (health check, etc.)
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({
                "ok": True,
                "mode": "static_data"
            })
        }

    elif request.method == "POST":
        # Handle POST requests for feature data
        try:
            body = json.loads(request.body)
            model_name = body.get("model_name")
            question_hash = body.get("question_hash")
            feature_idx = body.get("feature_idx", 0)

            # Load visualization data
            viz_data = load_visualization_data(model_name, question_hash)
            if viz_data is None:
                return {
                    "statusCode": 404,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": json.dumps({"error": "Data not found"})
                }

            # Get feature data
            features = viz_data.get("features", {})
            feature_key = str(feature_idx)

            if feature_key not in features:
                return {
                    "statusCode":
                    404,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body":
                    json.dumps({"error": f"Feature {feature_idx} not found"})
                }

            feature_data = features[feature_key]
            activations = feature_data["activations"]
            tokens = [f"token_{i}" for i in range(len(activations))]

            return {
                "statusCode":
                200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body":
                json.dumps({
                    "activations": activations,
                    "tokens": tokens,
                    "success": True
                })
            }

        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"error": str(e)})
            }

    else:
        return {
            "statusCode": 405,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": "Method not allowed"})
        }
