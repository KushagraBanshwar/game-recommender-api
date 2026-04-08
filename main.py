from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 GLOBAL MODEL (initially None)
model = None

# 🔥 LOAD MODEL AFTER SERVER STARTS
@app.on_event("startup")
def load_model():
    global model
    try:
        print("Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully ✅")
    except Exception as e:
        print("Model failed to load ❌", e)

# CONFIG
SUPABASE_URL = "https://dldlktgtpynkiiidiqwp.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Request schema
class Query(BaseModel):
    prompt: str

# API endpoint
@app.post("/recommend")
def recommend(query: Query):
    global model

    if model is None:
        return {"error": "Model still loading, try again in a few seconds"}

    user_input = query.prompt

    # Encode query
    query_embedding = model.encode(
        user_input,
        normalize_embeddings=True
    ).tolist()

    # Fetch candidates
    response = supabase.rpc("match_games", {
        "query_embedding": query_embedding,
        "match_count": 50
    }).execute()

    results = response.data

    final_results = []

    for r in results:
        similarity = r.get("similarity", 0)
        recs = r.get("recommendations", 0) or 0

        try:
            recs = int(recs)
        except:
            recs = 0

        popularity_score = np.log1p(recs)

        # 🎯 final hybrid score
        score = (similarity * 0.8) + (popularity_score * 0.2)

        final_results.append((r, score))

    # sort
    final_results.sort(key=lambda x: x[1], reverse=True)

    # top 20
    top_games = [r[0] for r in final_results[:20]]

    return {
        "results": top_games
    }


# 🔥 LOCAL RUN SUPPORT (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
