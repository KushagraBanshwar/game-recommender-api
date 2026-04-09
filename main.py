from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (HF handles this fine)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Supabase
SUPABASE_URL = "https://dldlktgtpynkiiidiqwp.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class Query(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/recommend")
def recommend(query: Query):
    query_embedding = model.encode(
        query.prompt,
        normalize_embeddings=True
    ).tolist()

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

        score = (similarity * 0.8) + (np.log1p(recs) * 0.2)
        final_results.append((r, score))

    final_results.sort(key=lambda x: x[1], reverse=True)

    return {
        "results": [r[0] for r in final_results[:20]]
    }
