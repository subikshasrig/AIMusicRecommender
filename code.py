#system and utilities
import os  #file and path operations
import time  #measure time or add delays
import json  #read/write JSON data
import requests  #make HTTP API calls
from IPython.display import clear_output  

#core scientific libraries
import numpy as np  #numerical computations
import pandas as pd  #data manipulation and analysis
import torch  #GPU acceleration and tensor operations

#ml and nlp 
from sentence_transformers import SentenceTransformer, util  #text embeddings and similarity
from sklearn.metrics.pairwise import cosine_similarity  #cosine similarity computation
from sklearn.preprocessing import normalize  #normalize data vectors
from sklearn.decomposition import PCA  #dimensionality reduction (for visualization)
from huggingface_hub import hf_hub_download  #download pretrained models
from openai import OpenAI  #interact with OpenAI API

#visualization and ui
import plotly.graph_objs as go  #interactive visualizations
import plotly.io as pio  #plotly rendering interface
import gradio as gr  #build interactive web UIs

#reading CSV file 
csv_path = '/content/top_4500_songs.csv'
df = pd.read_csv(csv_path)
df.head()

num_songs = len(df)
print(f"No of Songs: {num_songs}")

df = df.dropna(subset=['artists', 'album_name', 'track_name', 'track_genre'])

#Define a dictionary of available embedding models and their Hugging Face repo IDs
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",  # Fast and lightweight
    "MPNet": "sentence-transformers/all-mpnet-base-v2",  # High accuracy general-purpose model
    "BGE": "BAAI/bge-small-en-v1.5"  # Compact but strong performance for English
}

#Choose which models you want to load (you can pick one or multiple)
selected_models = ['BGE', 'MPNet', 'MiniLM']

#Loop through and download + load each selected model
for model_name in selected_models:
    repo_id = embedding_models[model_name]

    #Try downloading model config to check if it exists
    try:
        hf_hub_download(repo_id=repo_id, filename='config.json')
    except:
        pass

    #Load the SentenceTransformer model from Hugging Face
    embedding_models[model_name] = SentenceTransformer(repo_id)
    print(f"✅ Embedding model '{model_name}' loaded successfully.")

#Precompute embeddings for dictionary (for each model)
dict_embeddings = {}
for model_name in selected_models:
    model = embedding_models[model_name]
    dict_embeddings[model_name] = model.encode(word_dict, convert_to_tensor=True)

#Function to find top N matches
def find_top_matches(user_input, top_n=5):
    results = {}
    for model_name in selected_models:
        model = embedding_models[model_name]
        input_emb = model.encode(user_input, convert_to_tensor=True)
        similarities = util.cos_sim(input_emb, dict_embeddings[model_name])[0]
        top_indices = torch.topk(similarities, k=top_n).indices
        top_words = [word_dict[idx] for idx in top_indices]
        top_scores = [similarities[idx].item() for idx in top_indices]
        results[model_name] = list(zip(top_words, top_scores))
    return results

user_input = input("Enter your text: ")
matches = find_top_matches(user_input)
for model_name, top_matches in matches.items():
    print(f"\nTop matches using {model_name}:")
    for word, score in top_matches:
        print(f"{word} — {score:.4f}")

#Install and import required libraries
!pip install jsonlines
import jsonlines
from sklearn.preprocessing import normalize
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from sklearn.decomposition import PCA

#Function to convert song metadata into a descriptive text string
def song_to_text(row):
    """
    Convert a song's metadata row into a descriptive text string
    that summarizes its genre, audio features, and explicitness.
    """
    explicit_text = "explicit" if row['explicit'] else "clean"
    return (
        f"genre: {row['track_genre']}. "
        f"This song is {row['danceability']*100:.0f}% danceable, "
        f"energy level {row['energy']*100:.0f}%, "
        f"loudness {row['loudness']} dB, "
        f"speechiness {row['speechiness']*100:.0f}%, "
        f"acousticness {row['acousticness']*100:.0f}%, "
        f"instrumentalness {row['instrumentalness']*100:.0f}%, "
        f"liveness {row['liveness']*100:.0f}%, "
        f"valence {row['valence']*100:.0f}%, "
        f"tempo {row['tempo']} BPM. "
        f"It is an {explicit_text} track."
    )

#Generate descriptive texts for all songs in the dataframe
song_texts = df.apply(song_to_text, axis=1)

#Dictionary to store embeddings for different models
model_embeddings = {}

#Set plotly renderer for Google Colab (for interactive plots)
pio.renderers.default = "colab"

#Generate embeddings for each selected model
for name in selected_models:
    print(f"Generating embeddings with {name}...")

    #Load the embedding model
    model_path = embedding_models[name]
    if isinstance(model_path, SentenceTransformer):
        model = model_path
    else:
        model = SentenceTransformer(str(model_path))

    #Encode all song texts to embeddings and normalize them
    emb = model.encode(song_texts.tolist(), show_progress_bar=True, convert_to_numpy=True)
    emb = normalize(emb) #Normalize to unit vectors
    model_embeddings[name] = emb

    #Save embeddings to a JSONL file with track_id, text, and embedding
    output_file = f"{name}_song_embeddings.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        for idx, (_, row) in enumerate(df.iterrows()):
            writer.write({
                "track_id": row['track_id'],
                "text": song_texts.iloc[idx],
                "embedding": emb[idx].tolist()
            })

    print(f"Text embeddings for {name} saved successfully as JSONL: {output_file}")

print("All selected model embeddings generated, normalized, and stored in JSONL format.")


#Function to visualize embeddings from multiple models in 3D
def visualize_embeddings_3d_interactive(embeddings_dict, df, n_samples=None):
    """
    Create an interactive 3D scatter plot of embeddings from multiple models.
    Allows selection of models and comparison between them.

    Parameters:
    - embeddings_dict: dict of model_name -> embeddings (numpy arrays)
    - df: dataframe containing song metadata
    - n_samples: number of random samples to visualize (for performance)
    """

    all_traces = []  #List to store Plotly traces
    model_names = list(embeddings_dict.keys())

    #Choose a subset of songs if n_samples is specified
    if n_samples is not None and n_samples < len(df):
        sample_indices = np.random.choice(len(df), n_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = np.arange(len(df))
        n_samples = len(df)

    sampled_df = df.iloc[sample_indices].reset_index(drop=True)
    song_names = sampled_df['track_name'].tolist()

    color_schemes = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']

    #Create a 3D scatter trace for each model
    for idx, model_name in enumerate(model_names):
        emb = embeddings_dict[model_name][sample_indices]

        #Reduce embedding dimensionality to 3 for visualization using PCA
        pca = PCA(n_components=3)
        emb_3d = pca.fit_transform(emb)

        #Prepare hover text with song and artist info
        hover_texts = []
        for i, row_idx in enumerate(sample_indices):
            row = df.iloc[row_idx]
            hover_text = (
                f"<b>{row['track_name']}</b><br>"
                f"Artist: {row['artists']}<br>"
                f"Genre: {row['track_genre']}<br>"
                f"Model: {model_name}"
            )
            hover_texts.append(hover_text)

        #Create the Plotly scatter3d trace
        trace = go.Scatter3d(
            x=emb_3d[:, 0],
            y=emb_3d[:, 1],
            z=emb_3d[:, 2],
            mode='markers+text',
            name=model_name,
            text=sampled_df['track_name'],
            hovertext=hover_texts,
            hoverinfo='text',
            visible=True if idx == 0 else 'legendonly',
            marker=dict(
                size=6,
                color=np.arange(len(emb_3d)),
                colorscale=color_schemes[idx % len(color_schemes)],
                showscale=True,
                colorbar=dict(
                    title=model_name,
                    x=1 + (idx * 0.15),
                    len=0.5
                ),
                line=dict(width=0.5, color='white')
            ),
            textposition='top center',
            textfont=dict(size=8)
        )
        all_traces.append(trace)

    #Build the figure with all traces
    fig = go.Figure(data=all_traces)

    #Layout settings for the 3D scatter plot
    fig.update_layout(
        title={
            'text': f'3D Embeddings Comparison - {n_samples} Songs Across {len(model_names)} Models',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis=dict(title='PC1', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title='PC2', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title='PC3', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        updatemenus=[  #Dropdown menu for selecting models
            dict(
                buttons=[  #Buttons for all models, individual models, and pairwise comparison
                    dict(
                        label="All Models",
                        method="update",
                        args=[{"visible": [True] * len(model_names)}]
                    )
                ] + [
                    dict(
                        label=model_name,
                        method="update",
                        args=[{"visible": [i == idx for i in range(len(model_names))]}]
                    )
                    for idx, model_name in enumerate(model_names)
                ] + [
                    dict(
                        label=f"Compare: {model_names[i]} vs {model_names[j]}",
                        method="update",
                        args=[{"visible": [idx in [i, j] for idx in range(len(model_names))]}]
                    )
                    for i in range(len(model_names))
                    for j in range(i+1, len(model_names))
                ],
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.85,
                yanchor="top",
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
        ],
        annotations=[  #Label for dropdown
            dict(
                text="Select Models:",
                x=0.02,
                y=0.88,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        ]
    )

    #Save the interactive plot to an HTML file
    filename = f'embeddings_3d_all_models_{n_samples}_songs.html'
    fig.write_html(filename)

    return fig

#Visualize embeddings interactively
fig = visualize_embeddings_3d_interactive(model_embeddings, df, n_samples=30)

import json
import numpy as np
import os

def auto_convert_embeddings():

    #detect files in current directory
    base_dir = os.getcwd()
    all_files = os.listdir(base_dir)

    text_jsonls = [f for f in all_files if f.endswith("_song_embeddings.jsonl")]
    audio_jsonl = next((f for f in all_files if f == "Final_Audio_Embeddings.jsonl"), None)

    if not text_jsonls:
        print("⚠️ No text embedding JSONL files found.")
        return
    if not audio_jsonl:
        print("⚠️ No audio embedding JSONL file named 'audio_embedding.jsonl' found.")
        return

    #Create output directory for .npy files
    output_dir = os.path.join(base_dir, "converted_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    #Load audio embeddings into a dict keyed by track_id 
    print("Loading audio embeddings...")
    audio_data = {}
    with open(audio_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            track_id = data.get("file_name", "").replace(".mp3", "")
            audio_data[track_id] = data.get("embedding", [])
    print(f"✅ Loaded {len(audio_data)} audio embeddings from {audio_jsonl}")

    #Process each text embedding model 
    for text_jsonl in text_jsonls:
        model_name = text_jsonl.replace("_song_embeddings.jsonl", "")
        print(f"\nProcessing model: {model_name}")

        #Load text embeddings into a dict keyed by track_id
        text_data = {}
        with open(text_jsonl, "r") as f:
            for line in f:
                data = json.loads(line)
                track_id = str(data.get("track_id", ""))
                text_data[track_id] = data

        print(f"Loaded {len(text_data)} text embeddings for {model_name}.")

        #Save text-only embeddings as .npy 
        text_only_list = [
            {
                "track_id": tid,
                "embedding": entry.get("embedding", None),
                "text": entry.get("text", "")
            }
            for tid, entry in text_data.items()
        ]
        text_only_path = os.path.join(output_dir, f"{model_name}_text_embeddings.npy")
        np.save(text_only_path, text_only_list)
        print(f"💾 Saved text-only embeddings → {text_only_path} ({len(text_only_list)} entries)")

        #Save matched text+audio embeddings 
        matched = []
        for tid, entry in text_data.items():
            if tid in audio_data:
                matched.append({
                    "track_id": tid,
                    "text_embedding": entry.get("embedding", None),
                    "audio_embedding": audio_data[tid],
                    "text": entry.get("text", "")
                })
        matched_path = os.path.join(output_dir, f"{model_name}_text_audio_embeddings.npy")
        np.save(matched_path, matched)
        print(f"💾 Saved text+audio embeddings → {matched_path} ({len(matched)} matches)")

    #Save audio-only embeddings 
    audio_only_list = [
        {"track_id": tid, "audio_embedding": emb}
        for tid, emb in audio_data.items()
    ]
    audio_npy_path = os.path.join(output_dir, "audio_embeddings.npy")
    np.save(audio_npy_path, audio_only_list)
    print(f"\n💾 Saved standalone audio embeddings → {audio_npy_path} ({len(audio_only_list)} entries)")

    print("\n✅ All conversions complete!")
    print(f"📂 Output directory: {output_dir}")
    print(f"Total files generated: {len(text_jsonls) * 2 + 1}")

#Run automatically 
auto_convert_embeddings()

#Set the OpenRouter API key as an environment variable.
#This allows the client to authenticate requests securely without hardcoding the key in the code.
os.environ["OPENROUTER_API_KEY"] = "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"

#Initialize the OpenAI client with the OpenRouter base URL and the API key retrieved from environment variables.
#This client will be used to send requests to the model for generating responses.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

#Define a function that explains why a user might like a recommended song based on a song they already like.
def explain_recommendation(user_song, recommended_song):
    #Create a prompt that asks the AI to provide a short explanation
    #focusing on aspects like mood, energy, or style of the recommended song.
    prompt = (
        f"User likes the song '{user_song}'. Suggest why they might like '{recommended_song}' "
        f"in 2-3 sentences focusing on mood, energy, or style."
    )

    #Send a request to the chat completion endpoint of the model.
    #The model used here is 'qwen/qwen3-coder:free', but it could be replaced with another model.
    #'messages' is structured as a chat, with a user role sending the prompt.
    #'max_tokens' sets the maximum length of the generated response.
    response = client.chat.completions.create(
        model="qwen/qwen3-coder:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=999
    )

    #Extract the AI-generated explanation from the response object.
    explanation = response.choices[0].message.content

    #Return the explanation to whoever called the function.
    return explanation

!pip install yt_dlp
!pip install muq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from yt_dlp import YoutubeDL
import librosa
import tempfile
import os
import json
import torch

#Global variables for audio embeddings
audio_embedding_data = None
audio_embeddings_matrix = None
audio_track_ids = None

#Initialize MuQ model (load once globally)
try:
    from muq import MuQ
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq_model = muq_model.to(device).eval()
    MUQ_AVAILABLE = True
    print(f"✓ MuQ model loaded on {device}")
except Exception as e:
    MUQ_AVAILABLE = False
    print(f"⚠️ MuQ model not available: {e}")

def load_audio_embeddings():
    """
    Load audio embeddings from JSONL file and create mapping to dataset.
    """
    global audio_embedding_data, audio_embeddings_matrix, audio_track_ids

    if audio_embedding_data is not None:
        return audio_embeddings_matrix, audio_track_ids

    try:
        print("📂 Loading audio embeddings from audio_embeddings.jsonl...")

        embeddings_list = []
        track_ids = []

        with open('Final_Audio_Embeddings.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    filename = data['file_name']
                    embedding = data['embedding']

                    #Extract track_id from filename
                    track_id = filename.replace('.mp3', '').replace('.wav', '').replace('.flac', '')

                    #Convert to numpy array
                    if isinstance(embedding, list):
                        embeddings_list.append(np.array(embedding, dtype=np.float32))
                    else:
                        embeddings_list.append(np.array(embedding, dtype=np.float32))

                    track_ids.append(track_id)
                except json.JSONDecodeError:
                    continue  #Skip invalid lines

        #Stack into matrix
        audio_embeddings_matrix = np.vstack(embeddings_list)
        audio_track_ids = track_ids

        print(f"✓ Loaded {len(track_ids)} audio embeddings with shape {audio_embeddings_matrix.shape}")

        #Verify mapping
        dataset_track_ids = set(df['track_id'].astype(str).values)
        audio_track_ids_set = set(track_ids)
        matched = len(audio_track_ids_set.intersection(dataset_track_ids))
        print(f"✓ Matched {matched}/{len(track_ids)} embeddings with dataset")

        return audio_embeddings_matrix, audio_track_ids

    except FileNotFoundError:
        print("❌ audio_embeddings.jsonl not found")
        return None, None
    except Exception as e:
        print(f"❌ Error loading audio embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def download_youtube_audio(song_name):
    """Download audio from YouTube."""
    try:
        print(f"🔍 Searching YouTube for: {song_name}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'default_search': 'ytsearch1',
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{song_name}", download=True)
            if info and 'entries' in info and len(info['entries']) > 0:
                video_id = info['entries'][0]['id']
                audio_path = os.path.join(tempfile.gettempdir(), f"{video_id}.mp3")
                print(f"✓ Downloaded: {info['entries'][0]['title']}")
                return audio_path
        return None
    except Exception as e:
        print(f"❌ YouTube download failed: {str(e)}")
        return None


def extract_muq_embedding(audio_path, sr=24000):
    """Extract MuQ embedding from audio file."""
    if not MUQ_AVAILABLE:
        print("❌ MuQ model not available")
        return None

    try:
        wav, _ = librosa.load(audio_path, sr=sr)
        wavs = torch.tensor(wav).unsqueeze(0).to(device)

        with torch.no_grad():
            output = muq_model(wavs, output_hidden_states=True)
            embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        print(f"✓ Extracted MuQ embedding (shape: {embedding.shape})")
        return embedding
    except Exception as e:
        print(f"❌ MuQ extraction failed: {str(e)}")
        return None

def recommend_text_only(user_song, selected_model_name='BGE', top_k=5):
    """
    Text-based recommendation using embeddings + audio features.
    """
    if selected_model_name not in embedding_models:
        raise ValueError(f"Model '{selected_model_name}' not loaded.")

    selected_model = embedding_models[selected_model_name]
    embeddings = model_embeddings[selected_model_name]

    #Find song in dataset
    user_row = df[df['track_name'].str.lower() == user_song.lower()]

    #Semantic similarity
    user_embedding = selected_model.encode([user_song], convert_to_numpy=True)
    semantic_sims = cosine_similarity(user_embedding, embeddings).squeeze()

    #CASE 1: Song NOT found
    if user_row.empty:
        print(f"ℹ️ Song '{user_song}' not found. Using semantic matching...")
        popularity_scores = df['popularity'].values / 100.0
        combined_score = semantic_sims * 0.95 + popularity_scores * 0.05
        top_idx = combined_score.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            row = df.iloc[idx]
            results.append({
                "track_name": row['track_name'],
                "artists": row['artists'],
                "explanation": f"Semantic: {semantic_sims[idx]:.3f} | Genre: {row['track_genre']} | Tempo: {row['tempo']:.0f} BPM | Energy: {row['energy']:.2f}"
            })
        return results

    #CASE 2: Song FOUND
    user_row = user_row.iloc[0]
    user_idx = user_row.name
    print(f"✓ Found '{user_row['track_name']}' by {user_row['artists']}")

    #Audio feature similarity
    feature_sims = np.zeros(len(df))
    for i in range(len(df)):
        candidate = df.iloc[i]
        tempo_sim = max(0, 1 - abs(user_row['tempo'] - candidate['tempo']) / 100)
        energy_sim = 1 - abs(user_row['energy'] - candidate['energy'])
        dance_sim = 1 - abs(user_row['danceability'] - candidate['danceability'])
        valence_sim = 1 - abs(user_row['valence'] - candidate['valence'])
        loudness_sim = max(0, 1 - abs(user_row['loudness'] - candidate['loudness']) / 30)
        acoustic_sim = 1 - abs(user_row['acousticness'] - candidate['acousticness'])
        speech_sim = 1 - abs(user_row['speechiness'] - candidate['speechiness'])
        live_sim = 1 - abs(user_row['liveness'] - candidate['liveness'])

        feature_sims[i] = (
            tempo_sim * 0.25 + energy_sim * 0.20 + dance_sim * 0.20 +
            valence_sim * 0.15 + loudness_sim * 0.10 + acoustic_sim * 0.50 +
            speech_sim * 0.03 + live_sim * 0.02
        )

    genre_boost = np.where(df['track_genre'] == user_row['track_genre'], 1.1, 1.0)
    popularity_scores = df['popularity'].values / 100.0

    combined_score = (
        semantic_sims * 0.50 + feature_sims * 0.45 + popularity_scores * 0.05
    ) * genre_boost
    combined_score[user_idx] = -np.inf

    top_idx = combined_score.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        row = df.iloc[i]
        tempo_match = "identical" if abs(user_row['tempo'] - row['tempo']) < 5 else \
                      "very close" if abs(user_row['tempo'] - row['tempo']) < 15 else "similar"
        energy_match = "identical" if abs(user_row['energy'] - row['energy']) < 0.05 else \
                       "very close" if abs(user_row['energy'] - row['energy']) < 0.15 else "similar"

        explanation = (
            f"Semantic: {semantic_sims[i]:.3f} | {tempo_match} tempo ({row['tempo']:.0f} vs {user_row['tempo']:.0f} BPM) | "
            f"{energy_match} energy ({row['energy']:.2f} vs {user_row['energy']:.2f}) | Genre: {row['track_genre']}"
        )

        results.append({
            "track_name": row['track_name'],
            "artists": row['artists'],
            "explanation": explanation
        })

    return results

def recommend_audio_only(user_input, top_k=5, use_muq=True):
    """
    Audio-only recommendation.
    user_input can be: song name (string) OR audio file path (.wav, .mp3)
    """
    is_audio_file = isinstance(user_input, str) and os.path.isfile(user_input)

    if is_audio_file:
        print(f"🎵 Using audio file: {user_input}")

        #Extract MuQ embedding from file
        if not use_muq or not MUQ_AVAILABLE:
            raise ValueError("MuQ model required for audio file input")

        user_muq_embedding = extract_muq_embedding(user_input)
        if user_muq_embedding is None:
            raise ValueError("Failed to extract embedding from audio file")

        user_row = None
        user_idx = -1

    else:
        #Text input - try to find in dataset
        user_song = user_input
        user_row = df[df['track_name'].str.lower() == user_song.lower()]

        if not user_row.empty:
            user_row = user_row.iloc[0]
            user_idx = user_row.name
            print(f"✓ Found '{user_row['track_name']}' by {user_row['artists']}")
            user_muq_embedding = None
        else:
            #Download from YouTube
            print(f"ℹ️ Song '{user_song}' not in dataset. Downloading from YouTube...")
            audio_path = download_youtube_audio(user_song)

            if audio_path is None:
                raise ValueError(f"Could not download '{user_song}'")

            if use_muq and MUQ_AVAILABLE:
                print("🎵 Extracting MuQ embedding...")
                user_muq_embedding = extract_muq_embedding(audio_path)
            else:
                user_muq_embedding = None

            try:
                os.remove(audio_path)
            except:
                pass

            user_idx = -1

    #Calculate audio scores
    audio_scores = np.zeros(len(df))

    if user_muq_embedding is not None:
        #MuQ embedding matching
        dataset_embeddings, track_ids = load_audio_embeddings()

        if dataset_embeddings is not None:
            track_id_to_idx = {tid: idx for idx, tid in enumerate(track_ids)}
            track_id_to_dataset_idx = {str(row['track_id']): idx for idx, row in df.iterrows()}

            temp_scores = cosine_similarity(
                user_muq_embedding.reshape(1, -1),
                dataset_embeddings
            ).squeeze()

            matched_count = 0
            for track_id, emb_idx in track_id_to_idx.items():
                if track_id in track_id_to_dataset_idx:
                    dataset_idx = track_id_to_dataset_idx[track_id]
                    audio_scores[dataset_idx] = temp_scores[emb_idx]
                    matched_count += 1

            print(f"✓ Matched {matched_count} songs using MuQ embeddings")

    elif user_row is not None:
        #Feature-based matching
        for i in range(len(df)):
            candidate = df.iloc[i]
            tempo_sim = max(0, 1 - abs(user_row['tempo'] - candidate['tempo']) / 100)
            energy_sim = 1 - abs(user_row['energy'] - candidate['energy'])
            dance_sim = 1 - abs(user_row['danceability'] - candidate['danceability'])
            valence_sim = 1 - abs(user_row['valence'] - candidate['valence'])
            loudness_sim = max(0, 1 - abs(user_row['loudness'] - candidate['loudness']) / 30)
            acoustic_sim = 1 - abs(user_row['acousticness'] - candidate['acousticness'])
            instr_sim = 1 - abs(user_row['instrumentalness'] - candidate['instrumentalness'])

            audio_scores[i] = (
                tempo_sim * 0.30 + energy_sim * 0.25 + dance_sim * 0.20 +
                valence_sim * 0.12 + loudness_sim * 0.08 + acoustic_sim * 0.03 + instr_sim * 0.02
            )

        genre_boost = np.where(df['track_genre'] == user_row['track_genre'], 1.15, 1.0)
        audio_scores *= genre_boost
        audio_scores[user_idx] = -np.inf

    top_indices = audio_scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        song = df.iloc[idx]

        if user_row is not None and not user_row.empty:
            #Check if user_row is a Series (single row) or DataFrame
            if isinstance(user_row, pd.DataFrame):
                user_data = user_row.iloc[0]
            else:
                user_data = user_row

            tempo_diff = abs(user_data['tempo'] - song['tempo'])
            explanation = (
                f"Tempo: {song['tempo']:.0f} BPM (diff: {tempo_diff:.0f}) | "
                f"Energy: {song['energy']:.2f} | Dance: {song['danceability']:.2f} | "
                f"Genre: {song['track_genre']}"
            )
        else:
            explanation = (
                f"Audio match score: {audio_scores[idx]:.3f} | "
                f"Tempo: {song['tempo']:.0f} BPM | Energy: {song['energy']:.2f} | "
                f"Genre: {song['track_genre']}"
            )

        results.append({
            "track_name": song['track_name'],
            "artists": song['artists'],
            "explanation": explanation
        })

    return results

def recommend_hybrid(user_song, selected_model_name='BGE', top_k=5):
    """
    Hybrid: text embeddings + audio features.
    """
    if selected_model_name not in embedding_models:
        raise ValueError(f"Model '{selected_model_name}' not loaded.")

    selected_model = embedding_models[selected_model_name]
    embeddings = model_embeddings[selected_model_name]

    user_embedding = selected_model.encode([user_song], convert_to_numpy=True)
    semantic_sims = cosine_similarity(user_embedding, embeddings).squeeze()

    user_row = df[df['track_name'].str.lower() == user_song.lower()]

    if user_row.empty:
        print(f"ℹ️ Song not found. Using semantic matching...")
        popularity_scores = df['popularity'].values / 100.0
        combined_score = semantic_sims * 0.95 + popularity_scores * 0.05
    else:
        user_row = user_row.iloc[0]
        user_idx = user_row.name
        print(f"✓ Found '{user_row['track_name']}' - using hybrid approach")

        feature_sims = np.zeros(len(df))
        for i in range(len(df)):
            candidate = df.iloc[i]
            tempo_sim = max(0, 1 - abs(user_row['tempo'] - candidate['tempo']) / 100)
            energy_sim = 1 - abs(user_row['energy'] - candidate['energy'])
            dance_sim = 1 - abs(user_row['danceability'] - candidate['danceability'])
            valence_sim = 1 - abs(user_row['valence'] - candidate['valence'])
            loudness_sim = max(0, 1 - abs(user_row['loudness'] - candidate['loudness']) / 30)
            acoustic_sim = 1 - abs(user_row['acousticness'] - candidate['acousticness'])
            speech_sim = 1 - abs(user_row['speechiness'] - candidate['speechiness'])
            live_sim = 1 - abs(user_row['liveness'] - candidate['liveness'])

            feature_sims[i] = (
                tempo_sim * 0.25 + energy_sim * 0.20 + dance_sim * 0.20 +
                valence_sim * 0.15 + loudness_sim * 0.10 + acoustic_sim * 0.05 +
                speech_sim * 0.03 + live_sim * 0.02
            )

        genre_boost = np.where(df['track_genre'] == user_row['track_genre'], 1.12, 1.0)
        popularity_scores = df['popularity'].values / 100.0
        interaction_bonus = feature_sims * semantic_sims * 0.05

        combined_score = (
            semantic_sims * 0.40 + feature_sims * 0.50 +
            popularity_scores * 0.05 + interaction_bonus
        ) * genre_boost
        combined_score[user_idx] = -np.inf

    top_idx = combined_score.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        row = df.iloc[i]

        if not user_row.empty:
            explanation = (
                f"Hybrid | Semantic: {semantic_sims[i]:.3f} | "
                f"Tempo: {row['tempo']:.0f} vs {user_row['tempo']:.0f} BPM | "
                f"Energy: {row['energy']:.2f} vs {user_row['energy']:.2f} | "
                f"Genre: {row['track_genre']}"
            )
        else:
            explanation = (
                f"Semantic: {semantic_sims[i]:.3f} | "
                f"Tempo: {row['tempo']:.0f} BPM | Energy: {row['energy']:.2f} | "
                f"Genre: {row['track_genre']}"
            )

        results.append({
            "track_name": row['track_name'],
            "artists": row['artists'],
            "explanation": explanation
        })

    return results

def recommend_text_only_llm(user_song, selected_model_name='BGE', top_k=5, llm_client=None):
    """Text-based with LLM explanations."""
    base_recs = recommend_text_only(user_song, selected_model_name, top_k)

    user_row = df[df['track_name'].str.lower() == user_song.lower()]

    results = []
    for rec in base_recs:
        song_row = df[(df['track_name'] == rec['track_name']) & (df['artists'] == rec['artists'])].iloc[0]

        if llm_client and not user_row.empty:
            user_data = user_row.iloc[0]
            prompt = (
                f"Explain why '{song_row['track_name']}' by {song_row['artists']} is recommended "
                f"for someone who likes '{user_song}' by {user_data['artists']}. "
                f"Original: {user_data['track_genre']}, {user_data['tempo']:.0f} BPM, energy {user_data['energy']:.2f}. "
                f"Recommended: {song_row['track_genre']}, {song_row['tempo']:.0f} BPM, energy {song_row['energy']:.2f}. "
                f"Focus on musical similarities in 2-3 sentences."
            )

            response = llm_client.chat.completions.create(
                model="qwen/qwen-2.5-coder-32b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            explanation = response.choices[0].message.content.strip()
        else:
            explanation = rec['explanation']

        results.append({
            "track_name": rec['track_name'],
            "artists": rec['artists'],
            "explanation": explanation
        })

    return results

def recommend_audio_only_llm(user_input, top_k=5, use_muq=True, llm_client=None):
    """Audio-based with LLM explanations."""
    base_recs = recommend_audio_only(user_input, top_k, use_muq)

    is_audio_file = isinstance(user_input, str) and os.path.isfile(user_input)
    if not is_audio_file:
        user_row = df[df['track_name'].str.lower() == user_input.lower()]
    else:
        user_row = None

    results = []
    for rec in base_recs:
        song_row = df[(df['track_name'] == rec['track_name']) & (df['artists'] == rec['artists'])].iloc[0]

        if llm_client and user_row is not None and not user_row.empty:
            user_data = user_row.iloc[0]
            prompt = (
                f"Explain why '{song_row['track_name']}' matches '{user_input}' based on AUDIO characteristics. "
                f"Original: {user_data['tempo']:.0f} BPM, energy {user_data['energy']:.2f}, dance {user_data['danceability']:.2f}. "
                f"Recommended: {song_row['tempo']:.0f} BPM, energy {song_row['energy']:.2f}, dance {song_row['danceability']:.2f}. "
                f"Focus on rhythm and sonic qualities in 2-3 sentences."
            )

            response = llm_client.chat.completions.create(
                model="qwen/qwen-2.5-coder-32b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            explanation = response.choices[0].message.content.strip()
        else:
            explanation = rec['explanation']

        results.append({
            "track_name": rec['track_name'],
            "artists": rec['artists'],
            "explanation": explanation
        })

    return results

def recommend_hybrid_llm(user_song, selected_model_name='BGE', top_k=5, llm_client=None):
    """Hybrid with LLM explanations."""
    base_recs = recommend_hybrid(user_song, selected_model_name, top_k)

    user_row = df[df['track_name'].str.lower() == user_song.lower()]

    results = []
    for rec in base_recs:
        song_row = df[(df['track_name'] == rec['track_name']) & (df['artists'] == rec['artists'])].iloc[0]

        if llm_client and not user_row.empty:
            user_data = user_row.iloc[0]
            prompt = (
                f"Provide a compelling explanation for why '{song_row['track_name']}' by {song_row['artists']} "
                f"is recommended for someone who enjoys '{user_song}' by {user_data['artists']}. "
                f"Original: {user_data['track_genre']}, {user_data['tempo']:.0f} BPM, energy {user_data['energy']:.2f}, mood {user_data['valence']:.2f}. "
                f"Recommended: {song_row['track_genre']}, {song_row['tempo']:.0f} BPM, energy {song_row['energy']:.2f}, mood {song_row['valence']:.2f}. "
                f"Combine thematic and sonic connections in 3-4 engaging sentences."
            )

            response = llm_client.chat.completions.create(
                model="qwen/qwen-2.5-coder-32b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            explanation = response.choices[0].message.content.strip()
        else:
            explanation = rec['explanation']

        results.append({
            "track_name": rec['track_name'],
            "artists": rec['artists'],
            "explanation": explanation
        })

    return results

def detect_recommendation_request(message):
    """
    Detect if user is asking for song recommendations.
    Improved: Better pattern matching, handles more variations, extracts song name more robustly.
    Returns: (is_request, song_name, recommendation_type)
    """
    message_lower = message.lower().strip()

    #Expanded patterns for recommendation requests
    patterns = {
        'recommend': ['recommend', 'suggest', 'suggestions for', 'find me', 'give me', 'what are some'],
        'similar': ['similar to', 'like', 'sounds like', 'reminds me of', 'in the style of'],
        'based_on': ['based on', 'if i like', 'from', 'inspired by'],
        'like': ['i like', 'i love', 'i enjoy', 'my favorite is', 'i\'m into']
    }

    #Song contexts to confirm it's music-related
    song_contexts = ['song', 'music', 'track', 'artist', 'band', 'album', 'playlist', 'tune', 'hit']
    has_song_context = any(context in message_lower for context in song_contexts) or any(pattern in message_lower for sublist in patterns.values() for pattern in sublist)

    #Extract song name
    song_name = None
    is_request = False

    #Check patterns in order of specificity
    for category, phrases in patterns.items():
        for phrase in phrases:
            if phrase in message_lower:
                is_request = True
                parts = message_lower.split(phrase, 1)
                if len(parts) > 1:
                    potential_song = parts[1].strip()
                    #Refine extraction: look for quotes, or split by common separators
                    if '"' in potential_song:
                        song_name = potential_song.split('"')[1]
                    elif "'" in potential_song:
                        song_name = potential_song.split("'")[1]
                    elif 'by' in potential_song:
                        song_name = potential_song.split('by')[0].strip()
                    else:
                        song_name = potential_song
                break
        if is_request:
            break

    #Fallback: if no specific phrase but has "like [song]" or similar
    if not song_name and 'like' in message_lower and has_song_context:
        song_name = message_lower.split('like', 1)[-1].strip()
        is_request = True

    #Clean up song name
    if song_name:
        remove_words = ['songs', 'music', 'tracks', 'by', 'the', 'a', 'an', 'some', 'me', 'please', '?', '!', '.']
        for word in remove_words:
            song_name = song_name.replace(word, '').strip()
        #Capitalize properly if needed
        song_name = song_name.title()
        if len(song_name) < 3:
            is_request = False
            song_name = None

    #Determine recommendation type (improved: more keywords)
    rec_type = 'hybrid'  # default
    if any(word in message_lower for word in ['audio', 'sound', 'rhythm', 'beat', 'sonic', 'instrumental']):
        rec_type = 'audio_only'
    elif any(word in message_lower for word in ['text', 'lyrics', 'name', 'semantic', 'theme', 'story']):
        rec_type = 'text_only'

    return is_request, song_name, rec_type

def chat_with_recommendations(message, history=None, selected_model_name='BGE'):

    if history is None:
        history = []

    #Format history from Gradio format [[user, bot], ...] to list of dicts
    formatted_history = []
    for pair in history:
        if isinstance(pair, list) and len(pair) == 2:
            user_msg, bot_msg = pair
            if user_msg:
                formatted_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                formatted_history.append({"role": "assistant", "content": bot_msg})
        else:
            #If not standard Gradio, assume it's already dicts or skip
            pass

    is_rec_request, song_name, rec_type = detect_recommendation_request(message)

    if not is_rec_request or not song_name:
        #Normal chat (non-recommendation)
        if 'client' not in globals():
            return "Sorry, I can't respond right now without an LLM."
        system_prompt = "You are a helpful music chatbot. Respond naturally."
        messages = [{"role": "system", "content": system_prompt}] + formatted_history + [{"role": "user", "content": message}]
        #print(messages)  # Debug: Uncomment to check structure
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-coder:free",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in API call: {str(e)}"

    #Recommendation request
    try:
        #Call the appropriate recommendation function
        if rec_type == 'text_only':
            recs = recommend_text_only(song_name, selected_model_name, top_k=5)
        elif rec_type == 'audio_only':
            recs = recommend_audio_only(song_name, top_k=5)
        else:
            recs = recommend_hybrid(song_name, selected_model_name, top_k=5)

        #Format results as context
        results_context = "Based on the user's query for recommendations similar to '{}', here are the top results:\n".format(song_name)
        for i, r in enumerate(recs, 1):
            results_context += "{}. '{}' by {} - {}\n".format(
                i, r['track_name'], r['artists'], r.get('explanation', 'No explanation available')
            )

        #If no LLM, return formatted text
        if 'client' not in globals():
            return results_context

        #Send to LLM for natural response
        system_prompt = (
            "You are a friendly music recommendation assistant. "
            "Use the provided results to generate an engaging, conversational response. "
            "Start with something like 'Based on your interest in [song], here are some recommendations:' "
            "Explain briefly why each might appeal, and keep it fun and concise."
        )
        user_prompt = "{}\n{}".format(message, results_context)  # Include original message and results
        messages = [{"role": "system", "content": system_prompt}] + formatted_history + [{"role": "user", "content": user_prompt}]
        #print(messages)  # Debug: Uncomment to check structure
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-coder:free",
                messages=messages,
                max_tokens=500,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in API call: {str(e)}"

    except Exception as e:
        return "Sorry, there was an error generating recommendations: {}".format(str(e))

def chat(message, history=None, selected_model_name='BGE'):
    """
    Improved main chat handler:
    - Uses chat_with_recommendations for all logic.
    - Simplifies: directly calls it and returns the response.
    - Assumes Gradio or similar will handle history appending.
    """
    response = chat_with_recommendations(message, history, selected_model_name)
    return response

chat_interface = gr.ChatInterface(
    fn=chat,
    title="🎵 Music Recommendation Chatbot",
    description="Ask for song recommendations or chat with the bot."
)

chat_interface.launch(share=True, debug=True)




