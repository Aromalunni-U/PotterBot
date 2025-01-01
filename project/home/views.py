import os
import faiss
from django.conf import settings
from django.shortcuts import render
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


api_key = "api_key"
genai.configure(api_key=api_key)
genai_model = genai.GenerativeModel("gemini-pro")


file_path = os.path.join(settings.BASE_DIR, "home", "potter_files", "Harry_Potter_all_books_preprocessed.txt")
faiss_index_path = os.path.join(settings.BASE_DIR, "home", "potter_files", "harry_potter_index.faiss")

file = open(file_path, "r")
story_text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
story_chunks = splitter.split_text(story_text)

def index(request):
    text = " "
    if request.method == "POST":
        prompt = request.POST.get("question")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        # index = faiss.read_index(faiss_index_path)

        try:
            index = faiss.read_index(faiss_index_path)
        except FileNotFoundError:
            print(f"File not found: {faiss_index_path}")
        
        else:
            query = model.encode([prompt])
            distance, indices = index.search(query,k=5)
            
            relevant_chunks = " ".join([story_chunks[i] for i in indices[0]])
            print("Relavent chunks",relevant_chunks)


            ai_prompt = f"Based on the following story chunks, answer the question: {prompt}\n\nStory chunks:{relevant_chunks}\n\nUse only the chunks informations "
            response = genai_model.generate_content(ai_prompt)
            text = response.text
            print(text)
        
    return render(request,"index.html",{"text":text})
