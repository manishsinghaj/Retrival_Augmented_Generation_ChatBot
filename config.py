import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyDYMZTLdhXSSlKE4BpyVWcr-ESxB-A30pA")

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text
