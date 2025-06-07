from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import plotly.express as px
import io
import base64
import os
import traceback

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"

def query_llama3(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 1024
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq API Error: {str(e)}"

def parse_detailed_insights(text):
    insights = []
    current = {}

    for line in text.strip().split("\n"):
        line = line.strip()

        if line.lower().startswith("title:") or \
           line.lower().startswith("**") or \
           (line and line[0].isdigit() and "." in line[:3]):
            if current: insights.append(current)
            title = line.replace("**", "").split(":", 1)[-1].strip()
            current = {"title": title}

        elif line.lower().startswith("explanation:"):
            current["explanation"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("chart:"):
            current["chart"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("columns:"):
            cols = line.split(":", 1)[-1].strip().split(",")
            current["columns"] = [c.strip() for c in cols if c.strip()]
        elif line.lower().startswith("tip:"):
            current["tip"] = line.split(":", 1)[-1].strip()

    if current and current.get("title"):
        insights.append(current)
    return insights

def plot_chart(df, insight):
    try:
        cols = insight.get("columns", [])
        if not cols or any(c not in df.columns for c in cols):
            return None, "Missing or invalid columns"

        col1 = cols[0]
        col2 = cols[1] if len(cols) > 1 else None

        fig = None
        if col2:
            if df[col1].dtype == 'object' and df[col2].dtype == 'object':
                fig = px.bar(df.groupby([col1, col2]).size().reset_index(name='count'),
                             x=col1, y='count', color=col2, title=f"{col1} vs {col2}")
            elif df[col1].dtype != 'object' and df[col2].dtype != 'object':
                fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
            else:
                fig = px.bar(df.groupby(col1)[col2].mean().reset_index(),
                             x=col1, y=col2, title=f"{col1} vs {col2}")
        else:
            if df[col1].dtype == 'object':
                vc = df[col1].value_counts().reset_index(name="count")
                vc.columns = [col1, "count"]
                fig = px.bar(vc, x=col1, y="count", title=f"Distribution: {col1}")
            else:
                fig = px.histogram(df, x=col1, title=f"Histogram: {col1}")

        if fig:
            buffer = io.BytesIO()
            fig.write_image(buffer, format='png')
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return encoded, None
    except Exception as e:
        return None, str(e)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = file.filename
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        summary = df.describe(include='all').to_string()
        prompt = f"""You are a business data analyst.

Given the dataset summary below:
{summary}

Extract 5 powerful business insights.

For each:
- Title: short and clear
- Explanation: why this insight matters
- Chart: type of chart to visualize it (e.g., bar, scatter)
- Columns: columns needed for the chart
- Tip: business recommendation

Format like:
Title: ...
Explanation: ...
Chart: ...
Columns: ...
Tip: ...
"""

        ai_response = query_llama3(prompt)
        print("🧠 Raw AI response:", ai_response)
        insights = parse_detailed_insights(ai_response)

        if not insights:
            return jsonify({"error": "No insights extracted", "raw": ai_response}), 500

        chart, error = plot_chart(df, insights[0])
        return jsonify({
            "insights": insights,
            "chart": chart,
            "error": error
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
