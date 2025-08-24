import os
import requests
import tempfile
from PIL import Image
import whisper
import yt_dlp
from pytesseract import image_to_string
import re
from PyPDF2 import PdfReader
import zipfile
import warnings
from ddgs import DDGS
import wikipedia
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import base64
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Add a new cache for tool calls
TOOL_CACHE = {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define tools
@tool
def web_search(query: str) -> str:
    """Performs web searches using DuckDuckGo (DDG)."""
    if query in TOOL_CACHE:
        return TOOL_CACHE[query]
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        result_str = str(results)
        TOOL_CACHE[query] = result_str
        return result_str
    except Exception as e:
        return f"Error performing DDG search: {str(e)}"

@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for the given query and returns full page content."""
    if query in TOOL_CACHE:
        return TOOL_CACHE[query]
    try:
        page = wikipedia.page(query, auto_suggest=False)
        content = page.content[:4000]  # Limit to avoid overload
        TOOL_CACHE[query] = content
        return content
    except wikipedia.DisambiguationError as e:
        # Take first option
        page = wikipedia.page(e.options[0])
        content = page.content[:4000]
        TOOL_CACHE[query] = content
        return content
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def visit_webpage(url: str) -> str:
    """Retrieves and parses webpage content."""
    if url in TOOL_CACHE:
        return TOOL_CACHE[url]
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()[:8000]  # Limit length
        # Extract tables if present
        tables = pd.read_html(StringIO(str(soup)))
        if tables:
            table_str = "\n".join([df.to_csv(index=False) for df in tables[:3]])  # First 3 tables
            full_content = f"Text: {text}\n\nTables:\n{table_str[:8000]}"
            TOOL_CACHE[url] = full_content
            return full_content
        TOOL_CACHE[url] = text
        return text
    except Exception as e:
        return f"Error visiting webpage: {str(e)}"

@tool
def file_from_url(url: str) -> str:
    """Downloads a file from URL and returns local path."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        _, ext = os.path.splitext(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(response.content)
        return tmp.name
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def read_text_file(path: str) -> str:
    """Reads content from a text file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def extract_pdf_text(path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

@tool
def describe_image(path: str, prompt: str = "Describe this image in extreme detail, including counts of objects, positions, colors, and any text. For tasks involving counting species or items, list them explicitly.") -> str:
    """Describes an image using OpenAI vision model."""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
        ocr_text = ocr_image.invoke({"path": path})
        full_prompt = f"{prompt}\n\nOCR Extracted Text: {ocr_text}"
        message.content[0]["text"] = full_prompt
        return llm.invoke([message]).content
    except Exception as e:
        return f"Error describing image: {str(e)}"

@tool
def transcribe_audio(path: str) -> str:
    """Transcribes audio file to text."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        model = whisper.load_model("tiny")
        result = model.transcribe(path, fp16=False)
        # Basic post-processing: remove filler words, capitalize
        text = result["text"].strip().capitalize()
        return text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

@tool
def transcribe_youtube(url: str) -> str:
    """Transcribes YouTube video audio to text."""
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            ydl_opts = {
                "format": "bestaudio",
                "outtmpl": os.path.join(tempdir, "audio.%(ext)s"),
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audio_file = next(f for f in os.listdir(tempdir) if f.endswith('.mp3'))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            model = whisper.load_model("tiny")
            result = model.transcribe(os.path.join(tempdir, audio_file), fp16=False)
            text = result["text"].strip().capitalize()
            return text
    except Exception as e:
        return f"Error transcribing YouTube: {str(e)}"

@tool
def ocr_image(path: str) -> str:
    """Extracts text from image using OCR."""
    try:
        img = Image.open(path)
        rotations = [0, 90, 180, 270]
        texts = []
        for angle in rotations:
            try:
                rotated = img.rotate(angle, expand=True) if angle else img
                text = image_to_string(rotated)
                texts.append((len(text or ""), text))
            except Exception:
                continue
        if texts:
            texts.sort(key=lambda t: t[0], reverse=True)
            return texts[0][1]
        return image_to_string(img)
    except Exception as e:
        return f"Error in OCR: {str(e)}"

@tool
def extract_table_from_image(path: str) -> str:
    """Attempts to extract a table from an image into CSV-like text."""
    try:
        from pytesseract import image_to_data
        img = Image.open(path)
        data = image_to_data(img, output_type='dict')
        rows = {}
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt:
                continue
            top = data['top'][i]
            # bucket by approximate line using 10px tolerance
            key = top // 10
            rows.setdefault(key, []).append((data['left'][i], txt))
        ordered_lines = []
        for key in sorted(rows.keys()):
            line = rows[key]
            line.sort(key=lambda x: x[0])
            cells = [cell for _, cell in line]
            ordered_lines.append(",".join(cells))
        return "\n".join(ordered_lines)
    except Exception as e:
        return f"Error extracting table: {str(e)}"

@tool
def unzip_file(path: str) -> str:
    """Unzips an archive and returns a brief summary of its contents with previews."""
    try:
        preview = []
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(td)
            for root, _, files in os.walk(td):
                for name in files:
                    p = os.path.join(root, name)
                    e = os.path.splitext(p)[1].lower()
                    preview.append(f"FILE: {name}")
                    if e in [".txt", ".md", ".log"]:
                        try:
                            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                                preview.append(f.read(800))
                        except Exception:
                            pass
                    elif e == ".csv":
                        preview.append(summarize_csv.invoke({"path": p}))
                    elif e in [".xlsx", ".xls"]:
                        preview.append(summarize_excel.invoke({"path": p}))
                    elif e == ".pdf":
                        preview.append(extract_pdf_text.invoke({"path": p}))
                    elif e in [".png", ".jpg", ".jpeg", ".webp"]:
                        preview.append("[image]")
        return ("\n\n".join(preview))[:6000]
    except Exception as e:
        return f"Error unzipping: {str(e)}"

@tool
def readable_webpage(url: str) -> str:
    """Fetches a webpage and extracts readable main text (best-effort)."""
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
            tag.decompose()
        candidates = []
        selectors = [
            "article", "main", "div#content", "div#main", "div.article", "section", "div.content", "div.post"
        ]
        for sel in selectors:
            for el in soup.select(sel):
                txt = el.get_text(" ", strip=True)
                if txt and len(txt) > 200:
                    candidates.append((len(txt), txt))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1][:8000]
        # Fallback to full text
        text = soup.get_text(" ")
        return text[:8000]
    except Exception as e:
        return f"Error extracting readable webpage: {str(e)}"

@tool
def web_research(query: str) -> str:
    """Searches web (top 3) and returns readable extracts with URLs."""
    if query in TOOL_CACHE:
        return TOOL_CACHE[query]
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                url = r.get('href') or r.get('url') or r.get('link')
                if not url:
                    continue
                content = readable_webpage.invoke({"url": url})
                snippet = content if isinstance(content, str) else str(content)
                results.append(f"URL: {url}\nSNIPPET: {r.get('body', '')}\nCONTENT:\n{snippet[:2000]}\n---")
        result_str = "\n\n".join(results)[:8000]
        TOOL_CACHE[query] = result_str
        return result_str
    except Exception as e:
        return f"Error in web research: {str(e)}"

@tool
def summarize_csv(path: str) -> str:
    """Reads full CSV file content as string."""
    try:
        df = pd.read_csv(path)
        full_content = df.to_csv(index=False)
        return full_content[:10000]  # Limit to prevent overload
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

@tool
def summarize_excel(path: str) -> str:
    """Reads full Excel file content as string."""
    try:
        df = pd.read_excel(path)
        full_content = df.to_csv(index=False)
        return full_content[:10000]  # Limit to prevent overload
    except Exception as e:
        return f"Error reading Excel: {str(e)}"

@tool
def python_code_exec(code: str) -> str:
    """Executes Python code and returns the output."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get('result', 'No result defined. Set a "result" variable.'))
    except Exception as e:
        return f"Error executing code: {str(e)}"

tools = [
    web_search, wikipedia_search, visit_webpage, file_from_url, read_text_file,
    describe_image, transcribe_audio, transcribe_youtube, ocr_image,
    summarize_csv, summarize_excel, python_code_exec, extract_pdf_text,
    extract_table_from_image, unzip_file, readable_webpage, web_research
]
