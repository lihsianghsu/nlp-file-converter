import streamlit as st
import pandas as pd
import sqlite3
import re
import html
import os
from lxml import etree
from io import BytesIO
import opencc
import json
from typing import List, Optional

# ==============================
# 1. HELPER FUNCTIONS
# ==============================

def normalize_language(lang: str) -> str:
    """Normalize language variants: en-US, en-GB → en; fr-FR → fr; zh-TW, zh-HK → zht; zh-CN, zh-SG → zhs"""
    if not lang:
        return "unknown"
    lang = lang.strip().lower()
    if lang in ('zh-tw', 'zh-hk', 'zh-mo'):
        return 'zht'
    elif lang in ('zh-cn', 'zh-sg'):
        return 'zhs'
    else:
        return lang.split('-')[0]

def clean_segment_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'<[/]?(?:ph|bpt|ept|it|st|ut|g)[^>]*>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_trailing_lang_code(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"(?:\s*[A-Z]{2})?[a-z]{2}-[A-Z]{2}$", "", text).strip()

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

# ==============================
# 2. PARSERS (for TMX, SDLTM, XLIFF)
# ==============================
import tempfile
import os
import sqlite3
import pandas as pd

def parse_sdltm(file_obj, normalize=True, convert_zh_chars=False):
    """
    Parse .sdltm file uploaded via Streamlit.
    file_obj: UploadedFile from st.file_uploader
    """
    # Step 1: Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sdltm") as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_path = tmp_file.name

    try:
        # Step 2: Connect and list tables
        conn = sqlite3.connect(":memory:")
        conn.execute("ATTACH DATABASE ? AS sdltm", (tmp_path,))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sdltm.sqlite_master WHERE type='table';")
        tables = {row[0].lower() for row in cursor.fetchall()}
        cursor.close()

        # Step 3: Detect TU table
        if 'translationunits' in tables:
            tu_query = "SELECT source_segment, target_segment FROM sdltm.translationunits"
        elif 'translation_units' in tables:
            tu_query = "SELECT source_segment, target_segment FROM sdltm.translation_units"
        elif 'segment_pairs' in tables:
            tu_query = "SELECT source_text, target_text FROM sdltm.segment_pairs"
        else:
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query(tu_query, conn)
        conn.close()

        # Normalize columns
        if 'source_text' in df.columns:
            df.rename(columns={'source_text': 'source_segment', 'target_text': 'target_segment'}, inplace=True)

        # Apply cleaning
        df['source_text'] = df['source_segment'].astype(str).apply(clean_segment_text).apply(remove_trailing_lang_code)
        df['target_text'] = df['target_segment'].astype(str).apply(clean_segment_text).apply(remove_trailing_lang_code)
        df = df.drop(columns=['source_segment', 'target_segment'])
        df = df[(df['source_text'].str.len() > 0) & (df['target_text'].str.len() > 0)]

        if df.empty:
            return df

        # Step 4: Detect languages
        src_lang, tgt_lang = "unknown", "unknown"
        try:
            conn2 = sqlite3.connect(":memory:")
            conn2.execute("ATTACH DATABASE ? AS sdltm", (tmp_path,))
            lang_table = None
            if 'translationmemories' in tables:
                lang_table = 'translationmemories'
            elif 'translation_memories' in tables:
                lang_table = 'translation_memories'
            if lang_table:
                row = conn2.execute(f"SELECT source_language, target_language FROM sdltm.{lang_table} LIMIT 1").fetchone()
                if row:
                    src_lang, tgt_lang = (row[0] or "unknown"), (row[1] or "unknown")
            conn2.close()
        except:
            pass

        if normalize:
            src_lang = normalize_language(src_lang)
            tgt_lang = normalize_language(tgt_lang)

        df['source_lang'] = src_lang
        df['target_lang'] = tgt_lang

        # Optional: Chinese char conversion
        if convert_zh_chars:
            if src_lang == 'zhs' and tgt_lang == 'zht':
                converter = opencc.OpenCC('s2t')
                df['target_text'] = df['target_text'].apply(lambda x: converter.convert(x) if x else x)
            elif src_lang == 'zht' and tgt_lang == 'zhs':
                converter = opencc.OpenCC('t2s')
                df['target_text'] = df['target_text'].apply(lambda x: converter.convert(x) if x else x)

        return df

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


def parse_tmx(file_obj, normalize=True):
    parser = etree.XMLParser(strip_cdata=False, recover=True, encoding='utf-8')
    tree = etree.parse(file_obj, parser)
    root = tree.getroot()
    default_ns = root.nsmap.get(None)
    ns = {'ns': default_ns} if default_ns else {}
    body = root.find('ns:body', ns) if default_ns else root.find('body')
    if body is None:
        body = root.find('body')
    if body is None:
        return pd.DataFrame()
    data = []
    tu_elements = body.xpath('.//ns:tu', namespaces=ns) if default_ns else body.findall('.//tu')
    for tu in tu_elements:
        segs = {}
        tuvs = tu.xpath('./ns:tuv', namespaces=ns) if default_ns else tu.findall('./tuv')
        for tuv in tuvs:
            lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang') or tuv.get('lang')
            if not lang:
                continue
            seg_elem = tuv.xpath('./ns:seg', namespaces=ns)[0] if default_ns else tuv.find('seg')
            if seg_elem is None:
                continue
            text = clean_text(''.join(seg_elem.itertext()))
            if normalize:
                lang = normalize_language(lang)
            segs[lang] = text
        if len(segs) >= 2:
            langs = list(segs.keys())
            l1, l2 = langs[0], langs[1]
            data.append({
                'source_lang': l1,
                'target_lang': l2,
                'source_text': segs[l1],
                'target_text': segs[l2]
            })
    return pd.DataFrame(data)

def parse_xliff(file_obj, normalize=True):
    content = file_obj.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    content = content.lstrip(b'\x00')
    parser = etree.XMLParser(strip_cdata=False, recover=True, encoding='utf-8')
    root = etree.fromstring(content, parser)
    default_ns = root.nsmap.get(None)
    ns = {'ns': default_ns} if default_ns else {}
    trans_units = root.xpath('//ns:trans-unit', namespaces=ns) if default_ns else root.xpath('//trans-unit')
    data = []
    for tu in trans_units:
        if tu.get('translate') == 'no':
            continue
        src_elem = tu.find('ns:source', ns) if default_ns else tu.find('source')
        tgt_elem = tu.find('ns:target', ns) if default_ns else tu.find('target')
        if src_elem is None:
            continue
        src_text = clean_text(''.join(src_elem.itertext()))
        tgt_text = clean_text(''.join(tgt_elem.itertext())) if tgt_elem is not None else ""
        if not src_text:
            continue
    
        data.append({
            'source_lang': "unknown",
            'target_lang': "unknown",
            'source_text': src_text,
            'target_text': tgt_text,
            'id': tu.get('id', ''),
            'state': tgt_elem.get('state', '') if tgt_elem is not None else ''
        })
    return pd.DataFrame(data)

# ==============================
# 3. POST-PROCESSING WITH QUALITY FILTER
# ==============================

def is_good_alignment(src_sent: str, tgt_sent: str, src_lang: str, tgt_lang: str) -> bool:
    src = src_sent.strip()
    tgt = tgt_sent.strip()
    if not src and not tgt:
        return False
    if (not src and len(tgt) > 10) or (not tgt and len(src) > 10):
        return False
    src_len, tgt_len = len(src), len(tgt)
    if src_len > 0 and tgt_len > 0:
        ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
        if ratio > 5:
            return False
    if tgt_lang.startswith('zh') and tgt:
        if not re.search(r'[\u4e00-\u9fff]', tgt):
            return False
    if src_lang.startswith('fr') and src:
        if not re.search(r'[a-zA-Z]', src):
            return False
    return True

def split_sentences_spacy(text: str, lang: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    lang = lang.lower()
    
    sentences = re.split(r'(?<=[.!?。！？；])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def postprocess_with_quality_filter(df, src_lang_col='source_lang', tgt_lang_col='target_lang',
                                   src_text_col='source_text', tgt_text_col='target_text',
                                   min_words_to_split=15, drop_bad_alignments=True):
    if df.empty:
        return df.copy()
    new_rows = []
    dropped = 0
    for _, row in df.iterrows():
        src_text = str(row.get(src_text_col, ""))
        tgt_text = str(row.get(tgt_text_col, ""))
        src_lang = str(row.get(src_lang_col, "unknown"))
        tgt_lang = str(row.get(tgt_lang_col, "unknown"))
        if len(src_text.split()) <= min_words_to_split:
            new_rows.append(row.to_dict())
            continue
        src_sents = split_sentences_spacy(src_text, src_lang)
        tgt_sents = split_sentences_spacy(tgt_text, tgt_lang)
        if len(src_sents) <= 1 and len(tgt_sents) <= 1:
            new_rows.append(row.to_dict())
            continue
        max_len = max(len(src_sents), len(tgt_sents))
        for i in range(max_len):
            src = src_sents[i] if i < len(src_sents) else ""
            tgt = tgt_sents[i] if i < len(tgt_sents) else ""
            if drop_bad_alignments and not is_good_alignment(src, tgt, src_lang, tgt_lang):
                dropped += 1
                continue
            new_row = row.to_dict()
            new_row[src_text_col] = src
            new_row[tgt_text_col] = tgt
            new_rows.append(new_row)
    result = pd.DataFrame(new_rows).reset_index(drop=True)
    if dropped > 0:
        st.warning(f"⚠️ Dropped {dropped} low-quality sentence pairs.")
    return result

# ==============================
# 4. HUGGING FACE JSON EXPORT 
# ==============================

def df_to_hf_json(df):
    records = []
    for _, row in df.iterrows():
        src_lang = str(row.get('source_lang', 'unknown')).strip()
        tgt_lang = str(row.get('target_lang', 'unknown')).strip()
        src_text = str(row.get('source_text', '')).strip()
        tgt_text = str(row.get('target_text', '')).strip()

        if not src_text and not tgt_text:
            continue

        trans_dict = {}

        # Only use real language code if it's NOT "unknown"
        if src_lang != "unknown" and src_text:
            trans_dict[src_lang] = src_text
        if tgt_lang != "unknown" and tgt_text:
            trans_dict[tgt_lang] = tgt_text

        # If both are unknown, fall back to generic keys
        if not trans_dict:
            trans_dict = {"src": src_text, "tgt": tgt_text}
        elif len(trans_dict) == 1:
            # One known, one unknown → still use generic for unknown side
            if src_lang == "unknown" and src_text:
                trans_dict["src"] = src_text
            if tgt_lang == "unknown" and tgt_text:
                trans_dict["tgt"] = tgt_text

        records.append({"translation": trans_dict})
    
    return "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)

# ==============================
# 5. STREAMLIT APP
# ==============================

st.set_page_config(page_title="Translation File Converter", layout="wide")
st.title("🔤 Translation & NLP File Converter")

# === Language Support Notice ===
with st.expander("ℹ️ Supported Languages & Features", expanded=False):
    st.markdown("""
    This app works with **any language**, but **advanced features are optimized for English, French, and Chinese**:
    
    - 🌐 **Language normalization**:  
      `en-US`/`en-GB` → `en` | `fr-FR` → `fr`  
      `zh-CN`/`zh-SG` → `zhs` (Simplified) | `zh-TW`/`zh-HK` → `zht` (Traditional)
    - 🔄 **Chinese character conversion**:  
      Optional Simplified ↔ Traditional (only for `zhs`/`zht` pairs)
    - ✂️ **Sentence splitting & quality filtering**:  
      Uses text heuristics for **en/fr/zh**
    - 🧹 **Deduplication & Hugging Face export**:  
      Available for all languages
    
    > ℹ️ Files in other languages will convert correctly — but without language-specific enhancements.
    """)

uploaded_files = st.file_uploader(
    "📤 Upload one or more files", 
    type=["tmx", "sdltm", "xliff", "sdlxliff", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    input_format = st.selectbox("🔤 Input format", ["Auto-detect", "TMX", "SDLTM", "XLIFF/SDLXLIFF", "CSV", "Excel"])
    purpose = st.radio("🎯 Conversion purpose", ["Translation / Localization", "Model Training / Fine-tuning"])
    
    normalize_lang = st.checkbox("🔤 Normalize language codes", value=True)
    convert_zh = False
    if normalize_lang:
        if any('zh' in f.name.lower() or f.type.endswith('sdltm') for f in uploaded_files):
            convert_zh = st.checkbox("🔄 Convert Chinese characters (Simplified ↔ Traditional)")

    dedup = st.checkbox("🗑️ Remove duplicate (source, target) pairs", value=True)
    sentence_split = st.checkbox("✂️ Split long segments into sentences (with quality filtering)", value=False)
    
    all_dfs = []
    progress = st.progress(0)
    for i, file in enumerate(uploaded_files):
        file_ext = file.name.split('.')[-1].lower()
        fmt = input_format
        if fmt == "Auto-detect":
            fmt = {"tmx": "TMX", "sdltm": "SDLTM", "xliff": "XLIFF/SDLXLIFF", "sdlxliff": "XLIFF/SDLXLIFF", "csv": "CSV", "xlsx": "Excel"}.get(file_ext, "Unknown")
        
        try:
            if fmt == "SDLTM":
                df = parse_sdltm(file, normalize=normalize_lang, convert_zh_chars=convert_zh)
            elif fmt == "TMX":
                df = parse_tmx(file, normalize=normalize_lang)
            elif fmt in ["XLIFF/SDLXLIFF"]:
                df = parse_xliff(file, normalize=normalize_lang)
            elif fmt == "CSV":
                df = pd.read_csv(file)
            elif fmt == "Excel":
                df = pd.read_excel(file)
            else:
                st.error(f"Unsupported format for {file.name}")
                continue
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            st.error(f"❌ Failed to process {file.name}: {e}")
        progress.progress((i + 1) / len(uploaded_files))
    
    if not all_dfs:
        st.error("❌ No valid data extracted.")
    else:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        if dedup:
            df_combined = df_combined.drop_duplicates(subset=['source_text', 'target_text'], keep='first')
        if sentence_split:
            df_combined = postprocess_with_quality_filter(df_combined)
        
        st.success(f"✅ Processed {len(df_combined)} segments.")
        st.dataframe(df_combined.head(10))

        all_cols = list(df_combined.columns)
        default_cols = ['source_text', 'target_text'] if purpose == "Model Training / Fine-tuning" else all_cols
        selected_cols = st.multiselect("📊 Select output columns", all_cols, default=[c for c in default_cols if c in all_cols])
        df_out = df_combined[selected_cols] if selected_cols else df_combined

        output_format = st.selectbox("📥 Output format", ["CSV", "Excel", "TMX", "Hugging Face JSON"])

        buffer = BytesIO()
        if output_format == "CSV":
            df_out.to_csv(buffer, index=False, encoding='utf-8-sig')
            mime, fname = "text/csv", "output.csv"
        elif output_format == "Excel":
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as w:
                df_out.to_excel(w, index=False)
            mime, fname = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "output.xlsx"
        elif output_format == "TMX":
            required = {'source_lang','target_lang','source_text','target_text'}
            if not required.issubset(df_out.columns):
                st.error(f"❌ TMX requires columns: {required}")
            else:
                tmx_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n<tmx version="1.4"><header></header><body>'
                for _, r in df_out.iterrows():
                    tmx_bytes += f'<tu><tuv xml:lang="{r["source_lang"]}"><seg>{r["source_text"]}</seg></tuv><tuv xml:lang="{r["target_lang"]}"><seg>{r["target_text"]}</seg></tuv></tu>'.encode('utf-8')
                tmx_bytes += b'</body></tmx>'
                buffer.write(tmx_bytes)
                mime, fname = "application/xml", "output.tmx"
        elif output_format == "Hugging Face JSON":
            json_str = df_to_hf_json(df_out)
            buffer.write(json_str.encode('utf-8'))
            mime, fname = "application/json", "output.json"

        buffer.seek(0)
        st.download_button("⬇️ Download", data=buffer, file_name=fname, mime=mime)