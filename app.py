import streamlit as st
import os
import faiss
import pickle
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ==========================================
# ⚡ 設定エリア
# ==========================================
HF_TOKEN = "HF_TOKEN"  # ← あなたのトークンに書き換える！
STORE_DIR = "vector_store"
MODEL_NAME = "intfloat/multilingual-e5-small"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 鎌倉市の地区リスト（ユーザーに選ばせる用）
AREA_LIST = [
    "今泉", "大船", "岩瀬", "小袋谷", "高野", 
    "山ノ内", "台", "小坂", "津", "腰越", 
    "七里ガ浜", "極楽寺", "長谷", "坂ノ下", 
    "由比ガ浜", "材木座", "鎌倉山", "笛田",
    "手広", "常盤", "梶原", "寺分", "上町屋"
]

# ページ設定
st.set_page_config(page_title="鎌倉市ゴミ出しAI", page_icon="🗑️")

# --- 関数定義 ---
@st.cache_resource
def load_models():
    """モデルとデータを読み込む（キャッシュして高速化）"""
    if not os.path.exists(os.path.join(STORE_DIR, "index.faiss")):
        return None, None, None
        
    with open(os.path.join(STORE_DIR, "doc_map.pkl"), "rb") as f:
        doc_map = pickle.load(f)
    index = faiss.read_index(os.path.join(STORE_DIR, "index.faiss"))
    encoder = SentenceTransformer(MODEL_NAME)
    return index, doc_map, encoder

def get_date_info():
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    now = datetime.datetime.now()
    today_str = f"{now.strftime('%Y年%m月%d日')}（{weekdays[now.weekday()]}曜日）"
    tomorrow = now + datetime.timedelta(days=1)
    tomorrow_str = f"{tomorrow.strftime('%Y年%m月%d日')}（{weekdays[tomorrow.weekday()]}曜日）"
    return today_str, tomorrow_str

def generate_response(user_input, area, index, doc_map, encoder):
    # 検索
    search_query = f"{area} {user_input}"
    query_vector = encoder.encode([f"query: {search_query}"], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_vector), 3)
    
    results = []
    for idx in indices[0]:
        if idx < len(doc_map):
            results.append(doc_map[idx])
    
    # 回答生成
    client = InferenceClient(api_key=HF_TOKEN)
    today, tomorrow = get_date_info()
    context_str = "\n".join(results)
    
    system_instruction = (
        "あなたは鎌倉市のゴミ出し案内係です。\n"
        f"【ユーザーの住む地区: {area}】\n"
        f"【今日: {today}】\n"
        f"【明日: {tomorrow}】\n"
        "ユーザーの質問に対し、上記の【ユーザーの住む地区】のルールに基づいて回答してください。"
        "他の地区の情報は無視してください。"
        "情報がない場合は「資料にないのでわかりません」と答えてください。"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"【参考情報】\n{context_str}\n\n【質問】\n{user_input}"}
    ]
    
    try:
        response = client.chat_completion(
            model=CHAT_MODEL, messages=messages, max_tokens=500, temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラー: {e}"

# --- アプリの画面 ---
st.title("🗑️ 鎌倉市ゴミ出しAIチャット")

# サイドバー（設定エリア）
st.sidebar.header("⚙️ ユーザー設定")
selected_area = st.sidebar.selectbox("あなたのお住まいの地区を選んでください", AREA_LIST)
st.sidebar.write(f"現在の設定: **{selected_area}** 地区")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# モデル読み込み
index, doc_map, encoder = load_models()

if index is None:
    st.error("エラー: データベースが見つかりません。先に build_index.py を実行してください。")
else:
    # 過去のチャットを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 入力フォーム
    if prompt := st.chat_input("質問を入力してください（例：明日のゴミは？）"):
        # ユーザーのメッセージを表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIの回答を生成
        with st.chat_message("assistant"):
            with st.spinner("AIが考え中..."):
                response = generate_response(prompt, selected_area, index, doc_map, encoder)
                st.markdown(response)
        
        # 履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response})