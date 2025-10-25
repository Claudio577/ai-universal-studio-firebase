import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
from PIL import Image

from firebase_utils import init_firebase, baixar_arquivo
from multimodal_model import embed_texto, embed_imagem, embed_audio, combinar_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================
# ⚙️ Configuração inicial
# ==========================
st.set_page_config(page_title="AI Universal Studio (Firebase)", page_icon="🧠", layout="wide")
st.title("🧠 AI Universal Studio – Firebase Edition")
st.write("Sistema de IA que aprende automaticamente com **textos**, **imagens** e **áudios** armazenados no Firebase e realiza previsões multimodais.")

db, bucket = init_firebase()

# ==========================
# 🔁 Sessão compartilhada
# ==========================
for var, default in {
    "conceitos": [],
    "modelo": None,
    "X": None,
    "y": None
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ==========================
# 🧭 Abas
# ==========================
aba = st.tabs([
    "🧩 Etapa 1 - Palavras-chave e categorias",
    "⚙️ Etapa 2 - Treinar modelo",
    "🔮 Etapa 3 - Fazer previsão"
])

# ======================================================
# 1️⃣ ETAPA 1 – Definir palavras-chave
# ======================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Definir conceitos de aprendizado")
    st.write("Crie palavras-chave e associe a uma categoria. Esses dados buscarão automaticamente conteúdos no Firebase.")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        palavras = col1.text_input(f"📝 Palavra-chave {i+1}:", key=f"palavra_{i}")
        categoria = col2.selectbox(f"🎯 Categoria {i+1}:", ["Baixo", "Moderado", "Alto"], key=f"cat_{i}")
        if palavras:
            entradas.append({"palavra": palavras.lower(), "categoria": categoria})

    if entradas and st.button("💾 Salvar conceitos no Firebase"):
        for e in entradas:
            db.collection("conceitos").add(e)
        st.success("✅ Conceitos salvos no Firebase!")
        st.session_state.conceitos = entradas
        st.dataframe(pd.DataFrame(entradas))

# ======================================================
# 2️⃣ ETAPA 2 – Treinamento multimodal
# ======================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com dados do Firebase")
    st.write("O app irá buscar automaticamente **imagens**, **textos** e **áudios** no Firebase relacionados às palavras-chave definidas.")

    if st.button("🚀 Treinar modelo agora"):
        conceitos = list(db.collection("conceitos").stream())
        if not conceitos:
            st.warning("⚠️ Nenhum conceito encontrado. Vá para a Etapa 1 primeiro.")
        else:
            conceitos_dict = [c.to_dict() for c in conceitos]
            st.info("🔍 Buscando arquivos no Firebase...")

            arquivos = list(db.collection("arquivos").stream())
            arquivos_dict = [a.to_dict() for a in arquivos]

            X, y = [], []

            for conceito in conceitos_dict:
                palavra = conceito["palavra"]
                categoria = conceito["categoria"]
                st.write(f"🧩 Processando conceito **{palavra} ({categoria})**")

                relacionados = [a for a in arquivos_dict if palavra in [t.lower() for t in a.get("tags", [])]]

                for item in relacionados:
                    tipo = item["tipo"]
                    try:
                        if tipo == "texto":
                            feat = embed_texto(item["descricao"])
                        elif tipo == "imagem":
                            path = baixar_arquivo(item["url"])
                            feat = embed_imagem(path)
                        elif tipo == "audio":
                            path = baixar_arquivo(item["url"])
                            feat = embed_audio(path)
                        else:
                            continue

                        X.append(feat)
                        y.append(categoria)

                    except Exception as e:
                        st.error(f"Erro ao processar {item['url'][:40]}...: {e}")

            if len(X) < 3:
                st.warning("⚠️ Poucos dados para treinar. Adicione mais exemplos no Firebase.")
            else:
                X = np.vstack(X)
                modelo = RandomForestClassifier(n_estimators=200)
                modelo.fit(X, y)
                st.session_state.modelo = modelo
                st.session_state.X, st.session_state.y = X, y
                joblib.dump(modelo, "modelo_treinado.pkl")

                st.success("✅ Modelo treinado e salvo com sucesso!")
                st.write(f"**Amostras usadas:** {len(X)}")

# ======================================================
# 3️⃣ ETAPA 3 – Fazer previsão
# ======================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão multimodal")
    st.write("Envie imagem, texto e/ou áudio para o modelo prever a categoria mais provável.")

    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional):", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Texto descritivo (opcional):")
    uploaded_audio = st.file_uploader("🎤 Envie um áudio (opcional):", type=["wav", "mp3", "m4a"])

    features = []

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            features.append(embed_imagem(tmp.name))

    if texto_input:
        features.append(embed_texto(texto_input))

    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            features.append(embed_audio(tmp.name))

    if st.button("🔍 Fazer previsão"):
        if not st.session_state.modelo:
            st.warning("⚠️ Nenhum modelo treinado. Vá para a Etapa 2.")
        elif not features:
            st.warning("⚠️ Adicione pelo menos uma entrada (imagem, texto ou áudio).")
        else:
            entrada = combinar_features(features).reshape(1, -1)
            pred = st.session_state.modelo.predict(entrada)[0]
            proba = st.session_state.modelo.predict_proba(entrada)[0]
            categorias = st.session_state.modelo.classes_
            df_proba = pd.DataFrame({"Categoria": categorias, "Probabilidade": proba}).sort_values("Probabilidade", ascending=False)

            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}.get(pred, "blue")
            st.markdown(f"<h3>🧠 Previsão: <span style='color:{cor}'>{pred}</span></h3>", unsafe_allow_html=True)
            st.bar_chart(df_proba.set_index("Categoria"))
