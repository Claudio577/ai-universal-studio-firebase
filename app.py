# ======================================================
# 🧠 AI Universal Studio – Supabase Edition
# ======================================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tempfile
from PIL import Image
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier

# Suas funções de embedding
from multimodal_model import embed_texto, embed_imagem, embed_audio, combinar_features

# ======================================================
# ⚙️ CONFIGURAÇÃO INICIAL
# ======================================================
st.set_page_config(page_title="AI Universal Studio (Supabase)", page_icon="🧠", layout="wide")
st.title("🧠 AI Universal Studio – Supabase Edition")
st.write("Sistema de IA que aprende automaticamente com **textos**, **imagens** e **áudios** armazenados no **Supabase** e realiza previsões multimodais.")

# ======================================================
# 🔑 CONEXÃO COM SUPABASE
# ======================================================
url = "https://rkwevdkaklsbawymtuiv.supabase.co"  # 🔗 Seu Project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJrd2V2ZGtha2xzYmF3eW10dWl2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEzOTA1NDQsImV4cCI6MjA3Njk2NjU0NH0.BDaD-gOJx8QAgx-mzxdBPEPqvtl37diBAUFNa5-2XAQ"
supabase: Client = create_client(url, key)

# ======================================================
# 🔁 VARIÁVEIS DE SESSÃO
# ======================================================
for var, default in {
    "conceitos": [],
    "modelo": None,
    "X": None,
    "y": None
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ======================================================
# 🧭 ABAS
# ======================================================
aba = st.tabs([
    "🧩 Etapa 1 - Palavras-chave e categorias",
    "⚙️ Etapa 2 - Treinar modelo",
    "🔮 Etapa 3 - Fazer previsão"
])

# ======================================================
# 1️⃣ ETAPA 1 – DEFINIR CONCEITOS
# ======================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Definir conceitos de aprendizado")
    st.write("Crie palavras-chave e associe a uma categoria. Esses dados serão salvos no **Supabase**.")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        palavra = col1.text_input(f"📝 Palavra-chave {i+1}:", key=f"palavra_{i}")
        categoria = col2.selectbox(f"🎯 Categoria {i+1}:", ["Baixo", "Moderado", "Alto"], key=f"cat_{i}")
        if palavra:
            entradas.append({"palavra": palavra.lower(), "categoria": categoria})

    if entradas and st.button("💾 Salvar conceitos no Supabase"):
        try:
            for e in entradas:
                supabase.table("conceitos").insert(e).execute()
            st.success("✅ Conceitos salvos no Supabase!")
            st.session_state.conceitos = entradas
            st.dataframe(pd.DataFrame(entradas))
        except Exception as e:
            st.error(f"❌ Erro ao salvar conceitos: {e}")

# ======================================================
# 2️⃣ ETAPA 2 – TREINAR MODELO
# ======================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com dados do Supabase")
    st.write("O app busca **imagens**, **textos** e **áudios** no Supabase relacionados às palavras-chave definidas.")

    if st.button("🚀 Treinar modelo agora"):
        try:
            conceitos_resp = supabase.table("conceitos").select("*").execute()
            conceitos = conceitos_resp.data

            if not conceitos:
                st.warning("⚠️ Nenhum conceito encontrado. Vá para a Etapa 1 primeiro.")
            else:
                st.info("🔍 Buscando arquivos no Supabase...")

                arquivos_resp = supabase.table("arquivos").select("*").execute()
                arquivos = arquivos_resp.data

                X, y = [], []

                for conceito in conceitos:
                    palavra = conceito["palavra"]
                    categoria = conceito["categoria"]
                    st.write(f"🧩 Processando conceito **{palavra} ({categoria})**")

                    relacionados = [a for a in arquivos if palavra in [t.lower() for t in a.get("tags", [])]]

                    for item in relacionados:
                        tipo = item.get("tipo")
                        try:
                            if tipo == "texto":
                                feat = embed_texto(item.get("descricao", ""))
                            elif tipo == "imagem":
                                feat = embed_imagem(item["url"])
                            elif tipo == "audio":
                                feat = embed_audio(item["url"])
                            else:
                                continue

                            X.append(feat)
                            y.append(categoria)

                        except Exception as e:
                            st.error(f"Erro ao processar {item.get('url', '')[:40]}...: {e}")

                if len(X) < 3:
                    st.warning("⚠️ Poucos dados para treinar. Adicione mais exemplos no Supabase.")
                else:
                    X = np.vstack(X)
                    modelo = RandomForestClassifier(n_estimators=200)
                    modelo.fit(X, y)
                    st.session_state.modelo = modelo
                    st.session_state.X, st.session_state.y = X, y
                    joblib.dump(modelo, "modelo_treinado.pkl")

                    st.success("✅ Modelo treinado e salvo com sucesso!")
                    st.write(f"**Amostras usadas:** {len(X)}")

        except Exception as e:
            st.error(f"❌ Erro no treinamento: {e}")

# ======================================================
# 3️⃣ ETAPA 3 – FAZER PREVISÃO
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

