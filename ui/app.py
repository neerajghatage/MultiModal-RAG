"""
Streamlit demo UI for the RAG system.

Run:
    streamlit run app.py
"""

import os
import re
import streamlit as st
import requests
import time

# ── Config ────────────────────────────────────────────────────
API_URL = os.getenv("RAG_API_URL", "http://localhost:8002")
AZURE_AD_TENANT_ID = os.getenv("AZURE_AD_TENANT_ID", "")
AZURE_AD_CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID", "")
AZURE_AD_ALLOWED_DOMAIN = os.getenv("AZURE_AD_ALLOWED_DOMAIN", "<your-domain.com>")
AZURE_AD_ALLOWED_GROUP_ID = os.getenv("AZURE_AD_ALLOWED_GROUP_ID", "")  # Object ID of allowed AD group

st.set_page_config(
    page_title="Confidential RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS for responsive images ─────────────────────────
st.markdown("""
<style>
[data-testid="stImage"] img {
    max-width: 480px;
    width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Azure AD Authentication ──────────────────────────────────
def _require_auth():
    """Gate the app behind Azure AD login.

    Access is granted only when ALL of the following pass:
      1. User is authenticated with a Microsoft account
      2. Email domain matches AZURE_AD_ALLOWED_DOMAIN
      3. If AZURE_AD_ALLOWED_GROUP_ID is set, user must be a member of that group
         (group IDs come from the id_token 'groups' claim — requires group claim
          configured in App Registration > Token configuration)

    Returns the user email string, or None if auth is not configured (local dev).
    """
    if not AZURE_AD_CLIENT_ID or not AZURE_AD_TENANT_ID:
        # Auth not configured — skip (local dev)
        return None

    from msal_streamlit_authentication import msal_authentication

    login_token = msal_authentication(
        auth={
            "clientId": AZURE_AD_CLIENT_ID,
            "authority": f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}",
            "redirectUri": "/",
            "postLogoutRedirectUri": "/",
        },
        cache={
            "cacheLocation": "sessionStorage",
            "storeAuthStateInCookie": False,
        },
        login_request={"scopes": ["openid", "profile", "email"]},
        login_button_text="🔐 Login with Microsoft Account",
        logout_button_text="Logout",
        key="msal_auth",
    )

    if login_token is not None:
        # Fresh token — validate and cache in session state
        email = login_token.get("account", {}).get("username", "")

        # 1. Validate email domain
        if AZURE_AD_ALLOWED_DOMAIN and not email.lower().endswith(f"@{AZURE_AD_ALLOWED_DOMAIN.lower()}"):
            st.error(f"⛔ Access denied. Only @{AZURE_AD_ALLOWED_DOMAIN} accounts are allowed.")
            st.session_state.pop("auth_email", None)
            st.stop()

        # 2. Validate group membership (if a group is configured)
        if AZURE_AD_ALLOWED_GROUP_ID:
            id_token_claims = login_token.get("idTokenClaims", {})
            user_groups = id_token_claims.get("groups", [])
            if AZURE_AD_ALLOWED_GROUP_ID not in user_groups:
                st.error(
                    "⛔ Access denied. You are not a member of the required Azure AD group.\n\n"
                    "Contact your administrator to request access."
                )
                st.session_state.pop("auth_email", None)
                st.stop()

        st.session_state["auth_email"] = email
        return email

    elif "auth_email" in st.session_state:
        # Streamlit rerun — MSAL.js is reinitializing from sessionStorage, use cached value
        return st.session_state["auth_email"]

    else:
        st.warning("⛔ Please sign in with your Microsoft account to continue.")
        st.stop()

user_email = _require_auth()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    if user_email:
        st.caption(f"👤 {user_email}")
    st.title("⚙️ Settings")
    top_k = st.slider("Documents to retrieve", 1, 50, 10)
    include_related = st.checkbox("Include related elements", value=True)

    st.divider()
    st.subheader("📊 System Status")
    if st.button("Check Health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"Status: {data['status']}")
                st.json(data["collection"])
            else:
                st.error(f"API returned {r.status_code}")
        except requests.ConnectionError:
            st.error("Cannot connect to API. Is it running?")

    st.divider()
    st.subheader("📁 Ingest Documents")
    ingest_folder = st.text_input("Folder path", value="data/")
    col1, col2 = st.columns(2)
    recreate = col1.checkbox("Recreate collection")
    skip_sum = col2.checkbox("Skip summarization")
    if st.button("🚀 Start Ingestion"):
        with st.spinner("Ingesting documents…"):
            try:
                r = requests.post(
                    f"{API_URL}/ingest",
                    json={
                        "folder": ingest_folder,
                        "recreate": recreate,
                        "skip_summarize": skip_sum,
                    },
                    timeout=600,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ Loaded {data['documents_loaded']} docs, {data['elements_created']} elements")
                    st.json(data["upsert_result"])
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.ConnectionError:
                st.error("Cannot connect to API.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        try:
            requests.delete(f"{API_URL}/history", timeout=5)
            st.session_state.messages = []
            st.success("History cleared")
        except requests.ConnectionError:
            st.error("Cannot connect to API.")

# ── Inline image rendering helper ─────────────────────────────
# Match both {{IMG:id}} and {IMG:id} (LLM sometimes uses single braces)
_IMG_PATTERN = re.compile(r"\{?\{IMG:([^}]+)\}?\}")


def _render_answer_with_images(answer_text: str, images: list):
    """Render answer text with inline images where {{IMG:element_id}} markers appear."""
    image_map = {img["element_id"]: img for img in images}

    parts = _IMG_PATTERN.split(answer_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            element_id = part.strip()
            img = image_map.get(element_id)
            if img:
                img_url = f"{API_URL}/image/{element_id}"
                st.image(img_url, use_container_width=False)
            else:
                st.image(f"{API_URL}/image/{element_id}", use_container_width=False)


# ── Main chat area ────────────────────────────────────────────
st.title("🔍 Confidential RAG Chat")
st.caption("Ask questions about your documents. Powered by LangChain + Qdrant + Azure OpenAI.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("images"):
            _render_answer_with_images(msg["content"], msg["images"])
        else:
            st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "top_k": top_k,
                        "include_related": include_related,
                    },
                    timeout=120,
                )
                if r.status_code == 200:
                    data = r.json()
                    images = data.get("images", [])

                    # Render answer with inline images
                    if images:
                        _render_answer_with_images(data["answer"], images)
                    else:
                        st.markdown(data["answer"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "images": data.get("images", []),
                    })
                else:
                    st.error(f"API error: {r.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Start it with: `python -m uvicorn api:app --port 8002`")
