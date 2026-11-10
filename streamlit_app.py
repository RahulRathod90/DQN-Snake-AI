
import streamlit as st

st.set_page_config(
    page_title="DQN Snake AI",
    page_icon="🐍",
    layout="wide"
)

st.title("🐍 DQN Snake AI")
st.subheader("Deep Reinforcement Learning Platform")

st.markdown("""
Welcome to DQN Snake AI!

This project explores intelligent game-playing
using Deep Reinforcement Learning.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric("Algorithm", "Deep Q-Network")
    st.metric("State Representation", "11-D / 24-D")

with col2:
    st.metric("Pathfinding", "A* Search")
    st.metric("Framework", "PyTorch")

st.success("Streamlit application is running successfully!")
