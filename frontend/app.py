import streamlit as st
import requests

st.title("arXiv Classifier")
title = st.text_input("Title*")
abstract = st.text_area("Abstract (optional)")

if st.button("Predict"):
    if not title:
        st.error("Please enter a title!")
    else:
        try:
            response = requests.post(
                "http://api:8000/predict",
                json={"title": title, "abstract": abstract or ""}
            ).json()

            st.subheader("Results:")
            for pred in response["predictions"]:
                st.progress(pred["probability"])
                st.write(f"{pred['category']}: {pred['probability']:.1%}")

        except Exception as e:
            st.error(f"Error: {str(e)}")