import streamlit as st

def show():
    st.title("Entraînement du Modèle")
    st.write("Cliquez sur le bouton ci-dessous pour lancer l'entraînement du modèle.")

    if st.button("Lancer l'entraînement"):
        st.write("Entraînement en cours...")
        # Dummy training logic
        for i in range(1, 6):
            st.write(f"Étape {i}/5 complétée")
        st.write("Entraînement terminé avec succès.")

if __name__ == "__main__":
    show()
