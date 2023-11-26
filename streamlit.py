from io import BytesIO
import streamlit as st

from helpers import extract_text_from_zip, extract_messages_from_chats, generate_wordcloud



def main():
    st.title("ChatGPT Conversation Word Cloud App")

    # File Upload
    uploaded_file = st.file_uploader("Upload a ZIP file", type=["zip"])

    if uploaded_file is not None:
        # Extract text from files inside the zip
        zip_data = BytesIO(uploaded_file.getvalue())
        chats_df = extract_text_from_zip(zip_data)
        messages_df = extract_messages_from_chats(chats_df)

        user_texts = messages_df[lambda df: df['message_sender'] == 'user']['message_content']

        # Display some sample texts
        st.subheader("Sample Conversation Texts:")
        for i, text in enumerate(user_texts[:5]):
            st.write(f"Text {i + 1}:\n{text}")

        # Generate and display word cloud
        st.subheader("Word Cloud:")

        wordcloud, fig = generate_wordcloud(user_texts)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
