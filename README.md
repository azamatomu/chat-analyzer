# ChatGPT Usage Analysis

This project analyzes the usage patterns of the ChatGPT model based on user interactions. It processes chat data from conversations with ChatGPT and provides insights such as message frequency, commonly used words, and topic classification.

## Overview

The project consists of several Python scripts:

- `helpers.py`: Contains utility functions for data cleaning, text analysis, and visualization.
- `question_streamlit.py`: Implements a Streamlit app for interactive analysis and visualization of ChatGPT conversations.
- `requirements.txt`: Lists the dependencies required to run the project.

## Features

- **Message Frequency Analysis**: Visualize the number of messages sent per month to understand usage patterns over time.
- **Top Words Used**: Identify the most frequently used words in conversations with ChatGPT.
- **Topic Classification**: Classify messages into different topics (e.g., work-related, personal, world knowledge) using NLP techniques.
- **Difficulty Analysis**: Analyze the difficulty level of work-related conversations based on GPT responses.

## Usage

1. **Upload Chat Data**: Upload a zip file containing ChatGPT conversation data.
2. **Questionnaire**: Answer questions about your ChatGPT usage habits, such as the frequency of usage and commonly used words.
3. **Analysis Results**: View visualizations and insights based on your ChatGPT interactions, including message frequency, top words used, and topic classification.

## Requirements

Ensure you have the following dependencies installed:

- pandas
- matplotlib
- numpy
- wordcloud
- streamlit

Install the dependencies using `pip install -r requirements.txt`.

## Contributors

- [Azamat Omuraliev/azamatomu]

## License

This project is licensed under the [MIT License](LICENSE).
