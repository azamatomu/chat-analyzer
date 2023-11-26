import re
from zipfile import ZipFile
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def clean_string(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    
    _remove_specialcharacters = re.compile(r"[^a-zA-Z0-9]")
    text = _remove_specialcharacters.sub(" ", text)

    _combine_whitespace = re.compile(r"\s+")
    text = _combine_whitespace.sub(" ", text).strip()
    

    return text
def extract_text_from_zip(zip_file):
    with ZipFile(zip_file, 'r') as z:
        for file_info in z.infolist():
            print(file_info)
            if file_info.filename == 'conversations.json':
                with z.open(file_info.filename) as file:
                    chats_df = pd.read_json(file)
    return chats_df

def extract_messages_from_chats(chats_df):
    all_messages = {'conversation_order_id': [],
             'message_content': [],
             'message_sender': []
            }
             
    for i, row in enumerate(chats_df.sort_values('update_time').iterrows()):
        for response in row[1]['mapping'].values():
            if response['message']:
                if response['message']['content']['content_type'] == 'text':
                    all_messages['conversation_order_id'].append(i)
                    all_messages['message_content'].append(clean_string(' '.join(response['message']['content']['parts'])))
                    all_messages['message_sender'].append(response['message']['author']['role'])

    messages_df = pd.DataFrame.from_dict(all_messages)
    return messages_df

# Function to generate and display a word cloud
def generate_wordcloud(texts, title=None, bigrams=False):
    # Combine all texts into a single string
    combined_text = ' '.join(texts)
    
    # Generate WordCloud
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40,
        scale=3,
        collocations=bigrams,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(combined_text)    

    # Display the WordCloud using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    return wordcloud, fig
    
