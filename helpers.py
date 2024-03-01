import datetime
import re
from zipfile import ZipFile
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

stopwords = set(STOPWORDS)

def clean_string(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    
    _remove_specialcharacters = re.compile(r"[^a-zA-Z0-9]")
    text = _remove_specialcharacters.sub(" ", text)

    _combine_whitespace = re.compile(r"\s+")
    text = _combine_whitespace.sub(" ", text).strip()
    

    return text

def extract_name_from_zip(zip_file):
    with ZipFile(zip_file, 'r') as z:
        name = re.search(r'([^/-]+)-\d{4}-\d{2}-\d{2}\.zip', z.filename)
        print(name, z.filename)
    if name:
        return name.group(1)
    else:
        return None

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
             'message_sender': [],
             'message_time': []
            }
             
    for i, row in enumerate(chats_df.sort_values('update_time').iterrows()):
        for response in row[1]['mapping'].values():
            if response['message']:
                if response['message']['content']['content_type'] == 'text':
                    all_messages['conversation_order_id'].append(i)
                    all_messages['message_content'].append(clean_string(' '.join(response['message']['content']['parts'])))
                    all_messages['message_sender'].append(response['message']['author']['role'])

                    if response['message']['create_time']:
                        timestamp = datetime.datetime.fromtimestamp(response['message']['create_time'])
                    else:
                        timestamp = None
                    all_messages['message_time'].append(timestamp)

    messages_df = pd.DataFrame.from_dict(all_messages)
    return messages_df

# Function to generate and display a word cloud
def generate_wordcloud(texts, title=None, bigrams=False):
    # Combine all texts into a single string
    if type(texts) != str:
        combined_text = ' '.join(texts)
    else:
        combined_text = texts
    
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

    return wordcloud, fig, ax
    


def get_time_of_day(timestamp):
    hour = timestamp.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'day'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'

def get_day_of_week(timestamp):
    return timestamp.strftime('%A')  # %A gives the full name of the day

def is_weekend(timestamp):
    return timestamp.day_name() in ['Saturday', 'Sunday']

def add_datetime_columns(df, datetime_columnname):
    # Apply the functions to create new columns
    df['time_of_day'] = df[datetime_columnname].apply(get_time_of_day)
    df['day_of_week'] = df[datetime_columnname].apply(get_day_of_week)
    df['hour_of_day'] = df[datetime_columnname].apply(lambda timestamp: timestamp.hour)
    df['date'] = df[datetime_columnname].apply(lambda timestamp: timestamp.strftime('%d-%m-%Y'))
    df['yearmonth'] = df[datetime_columnname].apply(lambda timestamp: timestamp.strftime('%B %Y'))
    df['yearmonth_numeric'] = df[datetime_columnname].apply(lambda timestamp: timestamp.strftime('%Y-%m'))
    df['weekend'] = df[datetime_columnname].apply(is_weekend)

    return df


def get_vectorizer_input(df, content_columnname='message_content', groupby_columnname='conversation_order_id'):
    # intermediate_df = df.copy()
    # intermediate_df[content_columnname] = intermediate_df[content_columnname].apply(lambda x: x + ' ')
    # vector_tuples = intermediate_df.groupby(groupby_columnname)[content_columnname].sum()
    # vector_input = [seq.strip() for seq in vector_tuples.tolist()]
    vector_input = df[content_columnname].tolist()
    vector_index = df.index

    return vector_input, vector_index

def get_tfidf_counter(vector_input, vector_index):

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_vector = tfidf.fit_transform(vector_input)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=vector_index, columns=tfidf.get_feature_names_out())

    return tfidf_df

def get_counter(vector_input, vector_index):

    counter = CountVectorizer(stop_words='english')
    count_vector = counter.fit_transform(vector_input)
    count_df = pd.DataFrame(count_vector.toarray(), index=vector_index, columns=counter.get_feature_names_out())

    return count_df

def find_topic_types_transfomer(user_messages_df):

    vector_input, vector_index = get_vectorizer_input(user_messages_df, 
        content_columnname='message_content', groupby_columnname='conversation_order_id')
    print(f'{len(vector_input)} vector input messages.')
    tfidf_df = get_tfidf_counter(vector_input, vector_index)
    count_df = get_counter(vector_input, vector_index)

    top_tfidf = tfidf_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).tolist()
    top_count = count_df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).tolist()
    tfidf_count_topics = [' '.join(list(set(tf + ct))) for tf, ct in zip(top_tfidf, top_count)]

    print('Loading transformer...')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    types = ['this is about data science or software engineering)', 
              'this is about health, personal matters or personal finance matters',  # (incl. physical and mental)
              'this is about world knowledge (incl. history, politics, physics) or about culture (incl. movies, games, music)', 
    ]

    short_types = ['work', 'personal', 'world']#, 'culture', 'personal', 'money']
    encoded_types = model.encode([topic for topic in types])
    batch_size = 64
    encoded_topics = []
    for i in range(len(tfidf_count_topics) // batch_size + 1):
        encoded_topics.append(model.encode(tfidf_count_topics
                                        [i*batch_size:(i+1)*batch_size]
                                       )
                               )
    encoded_topics = np.concatenate(encoded_topics, axis=0)
    topic_similarity = cosine_similarity(encoded_topics, encoded_types)

    topic_types = [(short_types[arg]) for i, arg in enumerate(np.argmax(topic_similarity, axis=1))]
    print(f'Processed {len(topic_types)} messages.')
    return topic_types

def find_top_words_used(user_messages_df):

    wordcounter = CountVectorizer(stop_words='english')
    wordcount_vector = wordcounter.fit_transform([' '.join(user_messages_df['message_content'].tolist())])
    wordcount_df = pd.DataFrame(wordcount_vector.toarray(), index=['count'], columns=wordcounter.get_feature_names_out())
    top_words_df = wordcount_df.transpose().sort_values(by='count', ascending=False).head(25).reset_index()

    return top_words_df
