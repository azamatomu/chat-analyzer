import pandas as pd
import numpy as np
from helpers import clean_string, extract_messages_from_chats, generate_wordcloud, extract_text_from_zip, extract_name_from_zip
from helpers import add_datetime_columns, find_topic_types_transfomer, find_top_words_used

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# grouped_data = pd.DataFrame({
#     'Category': ['Category A', 'Category B', 'Category C', 'Category D', 'Category E'],
#     'Type 1': [30, 60, 20, 40, 10],
#     'Type 2': [40, 20, 50, 30, 60],
#     'Type 3': [10, 30, 40, 20, 50]
# })

message_type = 'Type 1'  # Replace with the actual message type you want to plot
color = 'blue'  # Replace with the desired color


# Placeholder dictionary, replace this with your actual data
chats_analysed_stats = {
    'chat_type_ratio': {'work': 40, 'personal': 30, 'world knowledge': 30},
    'chat_type_daytime': {'work': 'night', 'personal': 'evening', 'world knowledge': 'night'}
}

# Placeholder dictionary for chat types
chat_type_dictionaries = {
    'code': 'work',
    'data': 'work',
    'work': 'work',
    'chat': 'personal',
    'learn': 'world knowledge'
}

response_func = lambda value, condition: st.write(f"#### :white_check_mark: Correct! {value}" if condition else f"#### :x: Incorrect. {value}")
# Step 0
uploaded_file = st.file_uploader("Choose a file")

@st.cache_data
def process_chat_file(uploaded_file):
    print(uploaded_file)
    chats_df = extract_text_from_zip(uploaded_file)
    print('Read ZIP, df shape:', chats_df.shape)

    messages_df = extract_messages_from_chats(chats_df)
    user_messages_df = messages_df[lambda df: df['message_sender'] == 'user']
    user_messages_df = add_datetime_columns(user_messages_df, 'message_time')

    print(f'{messages_df.shape} messages in total.')
    print(f'{user_messages_df.shape} user messages.')

    user_messages_df['type'] = find_topic_types_transfomer(user_messages_df)

    user_messages_over_time_df = user_messages_df.groupby(['yearmonth', 'yearmonth_numeric'])['message_content'].count().to_frame('num_messages').reset_index().sort_values('yearmonth_numeric')

    top_words_df = find_top_words_used(user_messages_df)

    return user_messages_df, user_messages_over_time_df, top_words_df

def process_gpt_answer(uploaded_file):
    name = extract_name_from_zip(uploaded_file)
    if name: 
        gpt_classified_df = pd.read_csv(f'data/{name}_gpt_classification.csv')
        difficulty_pie_input = gpt_classified_df[lambda df: df['TYPE'] == 'work-related'].groupby('DIFFICULTY')['TYPE'].count().to_frame('count').reset_index()
        expert_topics = gpt_classified_df[lambda df: df['TYPE'] == 'work-related'][lambda df: df['DIFFICULTY'].isin(['expert', 'advanced'])][['TOPIC', 'DIFFICULTY']]#['TOPIC'].tolist()
        basic_topics = gpt_classified_df[lambda df: df['TYPE'] == 'work-related'][lambda df: df['DIFFICULTY'] == 'basic'][['TOPIC', 'DIFFICULTY']]#['TOPIC'].tolist()

        return difficulty_pie_input, expert_topics, basic_topics
  


def questionnaire(chats_analysed_stats, user_messages_over_time_df, user_messages_df, top_words_df,
                  difficulty_pie_input, expert_topics, basic_topics):


    # Step 1
    st.write(f"#### How many times do you think you have used ChatGPT in the last year?")
    times_used_guess = st.selectbox("",
                                (int(0.5 * chats_analysed_stats['times_used']),
                                 chats_analysed_stats['times_used'],
                                 int(1.6 * chats_analysed_stats['times_used'])),
                                index=None,
                                placeholder='Select amount...')

    # Step 2
    if times_used_guess:
        value = f"Actual value: {chats_analysed_stats['times_used']}"
        condition = times_used_guess == chats_analysed_stats['times_used']
        response_func(value, condition)
        # df = px.data.gapminder().query("country=='Canada'")
        # fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
        

        fig = px.line(user_messages_over_time_df, 
            x="yearmonth", 
            y="num_messages", 
            title=f'Number of messages sent per month, in total {chats_analysed_stats["times_used"]}')

        st.plotly_chart(fig, use_container_width=True)

        # st.write(":white_check_mark: Correct! {value}" if times_used_guess == chats_analysed['times_used'] else ":x: Incorrect. {value}")

    # Step 3
        st.write(f"#### What do you think is the word you use most often when talking to ChatGPT?")
        word_guess = st.selectbox("",
                           (chats_analysed_stats['top_count_words'][2],
                           chats_analysed_stats['top_count_words'][0],
                           chats_analysed_stats['top_count_words'][4],
                           chats_analysed_stats['top_count_words'][14]),
                           index=None, 
                           placeholder='Select word...')
        


    # Step 4
    if times_used_guess and word_guess:
        value = f"Actual word: {chats_analysed_stats['top_count_words'][0]}"
        condition = word_guess == chats_analysed_stats['top_count_words'][0]
        response_func(value, condition)


        # Calculate proportional width based on the y-values (pop)
        # data_canada = px.data.gapminder().query("country == 'Canada'")
        # normalized_width = data_canada['pop'] / data_canada['pop'].max()

        fig = px.bar(top_words_df, x='index', y='count', title='Words that you use most often')#, width=normalized_width)
        st.plotly_chart(fig, use_container_width=True)

    # Step 5
        chat_type = 'work' #chat_type_dictionaries[chats_analysed_stats['top_count_words'][0]]
        st.write(f"#### You have used the word **:rainbow[{word_guess}]** in the context of :star: *{chat_type}* :star:")

    #     next_question = st.button("Next question")

    # # Step 6
    # if times_used_guess and word_guess and next_question:
        st.write(f"#### When do you think you talk to ChatGPT about :star: *{chat_type}* :star: most often?")
        daytime_guess = st.selectbox("",
                                 ('morning', 'day', 'evening', 'night'),
                                 index=None, 
                                 placeholder='Select time of day...')

    if times_used_guess and word_guess and daytime_guess:
        # Step 7

        # Get values for the current message type
        grouped_data = user_messages_df.groupby(['hour_of_day', 'type']).size().unstack(fill_value=0)
        type_values = grouped_data[chat_type].values
        time_of_day = {
            'morning': grouped_data[chat_type][lambda df: (df.index > 6) & (df.index < 12)].sum(),
            'day': grouped_data[chat_type][lambda df: (df.index >= 12) & (df.index < 18)].sum(),
            'evening': grouped_data[chat_type][lambda df: (df.index >= 18) & (df.index < 23)].sum(),
            'night': grouped_data[chat_type][lambda df: (df.index >= 23) | (df.index <= 6)].sum()
        }
        chats_analysed_stats['chat_type_daytime'][chat_type] = list(time_of_day.keys()) [np.argmin(time_of_day.values())]
        # st.write(time_of_day)
        value = f"Actual answer: {chats_analysed_stats['chat_type_daytime'][chat_type]}"
        condition = daytime_guess == chats_analysed_stats['chat_type_daytime'][chat_type]
        response_func(value, condition)


        categories = [f"hour {t}" for t in list(grouped_data.index)]    



        # Plotly Polar Plot
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=type_values,
            # theta=grouped_data['Category'],
            theta=categories,
            fill='toself',
            mode='lines',
            name=chat_type,
            line=dict(color='orange', width=2),
        ))

        # Layout settings
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    ticks='',
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                )
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    # Step 9
        st.write(f"#### Want to learn more about :star: *WORK* :star: messages?")
        continue_analysis = st.toggle("Start analyZZZZing...")
        
        if continue_analysis:
            st.write(f"#### How advanced were your work-related chats?")

            fig = px.pie(difficulty_pie_input, values='count', names='DIFFICULTY')
            st.plotly_chart(fig, use_container_width=True)

            # expert_analysis = st.button("Show EXPERT topics...", type="primary")
            # basic_analysis = st.button("Show BASIC topics...", type="primary")

            # if expert_analysis:
            st.write(f"#### You seem to be really good at these!")
            st.dataframe(expert_topics)

            # if basic_analysis:
            st.write(f"#### You might want to brush up your knowledge on these...")
            st.dataframe(basic_topics)



        skip_questionnaire = False

    # Step 8
        if skip_questionnaire:
            st.write(f"#### Which type of messages do you want to learn more about?")
            learn_more_about = st.selectbox("",
                                            chats_analysed_stats['chat_type_ratio'].keys())

            # Step 9 (Placeholder for analyses)
            st.write("Placeholder for displaying analyses")

if uploaded_file is not None:
    skip_questionnaire = st.checkbox("(not recommended for first time use) Skip the questionnaire and jump to analysis results")
    difficulty_pie_input, expert_topics, basic_topics = process_gpt_answer(uploaded_file)
    user_messages_df, user_messages_over_time_df, top_words_df = process_chat_file(uploaded_file)
    chats_analysed_stats['times_used'] = user_messages_over_time_df["num_messages"].sum()
    chats_analysed_stats['top_count_words'] = top_words_df['index'].tolist()


    # if skip_questionnaire:
    #     questionnaire(chats_analysed_stats, user_messages_over_time_df, user_messages_df, top_words_df,
    #         skip_questionnaire=True)
    # else:
    #     questionnaire(chats_analysed_stats, user_messages_over_time_df, user_messages_df, top_words_df)
    questionnaire(chats_analysed_stats, user_messages_over_time_df, user_messages_df, top_words_df,
        difficulty_pie_input, expert_topics, basic_topics)

