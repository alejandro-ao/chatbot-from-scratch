import pandas as pd
import openai
import os
from dotenv import load_dotenv


def paraphrase_sentence(sentence):
    # Set up OpenAI API credentials
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Define the prompt and the maximum response length
    prompt = "Paraphrase the following sentence:\n\"{}\"\n\nParaphrased version:".format(
        sentence)
    max_tokens = 50

    # Generate paraphrased version using OpenAI GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract the paraphrased version from the response
    paraphrased_sentence = response.choices[0].text.strip()

    return paraphrased_sentence


def augment_df(df):
    # Create an empty dataframe to store augmented data
    augmented_df = pd.DataFrame(columns=['question', 'answer'])

    # Iterate over the rows of the original dataframe
    for index, row in df.iterrows():
        # Paraphrase the question and answer
        paraphrased_question = paraphrase_sentence(row['question'])
        paraphrased_answer = paraphrase_sentence(row['answer'])

        # Add the original question and answer to the augmented dataframe
        augmented_df = augmented_df.append(
            {'question': row['question'], 'answer': row['answer']}, ignore_index=True)

        # Add the paraphrased question and answer to the augmented dataframe
        augmented_df = augmented_df.append(
            {'question': paraphrased_question, 'answer': paraphrased_answer}, ignore_index=True)

    return augmented_df
