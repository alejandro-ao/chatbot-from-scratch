# Chatbot from scratch

## Description
This is a chatbot application developed in Python using the Random Forest model. The chatbot is designed to interact with users by understanding their input and providing relevant responses based on predefined intents. The model is trained on a dataset consisting of 500 question-answer pairs, which were augmented from an initial set of 90 pairs through paraphrasing techniques. The application is deployed using Streamlit, a Python library for building interactive web applications.

## Model Creation
The process of creating the chatbot model involved several steps:

1. Data Augmentation: The original dataset of 90 question-answer pairs was expanded to 500 pairs through paraphrasing techniques. This helped to increase the variety of inputs and improve the model's performance.

2. Data Preprocessing: Before training the model, the data underwent preprocessing steps, including:
   - Lowercasing: All text was converted to lowercase to ensure case-insensitive matching.
   - Punctuation Removal: Punctuation marks were removed to focus on the essential words and improve tokenization accuracy.
   - Lemmatization: Words were reduced to their base form (lemmas) to handle different inflections and improve generalization.
   - Stop Word Removal: Common words without significant meaning, such as articles and prepositions, were removed to reduce noise in the data.

3. Tokenization: The preprocessed text was split into individual tokens (words) to facilitate further processing and analysis.

4. Vectorization: To enable the machine learning model to work with the textual data, the tokens were transformed into numerical representations. This process, known as vectorization, allows the model to understand and process text data effectively.

5. Intent Classification: The chatbot model was trained to classify user inputs into one of 14 predefined intents. These intents represent the different types of questions or statements that users might make. The Random Forest model was chosen for its ability to handle multi-class classification tasks effectively.

6. Response Selection: Once the intent was determined, the model selected the most appropriate response from a predefined set of responses associated with each intent. The response selection is based on identifying the closest response that matches the user's intent.

## Deployment
The chatbot application is deployed using Streamlit, a popular Python library for creating web applications. Streamlit allows for the easy and interactive deployment of data-driven applications without extensive web development knowledge.

To run the chatbot application, follow these steps:

1. Ensure that you have Python and pip installed on your system.

2. Install the required dependencies by running the following command in your terminal:
   ```
   pip install -r requirements.txt
   ```

3. Once the dependencies are installed, navigate to the `app` directory.

4. Run the application using the following command:
   ```
   streamlit run main.py
   ```

5. The application will start running locally, and you can access it by opening the provided URL in your web browser.

6. Interact with the chatbot by typing your questions or statements in the input field and observing the responses provided by the model.

## Additional Notes
- The accuracy and performance of the chatbot model can be improved by further refining the training data, increasing the dataset size, or exploring other machine learning algorithms.
- The response selection mechanism can be enhanced by incorporating more sophisticated techniques, such as semantic similarity or deep learning-based approaches.
- The deployment of the chatbot application can be scaled to various platforms or cloud services to make it accessible to a broader audience.

Please note that this readme file provides an overview of the chatbot app, its creation process, and deployment instructions. For more detailed information, refer to the source code and relevant documentation.

Enjoy using the chatbot app!