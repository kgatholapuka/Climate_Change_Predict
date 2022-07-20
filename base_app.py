"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
# Data dependencies
import pandas as pd
import Data_cleaning as cleanText
from nlppreprocess import NLP
nlp = NLP()

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
image = Image.open('img/3.png')
st.set_page_config(page_title='Networkers.net',page_icon = image)

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	image = Image.open('img/1-removebg-preview.png')
	net = Image.open('img/net.jpeg')
	net = Image.open('img/net.jpeg')
	#intro = open("vid/ntr.mp4","rb")
	#video1 = open("vid/Donald.mp4","rb")
	#video2 = open("vid/N.mp4","rb")
	#video3 = open("vid/Neg.mp4","rb")
	#video4 = open("vid/Pro.mp4","rb")
	#video1 = st.video(video1)

	#st.image(image,caption='Climate Change is killing our future')
	
	link_news =  'Click the link:[News Sentiment](https://youtu.be/2UaaqoHKAK8)'

	link_Pro =  'Click the link:[Pro Sentiment](https://youtu.be/LnnDOMyZjbE)'

	link_Negative = 'Click the link:[Anti Sentiment](https://youtu.be/qpyFjUsm2PU)'

	link_Neutral =  'Click the link:[Neutral Sentiment](https://youtu.be/ga-RBuhcJ7w)'

	link =  'Click the link :[Learn more about climate change?](https://youtu.be/EtW2rrLHs08)'

	quzi = "Click the link to [Take a free online climate change quiz and see how you perceive climate change](https://www.earthday.org/the-climate-change-quiz/)"

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home🏠","Prediction❄️", "Information📰","Analysis📘","About"]
	selection = st.sidebar.selectbox("Main Menu", options)
	
   
	#Home Page
	if selection == 'Home🏠':
		st.title('Welcome to Networkers.net')
		st.subheader('Climate Change Project')
		st.image(image)
		st.markdown("""
		WE ARE THE FIRST GENERATION TO FEEL THE IMAPCT OF CLIMATE CHANGE AND THE LAST GENERATION THAT CAN DO SOMETHINGG ABOUT IT. -- BARACK OBAMA --                            
		""")
		st.write("""
		This web application aims to predict an individual's beliefs about climate change based on historical tweet data and provide insights into different climate change emotion 
		classes. Climate change has long been recognized by the scientific community as a major problem facing humankind. But even with the scientific consensus on climate change,
		 public opinion on this issue can be different. The goal of this task is to use tweet data to build a machine learning classification model that 
		 determines whether a person believes in climate change. 
		The project also provides insights into public opinion on climate change.
		""")
		
		
		st.markdown(link)

		
	

		#st.write('Thank you for stopping by! We\'re an online information system company ' 
		#'that focus on Artificial intelligence,Machine Learning and Data Analysis Development .')
		#st.write(
		#		"""
		#		Thank you for stopping by! We\'re an online information system company 
		#that focus on Artificial intelligence,Machine Learning and Data Analysis Development
		#		""")
		#st.image(net,caption="Networkers.net")



	# Building out the "Information" page
	if selection == "Information📰":
		st.info("DataSet Information")
		
		# You can read a markdown file from supporting resources folder
		st.markdown("""
		The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.
This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. 
Each tweet is labelled independently by 3 reviewers. 
This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).
    Each tweet is labelled as one of the following classes:
		""")

		st.markdown("""
		
		* 2 News: the tweet links to factual news about climate change
* 1 Pro: the tweet supports the belief of man-made climate change
* 0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change
* -1 Anti: the tweet does not believe in man-made climate change
		""")
       
		st.markdown("Below is raw data that we used for Bulding the model")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction❄️":
		st.header("Prediction of Climate change")
		st.write("Purpose: To provide data on how people perceive climate change. Our model will predict whether the text indicates whether climate change is caused by humans.")
		st.markdown(quzi)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter your Text")

		model_name = ["Decision Tree Classifier", "Random Forest Classifier", "Logistic Regression Classifier"]
		#selection = st.sidebar.selectbox("Choose Option", options)
		model_choice = st.selectbox(
                "Select a Classifier  Model", model_name)
		
	    

		if st.button("Classify"):
			# Transforming user input with vectorizer
			text = cleanText.cleaner(tweet_text) # cleaning the text using preprocessing function
			vect_text = tweet_cv.transform([text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			
			if model_choice == "Decision Tree Classifier":
				predictor_tree = joblib.load(open(os.path.join("resources/Dec_tree_model.pkl"),"rb"))
				prediction = predictor_tree.predict(vect_text)
			elif model_choice == "Support Vector Classifier":
				predictor_svm = joblib.load(open(os.path.join("resources/Random_model.pkl"),"rb"))
				prediction = predictor_svm.predict(vect_text)
			elif model_choice == "Logistic Regression Classifier":
				predictor_lr = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor_lr.predict(vect_text)

	
			

			if prediction ==1 :
				st.success("wow, you believe in climate change")
			elif prediction == 0 :
				st.success("ummh, you don't supports nor refuse the belief of man-made climate change")
			elif prediction == -1 :
				st.success("Nope, you don't believe in climate change")
			elif prediction == 2 :
				st.success("well, your tweet links to factual news about climate change")

			
		
	if selection == 'Analysis📘':
		st.header('Welcome to the Analysis page')
		st.write("""
		Global warming is the long-term increase in the average temperature of the Earth's climate. Human activity can cause 95% of global warming on Earth.
		 Our model will help companies  analyze tweets to determine if someone believes in climate change.
		""")
		graphs = ["Tweet Sentiment Distribution", "Sentiment Length Distribution","Popular Words for News Tweets","Popular Words for pro Tweets", "Popular Words for Neutral Tweets", "Popular Words for Anti Tweets"]
		graphs_choice = st.selectbox(
                "Select a graph", graphs)
		if graphs_choice == "Tweet Sentiment Distribution":
			with st.expander("See explanation"):
				st.write("""
				Taking a closer look at the distribution of tweets, we found that the data was severely skewed 
, with the majority of tweets being of the "professional" category, with 
 advocating belief in human-caused climate change. create. 
				""")
        
			st.image("img/Distribution.png",use_column_width=True)
			
		elif graphs_choice == "Popular Words for Anti Tweets":
			with st.expander("See explanation"):
				st.write("""
				Most of the keywords in  negative emotions are very political and scientific, showing many of the views of 
 world leaders  on the topic of climate change. Trump, a  climate change staunch individual appears very substantial including scientific terms, fabricated, fake, alarmist 
  shows that many people do not believe it to be true on a hunch or because  lack of scientific evidence to support the claim. 
 There are also a lot of words like scam, 
 money, man made that indicate one of the reasons why they might not really believe in climate change or feel negative about it.
				""")
			st.write(link_Negative)
			st.image("img/negative words.png")
			
		elif graphs_choice == "Popular Words for News Tweets":
			with st.expander("See explanation"):
				st.write("""
				As we can see, Donald Trump plays a huge role when it comes to  news sentiment. 
 They also reported that most of the problems were included in the analysis of  other included words. Executive Orders also appear frequently. 
 The word cloud  also shows that the words are well distributed and  almost identical in pronunciation. 
 Most  words are not asked often, except "climate change". .
				""")
			st.write(link_news)
			st.image("img/new words.png")
			
		elif graphs_choice == "Popular Words for pro Tweets":
			with st.expander("See explanation"):
				st.write("""
				The word cloud suggests that the most positive feelings are global warming, climate change, news, reality and the likes expressed in the word cloud. 
                No link or http shows that we did a lot of data cleaning and it works. With the word cloud, we can see  words that match positive sentiments.
				""")
			st.write(link_Pro)
			st.image("img/positive words.png")
			
		elif graphs_choice == "Popular Words for Neutral Tweets":
			with st.expander("See explanation"):
				st.write("""
				The majority of neutrals discuss, participate and question the impacts of climate change, as seen with the interviewers and scientists. 
                They talk about  penguins in danger from the effects of climate change. They talk about  climate change, global warming..
				""")
			st.write(link_Neutral)
			st.image("img/neutral words.png")
			
		elif graphs_choice == "Sentiment Length Distribution":
			with st.expander("See explanation"):
				st.write("""
				The graphs show that all opinions have roughly similar average lengths, which may be because each tweet has a word limit. However, 
				there is a noticeable difference when we compare the density of positive and negative sentiments.
				 To determine what needs to be done in the sentimental analysis stage, we will do additional analysis.
				""")
			st.image("img/line.png",use_column_width =True)	
			
		
	

	
	if selection == 'About':
		st.header("Networkers.net")
		st.image('img/Deep-Learning.jpg',caption="Networkers.net")
		#st.markdown(link, unsafe_allow_html=True)
		st.write(
				"""
				Welcome to Networkers, your number one source for Artificial intelligence,Machine Learning and Data Analysis Development. 
				We're dedicated to giving you the very best .
Founded in 2021 by Kgathola Puka, Networkers has come a long way from its beginnings in a
home office in Hillbrow. When Mr Puka first started out,
his  passion for Data which  drove him to  do intense research, and gave him the impetus to turn hard work and inspiration into to a booming online 
store. We now serve customers all over the Country and Soon will be going global, and are thrilled to be a part of the 
AI concept wing of the industry.

We hope you enjoy our products as much as we enjoy offering them to you. 
If you have any questions or comments, please don't hesitate to contact us.

Sincerely,

Kgathola Puka,
CEO and Founder
				""")

		st.header(":mailbox:Get in Touch with me!")
		contact_form = """

	 <form action="https://formsubmit.co/kgatholapuka@gmail.com" method="POST">
	 <input type="hidden" name="_captcha" value="false">	
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email" placeholder = "Your email" required>
	 <textarea name="message" placeholder="Your message"></textarea>
     <button type="submit">Send</button>
     </form>		"""

		st.markdown(contact_form,unsafe_allow_html=True)
		

	
		

		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")