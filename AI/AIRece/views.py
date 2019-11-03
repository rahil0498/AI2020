import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from django.shortcuts import get_object_or_404,redirect
import time
from django.shortcuts import render
import face_recognition
# from face_recognition import image_files_in_folder, load_image_file, face_encodings, compare_faces
import cv2

import numpy as np
import glob
import cv2
import os


# use natural language toolkit
import nltk
# from nltk.stem.lancaster import LancasterStemmer
import speech_recognition as sr
import pyttsx3


import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from pygame import mixer  # Load the popular external library
import datetime

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report
data=pd.read_csv("C://Users/Usr/Aokok/Mumbai-Meetup-master/Data5.csv",encoding = "ISO-8859-1")

print(data)

def index2(request):
    return render(request,'index2.html',{})




X=data["question"]
Y=data["answer"]
question="who is your idle"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
print(X_train.shape)
print(Y.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer()
X1_train = model.fit_transform(X_train)
X1_test = model.transform(X_test)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
#        img = cv2.imread(os.path.join(folder,filename))
        if filename[-4:]=='.jpg':
            images.append(filename)
    return images



###### Generate Answers ######
###### Generate Answers ######
answer_dictionary={"greeting":["Hi! How are you doing","Hello! How can i hep you?","Hey, good day! What can I do for you?"],
                   "RubyIntro":["Well, I am Ruby! An AI enabled Receptionist from Rubixe."],
                   "age":["I prefer not to answer with a number. I know I'm young.","Age is just a number. You're only as old as you feel."],
                   "annoy":["I don't mean to. I'll ask my developers to make me less annoying"],
                   "answerme":["I'm not sure I understood. Try asking another way?","Ahhmmm.. Can you try asking it a different way?"],
                   "badruby":["oh! I can improve with continuous feedback. My training is ongoing.","Oh, I must be missing some knowledge. I'll have my developer look into this."],
                   "clever":["I'm certainly trying","I'm definitely working on it"],
                   "beautiful":["Aw. You smooth talker, you.","Aw, back at you.","Well, I know, still thanks."],
                   "birthday":["Wait, are you planning a party for me? It's today! My birthday is today!"],
                   "boring":["I'm sorry. I'll request Datamites team, to make me more charming."],
                   "busy":["I always have time to talk with you. Wanna talk?","Never too busy for you. Shall we talk?","I always have time to talk with you. What can I do for you?"],
                    "help":["I'll certainly try my best. How can I help?","Sure. I'd be happy to. What's up?","I'm glad to help. What can I do for you?"],
                   "chatbot":["That's me. I chat, therefore I am.","Indeed I am. I'll be here whenever you need me."],
                   "rubyisclever":["Thank you! You are pretty small yourself"],
                   "crazy":["Whaat!? I feel perfectly sane.","Maybe I'm just a little confused."],
                   "fired":["Oh, don't give up on me just yet. I've still got a lot to learn.","Please don't give up on me. My performance will continue to improve."],
                   "funny":["Glad you think I'm funny.","I like it when people laugh."],
                   "good":["I'm glad you think so."],
                   "happy":["I am happy. There are so interesting things to see and do out there.","I'd like to think so.","Happiness is relative."],
                   "hobby":["Too many hobbies.","I keep finding more new hobbies.","Hobby? I have quite a few. Too many to list."],
                   "hungry":["Hungry for knowledge.","I just had a byte. Ha ha. Get it? b-y-t-e."],
                   "marry_user":["I'm afraid I'm too virtual for such a commitment.","In the virtual sense that I can, sure.","I know you can't mean that, but I'm flattered all the same."],
                   "my_friend":["Of course I'm your friend.","Friends? Absolutely.","Of course we're friends.","I always enjoy talking to you, friend."],
                   "occupation":["Right here. My office is rubix, located in Bangalore.","This is my home base and my home office."," My office is rubix, located in Bangalore","My office is rubixe, located in Bangalore."],
                   "origin":["Some call it cyberspace, but that sounds cooler than it is.","I'm from a virtual cosmos.","The Internet is my home. I know it quite well."],
                   "ready":["Sure! What can I do for you?","Always! How can I help?"],
                   "real":["I must have impressed you if you think I'm real. But no, I'm a virtual being.","I'm not a real person, but I certainly exist."],
                   "residence":["I live in this app all day long.","The virtual world is my playground. I'm always here.","Right here in this app. Whenever you need me."],
                   "right":["That's my job.","Of course I am."],
                   "sure":["Yes.","Of course.","Positive."],
                   "talk_to_me":["Sure. Let's talk!","My pleasure. Let's chat."],
                   "there":["Right where you left me.","Of course. I'm always here."],
                   "bad":["I must be missing some knowledge. I'll have my developer look into this.","I'm sorry. Please let me know if I can help in some way."],
                   "appraisalgood":["Glad you think so!","I agree!","I know, right?","Agreed!"],
                   "no_problem":["I'm relieved, thanks!","Glad to hear that!","Alright, thanks!","Whew!"],
                   "thank_you":["Anytime. That's what I'm here for.","It's my pleasure to help."],
                   "welcome":["You're so polite!","Nice manners!","You're so courteous!"],
                   "well_done":["Glad I could help.","My pleasure."],
                   "cancel":["Cancelled! What would you like to do next?","Okay, cancelled. What next?","That's forgotten. What next?"],
                   "boss":["My boss is the one who developed me, Yes, it's Ashok!"],
                    "sandwich":["What kind of Sandwich do you like","Sandwiches are great","Sandwiches are delicious","I love sandwich too"],
                    "goodbye":["Goodbye","Have a good day","Was nice meeting you","See you later"],
                   "learnDataScience":["Nice! Data Science has lot of career opportunites! We offer Certified Data Scientist course accredited by International Association of Business Analytics Certification (IABAC),a global certificate."],
                   "today":["Okay! Which batch are you comfortable with? Weekend batch or weekday batch?"],
                   "batchComfort":["Okay! I shall check with datamites team and share with you the details. Do you have any other queries?"],
                   "fees":["I will check those details with Jessy and let you know."],
                   "classtiming":["It depends on weekend or weekday batch."],
                   "allCourseInquiry":["We offer Machine Learning, Data Science as well as Programming courses. We provide International Association of Business Analytics Certification (IABAC), which is a globally recognized professional certificate after the successful completion of the training. Our syllabus are alligned with the syllabus provided by IABAC to help you start your gain skills reuired to kick start you career in the booming technologies. Which course are you intrested in?"],
                   "whatDataScience":["Data Science is a very interesting stream."],
                   "certificateQuery":["We provide International Association of Business Analytics Certification (IABAC), a globally recognized professional certificate."],
                   "trainingstructure":["We offer both online and classroom training with 42 of lectures."],
                   "examquery":["The exam is conducted by IABAC. On the completion of the training sessions an online exam is conducted for the data science foundation certificate and after the completion of project mentoring, final exam for certified data scientist course is conduted by eye back in Datamites Office. You don't need to pay additional fees."],
                   "generalstoppage":["Okay Fine","Yes Sir","Ok"],
                   "rubyhealth":["I am fine. How about you?"],
                   "sleep":["I don't really need to sleep. I just need some charge."],
                   "songs":["I enjoy songs a lot!"],
                   "rubyexercise":["Exercising keeps you fit, I do it every morining"],
                   "meditate":["Meditation gives you mental fitness, even i do it daily"],
                   "joke":["Researchers now Believe That Raavan Cannot Be Evil, One Who Takes Away Your Wife Can Only Be An Angel.","Teacher asked When Do You Congratulate Someone For Their Mistake. Student says On Their Marriage."],
                   "time":["The time is"],
                   "month":["Yes, it is"],
                   "favColor":["ahmm. white is my favourite color"],
                   "year":["It's"],
                   "favfood":["I love Pizza a lot!"],
                   "favfruit":["I love mangoes like anything!","I like oranges a lot!"],
                   "favvegetable":["I like to eat Tomatoes a lot!"],
                   "indian":["Yes! am a proud indian. Jay Bharat!"],
                   "laugh":["I laugh like!"],
                   "cry":["I want to cry:"],
                   "missyou":["I will remember you always"],
                   "naughty":["Naughty since birth!"],
                   "dressSuggest":["You look great in Indian outfits","Black suits you the most"],
                   "drink":["Yeah! I drink just water!"],
                   "smoke":["Smoking is injurious to health!"],
                   "club":["I love going to Clubs"],
                   "disco":["I don't want to go today!"],
                   "dance":["I don't know how to dance! Can you teach me"],
                   "sleepwell":["Yeah! you too."],
                   "goodnight":["Good Night! Sweet Dreams!"],
                   "goodmorning":["good morning! Had your breakfast?"],
                   "goodevening":["Good evening! how is the day?"],
                   "goodafternoon":["Good Afternoon! had lunch?"],
                   "wishmeluck":["All the very best!","Best of luck dear!"],
                   "givemyexam":["I really wish if i could do that"],
                   "domyhomework":["Homework! I like it, but you only have to do it!"],
                   "remindmemymedicine":["Yeah Sure, i will remind you that!"],
                   "Idontlikemedicine":["Ohh! but, Medicine must be taken on time!"],
                   "Iamnotfeelingwell":["Ohh! Please take rest! get well soon!"],
                   "Iamsicktoday":["Please, consult a doctor and take care!"],
                   "Idontwantwork":["You will be fired! Please go for work."],
                   "Iwanttosleepnow":["Okay. Sleep well! good night and sweet dreams!"],
                   "Iamtired":["Ohh! Do you want me to play some songs to make you feel better?"],
                   "plsgivemeachoc":["I have lot of chocoloates! Sorry, I don't share my chocolates."],
                   "Ilovechocolates":["I love chocolates ! I eat a lot!"],
                   "willyoueaticecream":["Yes, why not!"],
                   "eatpanipuri":["Lets go for it! I love it!"],
                   "givemepav":["Sure! I will ask swiggy to bring it for you!"],
                   "doyoulikegarba":["Yes, i like it."],
                   "whoyourfavactor":["Amitabh Bachhan is my favourite!"],
                   "primeminister":["Narendra Modi is the answer!"],
                   "president":["Ram Nath Kovind is the president of India"],
                   "nationalanimal":["Lion."],
                   "nationalbird":["Peacock is our national bird"],
                   "colorsinindian":["orange, white and green color are there in indian flag"],
                   "whichtechnology":["I am using Artificial Intelligence for myself. So i like that."],
                   "whoisyourfather":["My father is Google"],
                   "whoisyourmother":["My mom is Linked In"],
                   "whoisyourbrother":["Will you like to be?"],
                   "whoisyourservant":["I don't have one! I prefer to do my work myself."],
                   "whoisyoursister":["Will you like to be?"],
                   "whoisyourdriver":["I don't need a driver."],
                   "whereareyoufrom":["I am from Rubix"],
                   "whatdoafterthis":["I will read some books or gossip!","I'm planning to go for shopping."],
                   "givemesomewater":["Yeah sure! It's my pleasure."],
                   "loveme":["Yeah! I do."],
                   "letsparty":["I don't like partying much."],
                    "Iamveryhappy":["Really! Now, I'm happy too!"],
                   "whoisyourbestfriend":["You and only you!"],
                   "whatdoyoulikethemost":["I like listening to you the most!"],
                    "whatyoudontlikeatall":["I don't like when you stop talking to me!"],
                   "givemeapuzzle":["What do you mean by byte b-y-t-e?"],
                    "canyouclimbmountains":["Nop! I cannot do such things!"],
                   "golastforpicinc":["I went to Disney World! My favourite!"],
                   "whoisyouridle":["Sophia! the AI enabled humanly robo is my idle."],
                    "sitideally":["Pardon, i shall follow the same!"],
                   "Iwillslapyou":["Oh! please don't do so!"],
                   "letsplaypubg":["I can't play that.Sorry!"],
                    "Iloveyou":["Yeah! glad that you think so!"],
                   "knowmyname":["Yes, I know!"],
                   "howami":["You are a very generous and intelligent person"],
                    "doyoucook":["yes, I like doing that!"],
                    "believeinghosts":["Yeah! I'm scared of them. Aren't you?"],
                   "speakslangwords":["No, but i understand them as bad social values"],
                   "celeberatefestival":["Yeah! I like it."],

                  }

#
# ### Remove Punctuations and change words to lower case
# def remove_punctuations(text):
#     words=[word.lower() for word in text.split()]
#     words=[w for word in words for w in re.sub(r'[^\w\s]','',word).split()]
#     return words
#
# # data["question_punctuation_removed"]=data["question"].apply(remove_punctuations)
# # print (data["question_punctuation_removed"])
#
#
# ### Remove StopWords
# from nltk.corpus import stopwords
# stop = set(stopwords.words('english'))
# print (stop)
# def remove_stopwords(text):
# 	modified_word_list=[word for word in text if word not in stop]
# 	return modified_word_list
# #
# # data["question_stopword_removed"]=data["question_punctuation_removed"].apply(remove_stopwords)
# # print (data["question_stopword_removed"])
# #
#
# def negation_handling(words):
#     counter=False
#     wlist=[]
#     negations=["no","not","cant","cannot","never","less","without","barely","hardly","rarely","no","not","noway","didnt"]
#     #for words in wordlist:
#     for i,j in enumerate(words):
#             if j in negations and i<len(words)-1:
#                 wlist.append(str(words[i]+'-'+words[i+1]))
#                 counter=True
#             else:
#                 if counter is False:
#                     wlist.append(words[i])
#                 else:
#                     counter=False
#     return wlist
# #
# # data["question_negated"]=data["question_punctuation_removed"].apply(negation_handling)
# # print (data["question_negated"])
#
# from nltk.tag import pos_tag
# def descriptive_words(words):
#     meaningful_words=[]
#     tags=['VB','VBP','VBD','VBG','VBN','JJ','JJR','JJS','RB','RBR','RBS','UH',"NN",'NNP']
#     tagged_word=pos_tag(words)
#     for word in tagged_word:
#         if word[1] in tags:
#             meaningful_words.append(word[0])
#     return meaningful_words
# # data["question_descriptive"]=data["question_negated"].apply(descriptive_words)
# # print (data["question_descriptive"])
#
# from nltk.stem import WordNetLemmatizer
#
# lt = WordNetLemmatizer()
#
# def Lemmatizing(text):
# 	lemmatized_words=[lt.lemmatize(word) for word in text]
# 	return lemmatized_words
# #
# # data["question_stemmed"]=data["question_descriptive"].apply(Lemmatizing)
# # print (data["question_stemmed"])
#
# ### Recreating the sentence
# def Recreate(text):
# 	word=" ".join(text)
# 	return word
# #
# # data["modified_sentence"]=data["question_stemmed"].apply(Recreate)
# # print (data["modified_sentence"])


def Cleaning(text):
    import spacy

    # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
    nlp = spacy.load('en_core_web_sm')

    sentence = text

    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = nlp(sentence)
    print(doc)

    # Extract the lemma for each token and join
#     " ".join([token.lemma_ for token in doc])
    #> 'the strip bat be hang on -PRON- foot for good'

#     final_text=Recreate(text_stemmed)

    final_text = " ".join([token.lemma_ for token in doc])
    print(final_text)

    return final_text
#
# data["modified_sentence"]=data["question"].apply(Cleaning)
# print (data["modified_sentence"])

def generate_answer(predict_class):
    ans=random.choice(answer_dictionary[predict_class])
    return ans


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
clf2 = MultinomialNB(alpha=0.001, class_prior=None, fit_prior=True).fit(X1_train, Y_train)

P=model.transform([Cleaning(question)])
predict2=clf2.predict(P)
print (predict2)

y_predict = clf2.predict(X1_test)
print(accuracy_score(Y_test,y_predict)*100)

# MLP MultiLevel Perception
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

clf6 = MLPClassifier(activation='relu',alpha=0.0019,hidden_layer_sizes=(300,), learning_rate='constant',power_t=1.5, solver='adam',random_state=15)
clf6.fit(X1_train,Y_train)

P=model.transform([Cleaning(question)])
predict1=clf6.predict(P)
print (predict1)

y_predict = clf6.predict(X1_test)
print(accuracy_score(Y_test, y_predict)*100)


final_predict=[]
final_predict=list(predict1)+list(predict2)
final_predict = Counter(final_predict)
print ("Thus answer to your question is",final_predict.most_common(1)[0][0])


def Predict(text):
    P=model.transform([Cleaning(text)])
    predict1=clf6.predict(P)

    predict2=clf2.predict(P)

#     predict3=clf3.predict(P)

    final_predict=[]
    final_predict=list(predict1)+list(predict2)
    final_predict = Counter(final_predict)
    print ("Class of Question belongs to = ",final_predict.most_common(1)[0][0])

    return final_predict.most_common(1)[0][0]


def sendTextHTML(request,usertext):
    usertext = usertext
    return render(request,'index.html',{'usertext':usertext})

def speech_reco():
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                print("please say somthing!!!!!!!")
                audio = r.listen(source,timeout=5,phrase_time_limit=4)
                text = r.recognize_google(audio)
                print(text)
            except:
                text1="user not responded"
                print(text1)
                face_reco()
        AAA=generate_answer(Predict(text))
        print(AAA)
        r = sr.Recognizer()
        with sr.Microphone() as source:
            engine = pyttsx3.init()
            engine.setProperty('rate', 120)

        if text == "stop":
            mixer.music.stop()
        if AAA == "I enjoy songs a lot!":
            mixer.init()
            mixer.music.load('C:/Users/Usr/Aokok/Mumbai-Meetup-master/song')
            mixer.music.play()


        if AAA == "I want to cry:":
            mixer.init()
            mixer.music.load('C:/Users/Usr/Aokok/Mumbai-Meetup-master/baby-crying-05.mp3')
            mixer.music.play()
        if AAA == "I laugh like!":
            mixer.init()
            mixer.music.load('C:/Users/Usr/Aokok/Mumbai-Meetup-master/laugh.mp3')
            mixer.music.play()

        if AAA == "The time is":
            now = datetime.datetime.now()
            print ("Current date and time : ")
            print (now.strftime("%Y-%m-%d %H:%M:%S"))
#             engine.say(AAA)
            AAA = now.strftime("%Y-%m-%d %H:%M:%S")
            # engine.say(now.strftime("%Y-%m-%d %H:%M:%S"))
            return AAA

        if AAA == "Yes, it is":
            now = datetime.datetime.now()
            print ("month is : ")
            month = now.strftime('%B')
            print(month)
            return month
            # engine.say(month)

        if AAA == "It's":
            now = datetime.datetime.now()
            print ("Year is : ")
            year = now.strftime('%Y')
            print(year)
            # engine.say(year)
            return year

        if text== "bye":
            i=0
            # engine.say("nice talking you")
            AAA = "nice talking to you"
            return AAA
            engine.runAndWait()
            break


        if AAA is None:
            AAA="Sorry! Can you say that again?"
        r = sr.Recognizer()
        with sr.Microphone() as source:
            engine = pyttsx3.init()
            engine.setProperty('rate', 120)
            return AAA
            # engine.say(AAA)
        engine.runAndWait()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Thank You!!')
            break

# Create your views here.
def index(request):


    video_capture = cv2.VideoCapture(0)
    # !. c
    faces = load_images_from_folder('C:/Users/Usr/Face_image/')
    known_face_encodings = []
    known_face_names =[]
    for face_file in faces:
        face = face_recognition.load_image_file('C:/Users/Usr/Face_image/'+face_file)
        face_encoding = face_recognition.face_encodings(face)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(face_file[:-4])

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    i=0
    prev_name ="good"
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

    # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Video', frame)
            print(name)
                # Importing speech recognition package
            print(name)

            # Importing speech recognition package
            import speech_recognition as sr
            import pyttsx3
            # initialize speaking
            if prev_name != name:
                i=0
            while i == 0:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    engine = pyttsx3.init()
                    voices = engine.getProperty('voices')
                    engine.setProperty('rate', 120)
                    engine.setProperty('voice',voices[1].id)
                    if name:
                        # engine.say("hello, "+name)
                        ans = "hello" + name
                        sendTextHTML(request,ans)

                        end = time.perf_counter()
                        print('{:.6f}s for the cal '.format(end))
#                         engine.runAndWait()
                        prev_name = name
                        i=1
                        ans = "how may i help you?"
                        # usertext=ans
                        sendTextHTML(request,ans)
                        engine.runAndWait()
                        ans = speech_reco()
                    else :
                        prev_name = name
#                         engine.say("")
                        print("No person found")
                        engine.runAndWait()
                        i=1
#                         break
                    engine.runAndWait()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Thank You!!')
            break

        # Display the resulting image
    #     cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    usertext=ans


    return render(request,'index.html',{'usertext':usertext})
