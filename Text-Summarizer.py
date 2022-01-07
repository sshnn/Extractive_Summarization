import math
import nltk 
from nltk.stem import WordNetLemmatizer 
import spacy

filePath = "C:\\Users\\salih\\OneDrive\\Desktop\\Tasarım_Proje\doc.txt"

nltk.download('wordnet')


nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer() 


def file_text(filepath):
    with open(filepath) as f:
        text = f.read().replace("\n", '')
        return text


    

#Text'i alır 


txt_path = filePath
text = file_text(txt_path)


#Tf-Idf Matrix
#Her cümlede frekans hesaplayan fonksiyon

def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words

    for sent in sentences:
        freq_table = {} 
        
        
        words = [word.text.lower() for word in sent  if word.text.isalnum()]
       
        for word in words:  
            word = lemmatizer.lemmatize(word)  
            if word not in stopWords:           
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1

        freq_matrix[sent[:15]] = freq_table

    return freq_matrix


#Her kelimede frekans hesaplayan fonksiyon
#TF(t) = (Number of times term t appears in  document) / (Total number of terms in the document)
def tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, freq_table in freq_matrix.items():
        tf_table = {} 

        total_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


#cümledeki kelime frekansını bulur
#cümledeki kelimenin hesaplanan frekansına göre değer verir

def sentences_per_words(freq_matrix):
    sent_per_words = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in sent_per_words:
                sent_per_words[word] += 1
            else:
                sent_per_words[word] = 1

    return sent_per_words


#Ters Döküman Sıklığı
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


#bütün cümlelerdeki kelimelerin frekans matrisindeki değerleri birleştirir
def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

       #word1 and word2 are same
        for (word1, tf_value), (word2, idf_value) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(tf_value * idf_value)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# bütün cümlelerdeki kelimelerin frekans matrisindeki cümlelerin değerleri birleştirir

def score_sentences(tf_idf_matrix):
    
    sentenceScore = {}

    for sent, f_table in tf_idf_matrix.items():
        total_tfidf_score_per_sentence = 0

        total_words_in_sentence = len(f_table)
        for word, tf_idf_score in f_table.items():
            total_tfidf_score_per_sentence += tf_idf_score

        if total_words_in_sentence != 0:
            sentenceScore[sent] = total_tfidf_score_per_sentence / total_words_in_sentence

    return sentenceScore



# text'in genel frekans ortalaması hesaplanır
def average_score(sentence_score):
    
    total_score = 0
    for sent in sentence_score:
        total_score += sentence_score[sent]

    average_sent_score = (total_score / len(sentence_score))

    return average_sent_score


#Ortalama frekans değerinden yüksek veya eşitse yazdırır
def create_topic(sentences, sentence_score, threshold):
    topic = ''

    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= (threshold):
            topic += " " + sentence.text
        

    return topic




#Orijinal text'deki kelime sayısını sayma
original_words = text.split()
original_words = [w for w in original_words if w.isalnum()]
num_words_in_original_text = len(original_words)


text = nlp(text)

sentences = list(text.sents)
total_sentences = len(sentences)

#frekans matrisini oluşturur
freq_matrix = frequency_matrix(sentences)

#terim frekans matrisini oluşturu
tf_matrix = tf_matrix(freq_matrix)

#Frekansı fazla olan kelimenin cümlesini alır
num_sent_per_words = sentences_per_words(freq_matrix)
idf_matrix = idf_matrix(freq_matrix, num_sent_per_words, total_sentences)
tf_idf_matrix = tf_idf_matrix(tf_matrix, idf_matrix)


#Her cümlenin frekansını alır
sentence_scores = score_sentences(tf_idf_matrix)


threshold = average_score(sentence_scores)

#konu içeriğini oluşturur
topic = create_topic(sentences, sentence_scores, 1.3 * threshold)
print("\n")
print("*"*20,"KONU","*"*20)
print("\n")
print(topic)
print("\n\n")
print("Dökümandaki toplam kelime sayısı = ", num_words_in_original_text)
print("Konudaki toplam kelime sayısı = ", len(topic.split()))
