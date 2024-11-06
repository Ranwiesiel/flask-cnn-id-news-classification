import requests as req
from bs4 import BeautifulSoup as bs

# Library untuk data manipulation & visualisasi
import pandas as pd
import networkx as nx
import re

# Library untuk text preprocessing
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')

# Library untuk text vectorization & Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Cleaning text Berita
def clean_text(text: str=None) -> str:
	"""
	Mmembersihkan text dari karakter-karakter yang tidak diperlukan
	"""
	text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text) # Menghapus https* and www*
	text = re.sub(r'@[^\s]+', ' ', text) # Menghapus username
	text = re.sub(r'[\s]+', ' ', text) # Menghapus tambahan spasi
	text = re.sub(r'#([^\s]+)', ' ', text) # Menghapus hashtags
	text = re.sub(r"[^a-zA-Z0-9 / :\.]", "", text) # Menghapus tanda baca
	text = text.lower()
	text = text.encode('ascii','ignore').decode('utf-8') #Menghapus ASCII dan unicode
	text = re.sub(r'[^\x00-\x7f]',r'', text)
	text = text.replace('\n','') #Menghapus baris baru
	text = text.strip()
	return text

def preprocess_text_ringkas(text):
	"""
	Memproses text berita untuk ringkasan
	"""
	result = ""
	cleaned_text = clean_text(text)
	tokens = nltk.tokenize.word_tokenize(cleaned_text)
	result = ' '.join(tokens)
	kalimat = nltk.sent_tokenize(result)
	return kalimat


def network_graph(cossim):
	"""
	Membuat graph dari similarity matrix
	"""
	G_preprocessing = nx.DiGraph()
	for i in range(len(cossim)):
		G_preprocessing.add_node(i)

	for i in range(len(cossim)):
		for j in range(len(cossim)):
			similarity_preprocessing = cossim[i][j]
			if similarity_preprocessing > 0.1 and i != j:
				G_preprocessing.add_edge(i, j)
	return G_preprocessing

def centrality(G, centrality_type="degree"):
	"""
	Opsi untuk menghitung nilai centrality dari graph
	- degree
	- eigenvector
	- betweenness
	- closeness
	- pagerank
	"""
	if centrality_type == "degree":
		return nx.degree_centrality(G)
	elif centrality_type == "eigenvector":
		return nx.eigenvector_centrality(G)
	elif centrality_type == "betweenness":
		return nx.betweenness_centrality(G)
	elif centrality_type == "closeness":
		return nx.closeness_centrality(G)
	elif centrality_type == "pagerank":
		return nx.pagerank(G)
	else:
		raise ValueError(f"Unknown centrality type: {centrality_type}")

def sorted_result(node, kalimat, total=3):
	"""
	Mengurutkan hasil berdasarkan nilai centrality
	"""
	closeness_centrality = sorted(node.items(), key=lambda x: x[1], reverse=True)

	ringkasan = ""
	for node, closeness_preprocessing in closeness_centrality[:total]:
		top_sentence = kalimat[node]
		ringkasan += top_sentence + " "

		# print(f"Node {node}: Closeness Centrality = {closeness_preprocessing:.4f}")
		# print(f"Kalimat: {top_sentence}\n")
	return ringkasan






# Scraping berita
def scrape_news(soup: str) -> dict:
	"""
	Mengambil informasi berita dari url
	"""
	berita = {}
	texts = []
	# TODO:
	# ada struktur aneh https://www.cnnindonesia.com/olahraga/20240830134615-142-1139388/live-report-timnas-indonesia-vs-thailand-u-20
	
	berita["judul"] = soup.title.text

	if 'FOTO:' in berita["judul"]:
		div_content = soup.find("div", class_="detail-text text-cnn_black text-sm grow min-w-0")
		if div_content:
			full_text = div_content.get_text(strip=True)
			text = full_text.split('--', 1)[-1]
			text = text.split('var article')[0].strip()

			cleaned_text = clean_text(text)
			texts.append(cleaned_text)

		berita["tanggal"] = soup.find("div", class_="container !w-[1100px] overscroll-none").find_all("div")[1].find_all("div")[2].text

	else:
		text_list = soup.find("div", class_="detail-text text-cnn_black text-sm grow min-w-0")
		for text in text_list.find_all("p"):
			if 'para_caption' not in text.get('class', []):
				cleaned_text = clean_text(text.text)
				texts.append(cleaned_text)

		berita["tanggal"] = soup.find("div", class_="container !w-[1100px] overscroll-none").find_all("div")[1].find_all("div")[3].text

	berita["isi"] = "\n".join(texts)
	berita["kategori"] = soup.find("meta", attrs={'name': 'dtk:namakanal'})['content']
	berita["url"] = soup.find("meta", attrs={'property': 'og:url'})['content']
	return berita

# Mengambil html dari url
def get_html(url: str) -> str:
	"""
	Mengambil html dari url
	"""
	try:
		response = req.get(url).text
		return bs(response, "html5lib")
	
	except Exception as e:
		print(e)
		return ""

def get_news(news_url: str) -> pd.DataFrame:
	"""
	Mengambil informasi dari isi berita yang ada pada url
	"""
	news = []

	result = scrape_news(get_html(news_url))
	news.append(result)

	df = pd.DataFrame.from_dict(news)

	return df


def scrape_news_public(soup: str) -> dict:
	"""
	Mengambil informasi berita Public dari url
	"""
	berita = {}
	texts = []
	berita["judul"] = soup.title.text

	for text in soup.find_all("p"):
		if 'para_caption' not in text.get('class', []):
			cleaned_text = clean_text(text.text)
			texts.append(cleaned_text)
	berita["isi"] = "\n".join(texts)
	berita["url"] = soup.find("meta", attrs={'property': 'og:url'})['content']

	return berita

def get_news_public(news_url: str) -> pd.DataFrame:
	"""
	Mengambil informasi dari isi berita selain CNN ID yang ada pada url
	"""
	news = []
	result = scrape_news_public(get_html(news_url))
	news.append(result)

	df = pd.DataFrame.from_dict(news)

	return df


def stemming_indo(text: str) -> str:
	"""
	Menstemming kata atau lemmisasi kata dalam bahasa Indonesia
	"""
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	text = ' '.join(stemmer.stem(word) for word in text)
	return text

def clean_stopword(tokens: list) -> list:
	"""
	Membersihkan kata yang merupakan stopword
	"""
	listStopword =  set(stopwords.words('indonesian'))
	removed = []
	for t in tokens:
		if t not in listStopword:
			removed.append(t)
	return removed

def preprocess_text(content):
	"""
	Memproses text berita, membersihkan text, memperbagus kata, dan menghilangkan stopword
	"""
	result = []
	for text in content:
		tokens = nltk.tokenize.word_tokenize(text)
		cleaned_stopword = clean_stopword(tokens)
		stemmed_text = stemming_indo(cleaned_stopword)
		result.append(stemmed_text)
	return result



def model_tf_idf(data, _model):
	"""
	Membuat model TF-IDF dari data
	"""
	tfidf_matrix = _model.transform(data)
	feature_names = _model.get_feature_names_out()
	
	df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

	return df_tfidf


def ringkas_berita(url: str, ctrl: str) -> str:
	"""
	Mengambil ringkasan berita dari url
	"""
	df = get_news_public(url)
	link = df['url'][0]
	judul = df['judul'][0]
	preprocessed = preprocess_text_ringkas(df['isi'][0])
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed)
	
	cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
	G = network_graph(cosine_sim)
	closeness_centrality = centrality(G, ctrl)
	
	result = sorted_result(closeness_centrality, preprocessed)

	return result, judul, link

def ringkas_text(text: str, ctrl: str) -> str:
	"""
	Mengambil ringkasan dari text
	"""
	preprocessed = preprocess_text_ringkas(text)
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed)
	
	cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
	G = network_graph(cosine_sim)
	closeness_centrality = centrality(G, ctrl)
	
	result = sorted_result(closeness_centrality, preprocessed)

	return result