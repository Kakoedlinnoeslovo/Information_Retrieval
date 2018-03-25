# -*-coding: utf-8 -*-
from bs4 import BeautifulSoup
import os
from os import listdir
from os.path import isfile, join
#from collections import namedtuple
import re
import codecs
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import json
import time

TAG_RE = re.compile(r'<[^>]+>')
DEL_SYM = re.compile(r'[!@#$\+\-\{\}\¦]')


def checkNone(func):
	def temp(object):
		if not isinstance(None, type(object)):
			return func(object.text)
		else:
			return func('')
	return temp


@checkNone
def remove_tags(text):
	#text = TAG_RE.sub('', text)
	#text = DEL_SYM.sub('', text)
	text = text.strip().lower()
	text = ' '.join(text.split())
	# text = re.sub(r"http://.+.html", '', text)
	# text = re.sub(r"http://.+.com/\w+\?\w+=\w+", '', text)
	# text = re.sub(r"http://.+.php.+", '',text)
	# text = re.sub(r"http://.+.ru.+", '', text)
	# text = re.sub(r"https://.+.com.+", '', text)
	text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', text)
	text = re.sub(r"[@*©®_#=«»/\"()№…\-{}|:—\[\]❶•]", '', text)
	time_reg = re.compile("(24:00|2[0-3]:[0-5][0-9]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9]:[0-5][0-9]"
						  "|[0-1][0-9]:[0-5][0-9]|2[0-3]:[0-5][0-9])")
	text = re.sub(time_reg, '', text)
	return text

def remove_div(soup):
	# replace with `soup.findAll` if you are using BeautifulSoup3
	for div in soup.find_all("div", {'class': 'post'}):
		div.decompose()
	return soup


def remove_hr(soup):
	for hr in soup("hr"):
		hr.decompose()
	return soup


def remove_script(soup):
	for script in soup("script"):
		script.decompose()
	return soup


class FolderViewer:
	def __init__(self):
		pass

	def get_folder_list(self, path):
		return os.listdir(path)


	def get_files(self, path):
		onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)))]
		return onlyfiles


class DocReader:
	def __init__(self):
		self.fv = FolderViewer()
		self.path = "./data/content/"
		#self.Document = namedtuple('Document', ['url_number', 'title', 'body'])
		self.out = './data/pickle_dumps/'


	def _get_url(self, page):
		url = ''
		for char in page:
			if char == "\n":
				break
			url += char
		return url


	def _get_url_num(self, url, urls_number):
		url_num = None
		for url_num in urls_number.split('\n'):
			if url in url_num:
				url_num = url_num.split('\t')[0]
				break
		return url_num

	def pickle_save(self, file_list, path_str, folder_str):
		assert not isinstance(str, type(path_str))
		assert not isinstance(list, type(file_list))
		assert not isinstance(str, type(folder_str))

		full_path = path_str + folder_str
		if not os.path.exists(full_path):
			os.makedirs(full_path)

		with open(full_path + '/' + 'data.pkl', 'wb') as f:
			pickle.dump(file_list, f)


	def pickle_load(self, path):
		assert not isinstance(str, type(path))
		with open(path, 'rb') as f:
			list = pickle.load(f)
		return list

	def save_json(self, data):
		time_write = time.time()
		with open('./data/json/json_{}.txt'.format(time_write), 'w') as outfile:
			json.dump(data, outfile)


	def go_parse(self):
		folder_list = self.fv.get_folder_list(self.path)
		n_pools = len(folder_list)
		with Pool(n_pools) as p:
			p.map(self.parser, folder_list[0:2])


	def parser(self, folder = "20170702"):
		tuple_list = list()
		files = self.fv.get_files(self.path +folder)
		files = sorted(files)
		urls_number = open("./data/" + "urls.numerate.txt", 'r')
		urls_number = urls_number.read()


		for file in tqdm(files):
			page = codecs.open(self.path + folder + '/' + file, "r", encoding='utf-8', errors='ignore')
			page = page.read()
			url = self._get_url(page)
			url_num = int(self._get_url_num(url, urls_number))
			soup = BeautifulSoup(page, 'html.parser')
			#delete tag <script>
			soup = remove_script(soup)

			title = soup.find('title')
			body = soup.find('body')
			title = remove_tags(title)
			body = remove_tags(body)

			temp_list = [url_num, title, body]
			tuple_list.append(temp_list)

		#self.pickle_save(tuple_list, self.out, folder)
		self.save_json(tuple_list)


			#todo you can do some experiments with this
			#TITLE = remove_tags(TITLE)
			# article = remove_tags(article)

			#print(title + TITLE + '\n')
			#print(body + '\n')
			#print(page)






def main():
	doc = DocReader()
	doc.go_parse()

if __name__ =="__main__":
	main()