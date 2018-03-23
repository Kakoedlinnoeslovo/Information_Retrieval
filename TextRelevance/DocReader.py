# -*-coding: utf-8 -*-
from bs4 import BeautifulSoup
import os
from os import listdir
from os.path import isfile, join


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

	def parser(self):
		folder_list = self.fv.get_folder_list(self.path)
		#todo fixed first folder
		folder = folder_list[0]
		files = self.fv.get_files(self.path +folder)
		for file in files:
			page = open(self.path + folder + '/' + file, 'r')
			page =page.decode('utf-8')
			soup = BeautifulSoup(page, 'html.parser')
			name_box = soup.find('h1')
			print(name_box)






def main():
	doc = DocReader()
	doc.parser()

if __name__ =="__main__":
	main()