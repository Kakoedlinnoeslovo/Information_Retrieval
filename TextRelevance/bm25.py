from FolderViewer import FolderViewer
from DocReader import DocReader
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from pymystem3 import Mystem
from source_help.inverted_index import inverted_index
from tqdm import tqdm
from math import log
import pickle
from collections import defaultdict
import numpy as np
from multiprocessing import Pool


stem = Mystem()


class bmw25:
    def __init__(self):
        self.path = "./data/pickle_dumps/"
        self.viewer = FolderViewer()
        self.reader = DocReader()
        self.stop_words = stopwords.words('russian') + ['и', 'к','за', 'а',
                                                        'но', 'в', 'на',
                                                        'под']

        self.k1 = 2.0
        self.b = 0.75
        self.N = 38114 #that show me submission
        self.avdl_body = 0
        self.avdl_title = 0
        #self.body_index = inverted_index.Index()
        #self.title_index = inverted_index.Index()
        self.body_len = dict()
        self.title_len = dict()
        self.safe_body = dict()
        self.safe_title = dict()

    def _get_pickle(self, folder_name):
        files = self.viewer.get_files(self.path + folder_name)
        # todo do it for all files in folders
        cur_file = files[0]

        # temp_list = [url_num, title, body]
        # tuple_list.append(temp_list)
        full_path = self.path + folder_name + "/" + cur_file
        temp_list = self.reader.pickle_load(full_path)
        return temp_list


    def get_noise_free(self, data):
        noise_free = list(set([stem.lemmatize(word.encode('utf-8', 'replace'))[0]
                                     for word in data.split() if word not in self.stop_words]))
        return noise_free



    def fill_title_index(self, data):
        """
        :param data: old data[1] or data[2] - title or body
        :param inverted_index, that i want to fill
        :param index: data[0]
        :return: fill index body or title
        """
        noise_free_title = self.get_noise_free(data[1])
        self.title_len[data[0]] = len(noise_free_title)
        self.avdl_title += len(noise_free_title)
        noise_free_title = " ".join(str(x) for x in noise_free_title)
        the_file = open('./data/clear/title.txt', 'a')
        the_file.write('{}\t{}\n'.format(data[0], noise_free_title))
        #self.safe_title[data[0]] = noise_free_title
        #self.title_index.index(data[0], noise_free_title)

    def fill_body_index(self, data):
        noise_free_body = self.get_noise_free(data[2])
        self.body_len[data[0]] = len(noise_free_body)
        self.avdl_body += len(noise_free_body)
        noise_free_body = " ".join(str(x) for x in noise_free_body)
        the_file = open('./data/clear/body.txt', 'a')
        the_file.write('{}\t{}\n'.format(data[0], noise_free_body))
        #self.safe_body[data[0]] = noise_free_body
        #self.body_index.index(data[0], noise_free_body)


    def prepare_list(self, temp_list):
        self.fill_title_index(temp_list)
        self.fill_body_index(temp_list)


    def  _prepare(self, cur_folder):
        """
        :param cur_folder: get cur_folder,
        it is relative path of folder, example: 20170702

        this function:
        clear html,
        compute len,average length,
        build inverted index,
        open folder with queries and return it

        :return: queries_str, title_index, body_index, title_len, body_len
        """
        temp_list = self._get_pickle(cur_folder)

        n_pools = 10
        batch_size = 500
        for i in tqdm(range(0, len(temp_list)-1, batch_size)):
            with Pool(n_pools) as p:
                p.map(self.prepare_list, temp_list[i*batch_size:(i+1) * batch_size])



    def check_in_index(self, word, index):
        """
        :param word:
        :param index:
        :return: number of documents
        """
        qs = stem.lemmatize(word.encode('utf-8', 'replace'))[0]
        results, err = index.query(qs)
        return len(results)


    def compute_idf(self, N, n_qi):
        idf = np.log((N - n_qi + 0.5)/(n_qi + 0.5))
        return idf


    def compute_bm25(self, idf, freq, len_j, avdl):
        k1 = 2.0
        b = 0.75

        drob = len_j/avdl
        bm25 = (freq * (k1 +1))/ (freq + k1 * (1 - b + b * drob))
        bm25 = idf * bm25
        return bm25


    def compute_score(self, query, doc_number, title_index, body_index):
        """

        :param query: one query, one document_id
        :param body_index:
        :param title_index:
        :return: compute score for one query
        """
        score_title = defaultdict()
        score_body = defaultdict()

        #it will be weights for title and body: for example: 0.8 and 0.2
        bm25_score_title = 0
        bm25_score_body = 0
        for word in query[1].split():
            word_stem = stem.lemmatize(word.encode('utf-8', 'replace'))[0]

            n_i  = [len(title_index.inverted_index[word_stem].keys()),
                    len(body_index.inverted_index[word_stem].keys())]

            idf = [self.compute_idf(self.N, n_i[0]),
                    self.compute_idf(self.N, n_i[1])]

            # j = doc_number
            freq_i_j = [title_index.inverted_index[word_stem][doc_number],
                        title_index.inverted_index[word_stem][doc_number]]

            #length of j doc
            len_j = [self.title_len[doc_number], self.body_len[doc_number]]

            bm25_score_title+=self.compute_bm25(idf[0], freq_i_j[0], len_j[0], self.avdl_title)
            bm25_score_body+=self.compute_bm25(idf[1], freq_i_j[1], len_j[1], self.avdl_body)

        return 0.8 * bm25_score_title + 0.2 * bm25_score_body


    def get_query_docs(self, query, title_index, body_index):
        """
        :param query: is a text!
        :param body_index:
        :param title_index:
        :return: all docs that include this query (title and body)
        """
        doc_list = list()
        for word in query.split():
            word_stem = stem.lemmatize(word.encode('utf-8', 'replace'))[0]
            doc_s_index = list(title_index.inverted_index[word_stem].keys())
            doc_s_body = list(body_index.inverted_index[word_stem].keys())
            doc_list = list(set(doc_s_index.extend(doc_s_body)))
        return doc_list


    def run(self):
        folders = self.viewer.get_folder_list(self.path)

        score_bm25 = defaultdict()  # [0] is query, [1] is doc, return score
        #for cur_folder in folders:
        cur_folder = folders[0]
        self._prepare(cur_folder)



        # read file
        file_title = "./data/clear/title.txt"
        file_body = "./data/clear/title.txt"

        body_index = inverted_index.Index()
        title_index = inverted_index.Index()

        file_title = open(file_title, "r").read()
        title_list = file_title.splitlines()

        file_body = open(file_body, "r").read()
        body_list = file_body.splitlines()

        for data in title_list:
            data = data.split('\t')
            if len(data) == 2:
                title_index.index(int(data[0]), data[1])
            else:
                continue

        for data in body_list:
            data = data.split('\t')
            if len(data) ==2:
                body_index.index(int(data[0]), data[1])
            else:
                continue


        self.avdl_body /= self.N
        self.avdl_title /= self.N

        query_folder = "./data/queries.numerate.txt"
        queries_str = open(query_folder, "r").read()
        queries_list = queries_str.splitlines()
        queries_list = [[int(query[0]), query[2:]] for query in queries_list]


        for query in queries_list:
            # query[0] - number, query[1] - text
            #all docs for this query:
            doc_s = self.get_query_docs(query[1], title_index, body_index)
            for doc_number in doc_s:
                bm25 = self.compute_score(query[1], doc_number, title_index, body_index)
                score_bm25[query[0]][doc_number] = bm25

        return score_bm25



def unit_test():
    algo = bmw25()
    algo.run()

if __name__ == "__main__":
    unit_test()

