from FolderViewer import FolderViewer
from DocReader import DocReader
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from pymystem3 import Mystem
from source_help.inverted_index import inverted_index
from tqdm import tqdm
from math import log
import pickle


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

    def _get_pickle(self, folder_name):
        files = self.viewer.get_files(self.path + folder_name)
        # todo do it for all files in folders
        cur_file = files[0]

        # temp_list = [url_num, title, body]
        # tuple_list.append(temp_list)
        full_path = self.path + folder_name + "/" + cur_file
        temp_list = self.reader.pickle_load(full_path)
        return temp_list

    def literal(self, o):
        s = self.escape(o, self.encoders)
        if isinstance(s, bytes):
            return s.decode('utf8', 'surrogateescape')
        return s

    def  _prepare(self):
        folders = self.viewer.get_folder_list(self.path)
        #todo do it for all folders
        cur_folder = folders[0]
        temp_list = self._get_pickle(cur_folder)

        body_index = inverted_index.Index()
        title_index = inverted_index.Index()
        for data in tqdm(temp_list):
            #[url_num, title, body]
            noise_free_title = list(set([stem.lemmatize(word.encode('utf-8', 'replace'))[0]
                                         for word in data[1].split() if word not in self.stop_words]))
            noise_free_body = list(set([stem.lemmatize(word.encode('utf-8', 'replace'))[0]
                                        for word in data[2].split() if word not in self.stop_words]))

            noise_free_title = " ".join(str(x) for x in noise_free_title)
            noise_free_body = " ".join(str(x) for x in noise_free_body)


            title_index.index(data[0], noise_free_title)
            body_index.index(data[0], noise_free_body)
        #todo title and body would be with different weights
        #results, err = body_index.query("бесплатная")
        #print(results)
        #results, err = title_index.query("бесплатная")
        #print(results)

        query_folder = "./data/queries.numerate.txt"
        queries_str = open(query_folder, "r").read()

        with open("./data/index/index.pkl", 'wb') as f:
            pickle.dump(list(title_index), f)

        return queries_str, title_index, body_index

    def score_BM25(self, n, qf, N, dl, avdl):
        K = self.compute_K(dl, avdl)
        IDF = log((N - n + 0.5) / (n + 0.5))
        frac = ((self.k1 + 1) * qf) / (K + qf)
        return IDF * frac

    def compute_K(self, dl, avdl):
        return self.k1 * ((1 - self.b) + self.b * (float(dl) / float(avdl)))


    def check_in_index(self, word, index):
        """
        :param word:
        :param index:
        :return: number of documents
        """
        qs = stem.lemmatize(word.encode('utf-8', 'replace'))[0]
        results, err = index.query(qs)
        return len(results)

    def check_total_docs(self, word, index):
        """
        :param word:
        :param index:
        :return: total meeting number of word
        """
        qs = stem.lemmatize(word.encode('utf-8', 'replace'))[0]
        return sum(index.inverted_index[qs].values())



    def compute_idf(self):

        N = 38114 #show me submission
        '''
        N - count of all documents easy
        run for those queries if in submission and len(results) !=0
        n(q_i) - number of documents that include q_i
        f(q_i, d_i) -  frequency of query
        d_length - length of document
        avgdl - average length of document
        :return:
        '''
        queries_str, title_index, body_index = self._prepare()
        queries_list = queries_str.splitlines()
        queries_list = [[int(query[0]), query[2:]]  for query in  queries_list]
        sample_sub = open("./data/sample.submission.text.relevance.spring.2018.csv", "r").read()

        actual_sub = sample_sub.splitlines()[1:-1]#1 is header, -1, because +1
        #actual_sub[0],actual_sub[1]  is QueryId,DocumentId

        ni_index = dict()#number of documents that include q_i corresponding to title (index?)
        ni_body = dict()#number of documents that include q_i
        fqd =dict()#frequency(q_i, d_i)
        ld = dict()#length of ith doc
        running =0

        for query in queries_list:
            # query[0] - number, query[1] - text
            number_docs_index = 0
            number_docs_body = 0
            #1. calculate n(q_i)
            for word in query[1].split():
                number_docs_index += self.check_in_index(word, title_index)
                number_docs_body += self.check_in_index(word, body_index)
            ni_index[query[0]] = number_docs_index
            ni_body[query[0]] = number_docs_body

            #2. calculate f(q_i, d_i)
            while(actual_sub[running][2] == actual_sub[running+1][2]):#2, because tab
                running += 1
                if running > len(actual_sub):
                    break
                #todo glue together pieces of files (pickle dumps in different folders)
                for word in query[1].split():
                    fqd[query[0], running] += self.check_total_docs(word, body_index)
                    ld[runnig]














    def search(self):
        queries_str, title_index, body_index = self._prepare()
        queries_list = queries_str.splitlines()
        for query in queries_list:
            number, query_txt = query.split("\t")
            number = int(number)
            query_words = query_txt.split()
            for word in query_words:
                if word not in self.stop_words:
                    word = word.lower()
                    q = stem.lemmatize(word)[0]
                    results, err = title_index.query(q)
                    if len(results) !=0:
                        docarray = results
                    else:
                        docarray = {}
                    n = len(results)
                    for doc in docarray:
                        #qf = query frequency
                        qf = 1







        print("here")



def unit_test():
    algo = bmw25()
    algo.compute_idf()

if __name__ == "__main__":
    unit_test()

    