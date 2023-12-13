from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import StandardAnalyzer
from whoosh import index
from whoosh.index import create_in
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, QueryParser
import os.path
import re
import timeit
import matplotlib.pyplot as plt

schema = Schema(index=ID(stored=True),
                title=TEXT(stored=True),
                author=TEXT(stored=True),
                bibliography=TEXT(stored=True),
                body=TEXT(analyzer=StandardAnalyzer(stoplist=None)))

INDEX_DIRECTORY = "index_dir"
if not os.path.exists(INDEX_DIRECTORY):
    os.mkdir(INDEX_DIRECTORY)
ix = create_in(INDEX_DIRECTORY, schema)
ix = index.open_dir(INDEX_DIRECTORY)


class Text:

    def __init__(self, original):
        result = re.split(r'.T|.A|.B|.W', original.replace('\n', ' '))
        self.index, self.title, self.author, self.bibliography, self.body, *_ = result


class Query:

    def __init__(self, text):
        result = text.split('\n.W\n')
        self.index, self.body = map(lambda x: x.strip().replace('\n', ' '),
                                    result)


def parse_queries(filename):
    queries = []
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.split('.I')[1:]
        queries = list(map(lambda x: Query(x), txt))
    return queries


def parse_text(filename):
    words = []
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.split('.I')[1:]
        words = list(map(lambda x: Text(x), txt))
    return words


def get_ordered_relevant_searches(filename):
    query_relations = {}
    with open(filename, 'r') as file:
        txt = file.read()
        txt = txt.strip().split('\n')
        for i in txt:
            query, abstract, score = map(
                lambda x: int(x),
                filter(lambda x: len(x) > 0,
                       i.strip().split(' ')))
            if query - 1 not in query_relations:
                query_relations[query - 1] = [(abstract, score)]
            else:
                query_relations[query - 1].append((abstract, score))

    #ordenando as relations por rank
    for i in query_relations:
        query_relations[i].sort(key=lambda x: x[1])

    return query_relations


def search_results(parser, queries: list[Query], limits: list[int]):
    results_dict = {}
    with ix.searcher() as searcher:
        for i, (limit, query_to_parse) in enumerate(zip(limits, queries)):
            query = parser.parse(query_to_parse.body)
            results = searcher.search(query, limit=max(limit, 10))
            results_dict[i] = list(
                map(lambda x: (int(x.get('index')), x.score), results))
    return results_dict


def precision_at_k(search, relevant, k=None):
    if k is None or k > len(search) or k > len(relevant):
        k = min(len(search), len(relevant))

    search = search[:k]
    relevant = relevant[:k]

    search_indexes = set(map(lambda x: x[0], search))
    relevant_indexes = set(map(lambda x: x[0], relevant))

    den = len(relevant_indexes)
    num = len(relevant_indexes.intersection(search_indexes))

    if den != 0:
        return num / den

    return None


def recall_at_k(search, relevant, k=None):
    if k is None or k > len(search) or k > len(relevant):
        k = min(len(search), len(relevant))

    search = search[:k]
    relevant = relevant[:k]

    search_indexes = set(map(lambda x: x[0], search))
    relevant_indexes = set(map(lambda x: x[0], relevant))

    den = len(relevant)
    num = len(relevant_indexes.intersection(search_indexes))

    if den != 0:
        return num / den

    return None


def all_results_by_func(search_results_dict,
                        revelant_results_dict,
                        func,
                        k=None):
    results = list(
        map(
            lambda x: func(search_results_dict[x], revelant_results_dict[x], k
                           ), range(len(revelant_results_dict))))
    return results


def plot_results(search_results_dict,
                 revelant_results_dict,
                 func,
                 title: str = '',
                 k_s=[1, 5, 10, None]):
    fig, axis = plt.subplots(len(k_s))
    fig.suptitle(title)
    for i, k in enumerate(k_s):
        r = all_results_by_func(search_results_dict,
                                revelant_results_dict,
                                func,
                                k=k)
        axis[i].plot(range(len(r)), r)
        k_name = f'{k}' if k is not None else 'MAX'
        axis[i].set_title(f'{func.__name__}={k_name}')
    plt.subplots_adjust(hspace=0.8)
    plt.show()


#OBTENDO PALAVRAS
queries = parse_queries('cran/cran.qry')

#OBTENDO QUERIES
words = parse_text('cran/cran.all.1400')

#OBTENDO BUSCAS RELEVANTES
query_relations = get_ordered_relevant_searches('cran/cranqrel')

#INDEXANDO RESULTADOS
t0 = timeit.default_timer()
writer = ix.writer()
error = False

for word in words:
    try:
        writer.add_document(index=f'{word.index}',
                            title=word.title,
                            author=word.author,
                            bibliography=word.bibliography,
                            body=word.body)
    except ValueError:
        error = True
        break

if error:
    writer.cancel()
else:
    writer.commit()

t1 = timeit.default_timer()
print(f'TEMPO DE INDEXAÇÃO {(t1 - t0):.2f}s')

limits = list(map(lambda x: len(x), query_relations.values()))

parser = MultifieldParser(fieldnames=["title", "author", "body"],
                          schema=schema,
                          group=OrGroup)
parser.add_plugin(FuzzyTermPlugin())

#BUSCA 1: TITULO, AUTOR E CORPO
t0 = timeit.default_timer()
results_dict = search_results(parser, queries, limits)
t1 = timeit.default_timer()
print(f'TEMPO DA BUSCA 1 {(t1 - t0):.2f}s')

k_s = [1, 5, 10, None]

#CALCULANDO TODAS AS PRECISIONS PARA BUSCA 1
plot_results(results_dict, query_relations, precision_at_k,
             'Cálculo das precisions para busca 1', k_s)

#CALCULANDO TODAS OS RECALLS PARA BUSCA 1
plot_results(results_dict, query_relations, recall_at_k,
             'Cálculo dos recalls para busca 1', k_s)

parser_query = QueryParser("body", schema=schema, group=OrGroup)
parser_query.add_plugin(FuzzyTermPlugin())

#BUSCA 2: SOMENTE CORPO
t0 = timeit.default_timer()
results_dict_qp = search_results(parser_query, queries, limits)
t1 = timeit.default_timer()
print(f'TEMPO DA BUSCA 2 {(t1 - t0):.2f}s')

#CALCULANDO TODAS AS PRECISIONS PARA BUSCA 2
plot_results(results_dict_qp, query_relations, precision_at_k,
             'Cálculo das precisions para busca 2', k_s)

#CALCULANDO TODAS OS RECALLS PARA BUSCA 2
plot_results(results_dict_qp, query_relations, recall_at_k,
             'Cálculo dos recalls para busca 2', k_s)

ix.close()
