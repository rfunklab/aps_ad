import sys
import itertools
import pickle
import time
import argparse
import pandas as pd

import h5py
import numpy as np
import sklearn
import mysql.connector
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import lil_matrix
import db_config

alpha_A = .54
alpha_S = .75
alpha_R = .19
alpha_C = 1.02

beta_1 = 0.1
beta_2 = .2
beta_3 = 0.1

OUT_LOC = '/home/gebhart/projects/rfunklab/aps_ad/models'


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

mydb = mysql.connector.connect(
        host=db_config.DB_CONFIG['host'],
        user=db_config.DB_CONFIG['user'],
        passwd=db_config.DB_CONFIG['passwd'])
mycursor = mydb.cursor()

def sa(A,i,j): return (len(set(A[i]).intersection(A[j])))/(min(len(A[i]),len(A[j])))# remove the author that tie these two papers together
def sr(R,i,j): return len(set(R[i]).intersection(R[j]))/min(len(R[i]),len(R[j]))
def sc(C,i,j): return len(set(C[i]).intersection(C[j]))/min(len(C[i]),len(C[j]))
def sx(R,i,j): return int(i in R[j])+int(j in R[i])

# flush print
def flushPrint(d,n):
    sys.stdout.write('\r')
    sys.stdout.write('{:0g}/{:0g}'.format(d,n))
    sys.stdout.flush()

def run(load_sims=False, out_loc=OUT_LOC):

    if not load_sims:

        # train on patentsview
        print('Querying Author Map')
        mycursor.execute("USE aps")

        mycursor.execute("SELECT COUNT(record_id) FROM authors_raw")
        res = mycursor.fetchall()
        rows = res[0][0]
        mycursor.execute("SELECT record_id, firstname, surname, name FROM authors_raw")

        A_train = {}
        A_papers = {}
        i = 0
        Xs = mycursor.fetchall()
        for x in Xs:
            if i % 10000 == 0:
                flushPrint(i/10000, rows//10000)
            if x[2] is not None:
                surname = str(x[2].decode('utf-8'))
                if x[1] is not None:
                    firstname = str(x[1].decode('utf-8'))
                    if len(firstname) > 0:
                        fi = firstname[0].lower()
                    else:
                        fi = ''
                else:
                    fi = ''
                author = Author(fi + ' ' + surname.lower(), i, firstname=firstname, surname=surname)
            else:
                try:
                    name = str(x[3].decode('utf-8'))
                    author = Author(name.lower(), i)
                except AttributeError:
                    print(x[1], x[2], x[3])
                    continue
            publication = str(x[0].decode('utf-8'))
            if publication in A_papers:
                A_papers[publication].append(author)
            else:
                A_papers[publication] = [author]
            if author in A_train:
                A_train[author].append(publication)
            else:
                A_train[author] = [publication]
            i += 1

        print('\n Querying Reference/Citation Map')

        mycursor.execute('SELECT COUNT(citing_doi) FROM citations_raw')
        res = mycursor.fetchall()
        rows = res[0][0]

        q = 'SELECT citing_doi, cited_doi FROM citations_raw'
        mycursor.execute(q)

        C_train = {}
        R_train = {}
        Xs = mycursor.fetchall()
        i = 0
        for x in Xs:
            if i % 10000 == 0:
                flushPrint(i/10000,rows//10000)
            try:
                citing = str(x[0].decode('utf-8'))
                cited = str(x[1].decode('utf-8'))
            except AttributeError:
                print('Continuing', i)

            if cited in C_train:
                C_train[cited].append(citing)
            else:
                C_train[cited] = [citing]
            if citing in R_train:
                R_train[citing].append(cited)
            else:
                R_train[citing] = [cited]
            i += 1

        with open(os.path.join(out_loc, 'A_train.pkl'), 'wb') as f:
            pickle.dump(A_train, f)
        with open(os.path.join(out_loc, 'A_papers.pkl'), 'wb') as f:
            pickle.dump(A_papers, f)
        with open(os.path.join(out_loc, 'R_train.pkl'), 'wb') as f:
            pickle.dump(R_train, f)
        with open(os.path.join(out_loc, 'C_train.pkl'), 'wb') as f:
            pickle.dump(C_train, f)

    else:

        with open(os.path.join(out_loc,'A_train.pkl'), 'rb') as f:
            A_train = pickle.load(f)
        with open(os.path.join(out_loc, 'A_papers.pkl'), 'rb') as f:
            A_papers = pickle.load(f)
        with open(os.path.join(out_loc, 'R_train.pkl'), 'rb') as f:
            R_train = pickle.load(f)
        with open(os.path.join(out_loc, 'C_train.pkl'), 'rb') as f:
            C_train = pickle.load(f)


    print('\n Beginning Similarity Calculations ... ')
    Sims = {}
    Labels={}
    Pixmaps = {}
    disambs = {}
    rows = len(A_train)
    n=0
    es = list(A_train.keys())
    for e in es:
        if n%10==0:
            flushPrint(n/10, rows//10) #868
        # merge all papers
        papers = A_train[e]
        if 2<=len(papers):
            sim = np.zeros((len(papers),len(papers)))
            # sim = lil_matrix((len(papers),len(papers)))
            pixmap = {papers[i]:i for i in range(len(papers))}
            rev_pixmap = {i:papers[i] for i in range(len(papers))}
            for i,j in itertools.combinations(papers, 2):
                try:
                    sas = sa(A_papers,i,j)
                except Exception as ex:
                    # print('Error in sas', ex)
                    sas = 0.0
                try:
                    srs = sr(R_train,i,j)
                except Exception as ex:
                    # print('Error in srs', ex)
                    srs = 0.0
                try:
                    scs = sc(C_train,i,j)
                except Exception as ex:
                    # print('Error in scs', ex)
                    scs = 0.0
                try:
                    sxs = sx(R_train,i,j)
                except Exception as ex:
                    # print('Error in sxs', ex)
                    sxs = 0.0
                try:
                    sim[pixmap[i],pixmap[j]] = alpha_A*sas + alpha_R*srs + alpha_C*scs + alpha_S*sxs
                    # sim[pixmap[j],pixmap[i]] = alpha_A*sas + alpha_R*srs + alpha_C*scs + alpha_S*sxs
                except Exception as ex:
                    print(ex)


        del A_train[e]
        n+=1

        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single', distance_threshold=beta_1)
        clustering2 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single', distance_threshold=beta_3)
        X = symmetrize(sim)
        labels = clustering.fit_predict(X)
        n_clusters = clustering.n_clusters_
        if n_clusters > 2:
            S_Sims = np.zeros(shape=(n_clusters,n_clusters))
            for c,d in itertools.combinations(list(range(n_clusters)),2):
                S = 0
                cidxs = np.argwhere(labels == c)
                didxs = np.argwhere(labels == d)
                for cidx in cidxs:
                    for didx in didxs:
                        i = rev_pixmap[cidx[0]]
                        j = rev_pixmap[didx[0]]
                        try:
                            sas = sa(A_papers,i,j)
                        except Exception as ex:
                            # print('Error in sas', ex)
                            sas = 0.0
                        try:
                            srs = sr(R_train,i,j)
                        except Exception as ex:
                            # print('Error in srs', ex)
                            srs = 0.0
                        try:
                            scs = sc(C_train,i,j)
                        except Exception as ex:
                            # print('Error in scs', ex)
                            scs = 0.0
                        try:
                            sxs = sx(R_train,i,j)
                        except Exception as ex:
                            # print('Error in sxs', ex)
                            sxs = 0.0
                        s = alpha_A*sas + alpha_R*srs + alpha_C*scs + alpha_S*sxs
                        if s > beta_2:
                            S += s/(cidxs.shape[0]*didxs.shape[0])
                            S_Sims[c,d] = S
                            S_Sims[d,c] = S
            labels2 = clustering2.fit_predict(S_Sims)
            n_clusters_2 = clustering2.n_clusters_
            t_disam = {}
            for i in range(n_clusters_2):
                u_paper_idxs = np.argwhere(labels2 == i)
                t_disam[i] = []
                for idx in u_paper_idxs:
                    cidxs = np.argwhere(labels == idx)
                    for pidx in cidxs:
                        t_disam[i].append(rev_pixmap[pidx[0]])
            disambs[e] = t_disam
        else:
            disambs[e] = {0:[rev_pixmap[i] for i in range(sim.shape[0])]}

    with open(os.path.join(out_loc, 'Disambiguated.pkl'), 'wb') as f:
        pickle.dump(disambs, f)

class Author(object):
    def __init__(self, name, table_id, firstname=None, surname=None):
        self.name = name
        self.id = table_id
        self.firstname = firstname
        self.surname = surname

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name + '_' + str(self.table_id)

    def __eq__(self, other):
        if isinstance(other, Author):
            return self.name == other.name
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aps author disambiguation')
    parser.add_argument('-o', '--output-location', type=str, required=False,
                        help='location to store the disambiguated model and related data',
                        default=OUT_LOC)
    parser.add_argument('-ls', '--load-sims', action='store_true',
                        help='whether to load saved metadata to compute disambiguation')

    args = parser.parse_args()
    run(load_sims=args.load_sims, out_loc=args.output_location)


# end
