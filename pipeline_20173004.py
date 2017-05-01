#!/usr/bin/env python3

from __future__ import division
import csv
import os
import re
from math import ceil  
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nxmetis import partition
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from operator import itemgetter 

__author__       = 'Johanna Korte'
__author_email__ = '<johanna.korte@student.auc.nl>'

lod = ['Feb 20', 'Feb 21', 'Feb 22', 'Feb 23', 'Feb 24', 'Feb 25',
       'Feb 26', 'Feb 27', 'Feb 28', 'Mar 01', 'Mar 02', 'Mar 03', 'Mar 04',
       'Mar 05', 'Mar 06', 'Mar 07', 'Mar 08', 'Mar 09', 'Mar 10', 'Mar 11',
       'Mar 12', 'Mar 13', 'Mar 14', 'Mar 15', 'Mar 16', 'Mar 17', 'Mar 18']

weeks = [['Feb 20','Feb 21', 'Feb 22', 'Feb 23', 'Feb 24', 'Feb 25', 'Feb 26'],
         ['Feb 27', 'Feb 28', 'Mar 01', 'Mar 02', 'Mar 03', 'Mar 05', 'Mar 06'],
         ['Mar 07', 'Mar 08', 'Mar 09', 'Mar 10', 'Mar 11', 'Mar 12', 'Mar 13'],
         ['Mar 14', 'Mar 15', 'Mar 16', 'Mar 17', 'Mar 18']]

def pipeline(data, dates, output_graph, plot):
    """
    data = filename.csv
    dates = list of dates, or 'all'
    output_graph = True or False (want output edgelist and nodelist or not)
    plot = True or False (plots convergiance of RWC score)
    """
    graph_dict = make_graph_dict(data, dates)
    if graph_dict == 'NaN':
        print 'nan'
        rwc = graph_dict
    else:
        G = build_graph(graph_dict)
        G_removed_outliers = exclude_nodes_GC(G)
        #G_removed_outliers2 = exclude_nodes_degree(G_removed_outliers, 2)
        partitioned_graph = metis_partition(G_removed_outliers)
        rwc = random_walk(partitioned_graph,0.05,0.001, plot)
        
        if output_graph == True:
            output(partitioned_graph)
        if rwc == 'NaN':
            print 'nan'
    return rwc
    

def make_graph_dict(data, dates):
    """Takes a csv filename containing tweets (rows are 
    id_str,from_user,text,created_at) and dates. Outputs the corresponding
    dictionary with as keys the tweeters and as values who they retweeted.
    """
    with open(data, 'rb') as tweet_file:
        graph_dict = {}
        RTpattern = re.compile("RT @\w+")
        tweet_reader = csv.reader(tweet_file, delimiter=',')
        for row in tweet_reader:
            # Extract retweets
            retweet = RTpattern.match(row[2])
            # Extract correct dates 
            if dates == 'all':
                date_match = True
            else: 
                for date in dates:
                    date_match = re.compile('.*' + date + '.*').match(row[3])
                    if date_match != None:
                        break
                
            if (retweet != None) & (date_match != None):
                retweeter = '@' + row[1]
                tweeter = retweet.group()[3:]

                #build graph
                if retweeter in graph_dict.keys():
                    graph_dict[retweeter] = graph_dict[retweeter] + [tweeter]
                else:
                    graph_dict[retweeter] = [tweeter]
        if graph_dict == {}:
            return 'NaN'
        else:
            return graph_dict


def build_graph(graph_dict):
    """ Takes a dictionary containing tweeters as keys and the handles they 
    retweeted as values (as returned by make_graph_dict) and returns a
    networkX graph
    """ 
    #make networkX graph
    G = nx.Graph()
    G.add_nodes_from(graph_dict.keys())
    for key in graph_dict:
        for i in range(len(graph_dict[key])):
            G.add_edge(key,graph_dict[key][i])
    return G


def exclude_nodes_degree(G, min_degree):
    """ Takes a graph and a parameter min_degree, and returns a new graph with  all nodes from
    G that have a degree smaller than min_degree removed. 
    """
    remove = [node for (node, degree) in G.degree().items() if degree < min_degree]
    G.remove_nodes_from(remove)
    #remove new nodes without edges
    remove_zero_degree = [node for (node, degree) in G.degree().items() if degree == 0]
    G.remove_nodes_from(remove_zero_degree)
    return G


def exclude_nodes_GC(G):
    """ Returns a graph with  the nodes that are not part of the giant component
    removed"""
    remove, present = [], []
    # Find giant component
    Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    G0 = Gcc[0]
    for node in G.nodes():
        if node not in G0.nodes():
            remove.append(node)
            G0.add_node(node,GC= 0)
        else:
            present.append(node)
            G0.add_node(node, GC= 1)
    # Remove nodes not in giant component
    remove_outliers = [node for node in G.nodes() if node not in G0.nodes()]
    G.remove_nodes_from(remove_outliers)
    return G


def metis_partition(G):
    """ Takes a networkX graph and the number of clusters to be formed,
    and partitions the graph in that number of clusters using the
    METIS algorithm. Returns the graph with added cluster attributes.  
    """
    partition_list = partition(G, 2)[1]
    for i in range(2):
        for username in partition_list[i]:
            G.add_node(username, cluster=i)
            
    return G


def random_walk(G, k, convergence_threshold, plot):
    """ Takes a graph G, the amount of random edge selections (walks) and how many times a new random node should
    be initialized (iterations), and returns a controversy measure based on the proportion of times a random walk 
    initialized in one cluster, ends up in the other. If plot is True, a plot of the RWC value over time will be 
    shown. 
    """
    #SET: minimum amount of iterations
    min_iterations = 1000
    #SET: minimum amount of walks
    min_walks = 10
    #SET: maximum amount of walks
    max_walks = 5000 

    double_zero, double_one, zero_one, one_zero = 0, 0, 0, 0
    RWC_list = []
    convergence = 10000
    i = 0

    nodes0 = [node for node in G.nodes(data=True) if node[1]['cluster'] == 0]
    nodes1 = [node for node in G.nodes(data=True) if node[1]['cluster'] == 1]

    if nodes0 == [] or nodes1 == []:
        return 'NaN'
    
    degrees0 = sorted([(node[0], G.degree(node[0])) for node in nodes0], key=itemgetter(1), reverse=True)
    degrees1 = sorted([(node[0], G.degree(node[0])) for node in nodes1], key=itemgetter(1), reverse=True)

    k_tuples= degrees0[:int(ceil(k*len(nodes0)))] + degrees1[:int(ceil(k*len(nodes1)))]
    k_nodes= [node for (node, degree) in k_tuples]
    
    while convergence > convergence_threshold or i < min_iterations:
        # choose random cluster (choose random between 0,1), prob is 0.5
        begin_cluster = random.choice([0, 1])

        # choose random node in cluser
        if begin_cluster == 0:
            current_node = random.choice(nodes0)
        else:
            current_node = random.choice(nodes1)

        # choose random edge from cluster (repeat)
        current_node = current_node[0]

        j = 0
        while j < max_walks:
            previous_node = current_node
            current_node = random.choice(G.neighbors(current_node))
            #prevent self_loops
            if previous_node == current_node:
                current_node = previous_node
                j+=1
                continue
            #print('{}'.format(current_node))
            if current_node in k_nodes:
                if j < min_walks:
                    continue
                else:
                    break
            j += 1 

        # what cluster end node
        end_cluster = G.node[current_node]['cluster']

        #Keep tally of outcomes
        if begin_cluster == 0:
            if end_cluster == 0:
                double_zero += 1
            else:
                zero_one += 1
        else:
            if end_cluster == 0:
                one_zero += 1
            else:
                double_one += 1

        #calculate conditional probabilities
        total = double_zero + double_one + zero_one + one_zero

        prob00 = (double_zero/total) / 0.5
        prob11 = (double_one/total) / 0.5
        prob10 = (one_zero/total)/ 0.5
        prob01 = (zero_one/total)/ 0.5

        rwc = prob00*prob11 - prob10*prob01

        #update convergence 
        if RWC_list != []:
            convergence = abs(rwc - RWC_list[-1])
        
        i += 1 
        RWC_list.append(rwc)

    # Plot RWC scores over time 
    if plot == True:
        plt.plot(RWC_list)
        plt.show()

    return(rwc)
    

def output(G):
    """Generates two .tsv files: edges.tsv and nodes.tsv. Containing the edge
    and node list of G respectively 
    """
    with open('/Users/johannakorte/Desktop/edges_no_outliers.tsv', 'wb') as edge_list:
        writer = csv.writer(edge_list, delimiter = '\t')
        # write header row
        writer.writerow(['Source', 'Target'])
        for edge in G.edges():
            writer.writerow([edge[0], edge[1]])
                    
    with open('/Users/johannakorte/Desktop/nodes_no_outliers.tsv', 'wb') as node_list:
        writer = csv.writer(node_list, delimiter = '\t')
        writer.writerow(['Id', 'Cluster'])
        # Write nodes and respective cluster number
        for node in G.nodes(data=True):
            writer.writerow([node[0], node[1]['cluster']])
    return


#datasource = '/Users/johannakorte/Desktop/Results/20171903/clean_2017_03_19/'
#output= '/Users/johannakorte/Desktop/20171903_weekly.csv'

def RWC_table(datasource, output, time):
    """ Takes a folder containing cleaned csv file with tweets, an output filename and the 
    time interval to be used ('day' or 'week'), returns a file with the RWC scores
    per time interval per hashtag
    """
    with open(output + '_' + time + '.csv', 'wb') as destination:
        writer = csv.writer(destination)
        if time == 'day':
            writer.writerow(['file'] + lod)
        elif time == 'week':
            writer.writerow(['file'] + weeks)
        
        for file in os.listdir(datasource):
            rowlist = []
            if file.endswith(".csv"):
                rowlist.append(file)
                print(file)

                # Check timeinterval and run pipeline with those settings
                if time == 'day':
                    for day in tqdm(lod):
                        rowlist.append(pipeline(datasource + file, [day], False, False))
                    writer.writerow(rowlist)

                if time == 'week':
                    for week in tqdm(weeks):
                        rowlist.append(pipeline(datasource + file, week, False, False))
                    writer.writerow(rowlist)

#RWC_table('/Users/johannakorte/Desktop/Results/20173004/Party/', '/Users/johannakorte/Desktop/Results/20173004/party', 'day')
#RWC_table('/Users/johannakorte/Desktop/Results/20173004/Country/', '/Users/johannakorte/Desktop/Results/20173004/country', 'day')
RWC_table('/Users/johannakorte/Desktop/Results/20173004/Individual/', '/Users/johannakorte/Desktop/Results/20173004/individual', 'day')
