import os
import json
import math
import operator
import warnings
from statistics import mean
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import community.community_louvain as com
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, spectral_clustering, DBSCAN, MeanShift
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn import mixture

from settings import EMBEDDING_FOLDERS
# from xls_utils import write_stats_clusterings_xls, write_stats_datasets_xls


current_location = "/data/contrastive-mir-learning"
genre_ids_to_name = json.load(open(current_location+'/json/genre_gn_all.json', 'rb'))


def remove_failed_embeddings(datasets):
    """
    Check all embedding folders to list missing files.
    Remove them from the datasets.

    """
    all_ids = json.load(open('all_sound_ids.json', 'rb'))
    missing_embeddings = []

    # parse all embedding folders and see which files are missing
    for embedding_folder in EMBEDDING_FOLDERS.values():
        if embedding_folder:
            embeddings_files = os.listdir(embedding_folder)
            for s_id in all_ids:
                if '{0}.npy'.format(s_id) not in embeddings_files:
                    missing_embeddings.append(s_id)

    missing_embeddings = set(missing_embeddings)

    # remove sounds in the datasets
    for dataset in datasets:
        for s_id in missing_embeddings:
            try:
                dataset['sound_ids'].remove(s_id)
            except:
                pass
        for _, obj in dataset['dataset'].items():
            for s_id in missing_embeddings:
                try:
                    obj.remove(s_id)
                except:
                    pass


def create_label_vector(dataset):
    """
    Returns given dataset with some label fields.

    """
    labels = []
    label_ids = []
    label_names = []
    sound_ids = []
    for node_id, obj in dataset['dataset'].items():
        label_ids.append(node_id)
        label_names.append(ontology_by_id[node_id]['name'])
        sound_ids += obj
        for sound_id in obj:
            labels.append(label_ids.index(node_id))
    dataset['sound_ids'] = sound_ids      # order of the sounds
    dataset['labels'] = labels            # idx
    dataset['label_ids'] = label_ids      # audioset id
    dataset['label_names'] = label_names  # name
    return dataset


def load_features(dataset):
    """
    Returns given dataset with embedding features.

    """
    track_ids = dataset['track_ids']
    for embedding_name, embedding_folder in EMBEDDING_FOLDERS.items():
        if embedding_folder:
            embedding_files = [embedding_folder + '{0}.npy'.format(track_id)
                            for track_id in track_ids]

            features = [np.load(f) for f in embedding_files]
            if embedding_name == 'mfcc':
                X = features
            else:
                X = [np.mean(f, axis=0) for f in features]

            dataset['X_{}'.format(embedding_name)] = X

    return dataset


def compute_similarity_matrix(X):
    """
    Compute similarity matrix of the given features.

    """
    #euclidian_distances = euclidean_distances(X)
    #similarity_matrix = 1 - euclidian_distances/euclidian_distances.max()
    #similarity_matrix = np.exp(-1 * euclidian_distances / euclidian_distances.std())
    similarity_matrix = cosine_similarity(X)
    return similarity_matrix

def point_centralities(graph, communities):
    """Computes graph centrality of each node in the given communities (clusters) of the given graph.
    This may be useful for selecting representative examples of a cluster. A sound that is central in his cluster may be
    represent what a cluster contains the most.
    Args:
        graph (nx.Graph): NetworkX graph representation of sounds.
        communities (List[List[Int]]): List storing Lists containing the Sound ids that are in each community (cluster).
    Returns:
        Dict{Int: Float}: Dict containing the community centrality value for each sound
            ({<sound_id>: <community_centrality>}).
    """
    #
    subgraphs = [graph.subgraph(community) for community in communities]
    communities_centralities = [nx.algorithms.centrality.degree_centrality(subgraph) for subgraph in subgraphs]
    # merge and normalize in each community
    if max(communities_centralities[0].values()) > 0:
        node_community_centralities = {k: v/max(d.values()) for d in communities_centralities for k, v in d.items()}
    else:
        node_community_centralities = communities_centralities[0]
    return node_community_centralities

def cluster_dataset(dataset):
    """
    Aplly clustering on the given dataset for the different features.
    Saves results in the dataset dict.
    Display evaluation results.

    """
    print('\n')
    print('{} dataset:\n'.format(dataset['name']))
    for embedding_name, _ in EMBEDDING_FOLDERS.items():
        X = dataset['X_{}'.format(embedding_name)]
        # true_labels = dataset['labels']

        # knn graph clustering
        playlist_ids = {}
        for idx in range(len(X)):
            if dataset['unseen'][dataset['track_ids'][idx]] !='random':
                playlist_ids[idx] = 1
        labels, graph_json = cluster(X, playlist_ids)

        # kmeans clustering
        # labels, graph_json = cluster_kmeans(X, num_clusters=max(dataset['labels'])+1)

        dataset['labels_{}'.format(embedding_name)] = labels

        # evaluation metrics
        # purity, adjusted_mutual_info, adjusted_rand, cluster_cohesiveness, \
        #         semantic_cohesiveness, all_semantic_cohesiveness = evaluate(labels, true_labels)
        #
        # Associate semantic cohesiveness to label name
        # semantic_cohesiveness_per_category = {
        #     label_name: s_cohesiveness for label_name, s_cohesiveness
        #         in zip(dataset['label_names'], all_semantic_cohesiveness)
        # }
        #
        # dataset['evaluation_metrics_{}'.format(embedding_name)] = {
        #     'purity': purity,
        #     'adjusted_mutual_info': adjusted_mutual_info,
        #     'adjusted_rand': adjusted_rand,
        #     'average_cluster_cohesiveness': cluster_cohesiveness,
        #     'average_semantic_cohesiveness': semantic_cohesiveness,
        #     'semantic_cohesiveness_per_category': semantic_cohesiveness_per_category
        # }
        #
        # print('{} embeddings'.format(embedding_name))
        # print(dataset['evaluation_metrics_{}'.format(embedding_name)])

        # save clustered graph as json file
        for node in graph_json['nodes']:
            node.update({
                'track_id': dataset['track_ids'][node['id']],
                'unseen': dataset['unseen'][dataset['track_ids'][node['id']]],
                'preview': dataset['preview'][dataset['track_ids'][node['id']]],
                'track_name': dataset['track_names'][dataset['track_ids'][node['id']]],
                'track_labels': ', '.join([
                    genre_ids_to_name[genre_id]
                    for genre_id in dataset['labels'][dataset['track_ids'][node['id']]]
                    if (genre_id in genre_ids_to_name and not genre_id.endswith('1'))
                ]),
                'artist_name': dataset['artist_names'][dataset['track_ids'][node['id']]]
            })
        json.dump(graph_json, open(current_location+'/visualisation/web-visu/json/{}-{}.json'.format(dataset['name'],
                                                                     embedding_name), 'w'))
    return dataset


def cluster(X, playlist_ids):
    """
    Apply clustering with the features given as input.

    """
    similarity_matrix = compute_similarity_matrix(X)
    labels_knn, graph_json = knn_graph_clustering(playlist_ids, similarity_matrix, 20)
    return labels_knn, graph_json


def knn_graph_clustering(playlist_ids, similarity_matrix, k):
    """
    Apply k-nn graph-based clustering on items of the given similarity matrix.

    """
    graph = create_knn_graph(playlist_ids, similarity_matrix, k)
    #classes = com.best_partition(graph)

    # export clustered graph as json
    nx.set_node_attributes(graph, "something", 'group')

    """
    communities = []
    for node_id in range(similarity_matrix.shape[0]):
            communities.append(node['id'])
    node_community_centralities = point_centralities(graph, [communities])
    # Add cluster and centralities info to graph
    """
    node_community_centralities = point_centralities(graph, [range(len(similarity_matrix))])
    nx.set_node_attributes(graph, node_community_centralities, 'group_centrality')
    graph_json = json_graph.node_link_data(graph)
    for l in graph_json['links']:
        l['strength']=similarity_matrix[l['source'],l['target']].astype(float)


    #return [classes[k] for k in range(len(classes.keys()))], graph_json
    return ["seen","unseen"], graph_json


def create_knn_graph(playlist_ids, similarity_matrix, k):#, only_ids):
    """
    Returns a k-nn graph from a similarity matrix - NetworkX module.

    """
    np.fill_diagonal(similarity_matrix, 0) # for removing the 1 from diagonal
    g = nx.Graph()
    g.add_nodes_from(range(len(similarity_matrix)))
    for idx in range(len(similarity_matrix)):
        #print (dataset['unseen'])
        #print( dataset['track_ids'][idx])
        if idx in playlist_ids:
            #if idx in only_ids:
            g.add_edges_from([(idx, i) for i in nearest_neighbors(similarity_matrix, idx, k)])
    return g


def nearest_neighbors(similarity_matrix, idx, k):
    """
    Returns the k nearest meighbots in the similarity matrix.
    """
    distances = []
    for x in range(len(similarity_matrix)):
        distances.append((x,similarity_matrix[idx][x]))
    distances.sort(key=operator.itemgetter(1), reverse=True)
    return [d[0] for d in distances[0:k]]


def cluster_kmeans(X, num_clusters=5):
    """
    Applies k-means clustering and creates a knn graph for visualisation.
    Returns the labels in an array.

    """
    labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(X)
    classes = {idx: str(v) for idx, v in enumerate(labels)}

    similarity_matrix = compute_similarity_matrix(X)
    graph = create_knn_graph(similarity_matrix, 10)

    # export clustered graph as json
    nx.set_node_attributes(graph, classes, 'group')
    graph_json = json_graph.node_link_data(graph)

    return list(map(int,labels)), graph_json


def evaluate(predicted_labels, true_labels):
    """
    Returns different metrics of the evaluation of the predicted labels against the true labels.

    """
    adjusted_rand = metrics.adjusted_rand_score(true_labels, predicted_labels)
    adjusted_mutual_info = metrics.adjusted_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(np.array(true_labels), np.array(predicted_labels))
    cluster_cohesiveness = average_cluster_cohesiveness(true_labels, predicted_labels)
    semantic_cohesiveness, all_semantic_cohesiveness = average_semantic_cohesiveness(true_labels, predicted_labels)
    return round(purity, 4), round(adjusted_mutual_info, 4), round(adjusted_rand, 4), \
           round(cluster_cohesiveness, 4), round(semantic_cohesiveness, 4), all_semantic_cohesiveness


def purity_score(y_true, y_pred):
    """
    Returns the purity score.

    """
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner
    return metrics.accuracy_score(y_true, y_labeled_voted)


def average_cluster_cohesiveness(y_true, y_pred):
    # TODO: is there a way to extend this for multilabel (eg tags)?
    # -> automatically filter low cohesive cluster in Freesound
    num_clusters = max(y_pred) + 1
    num_labels = max(y_true) + 1

    num_class_members_in_clusters = [[0]*num_labels for _ in range(num_clusters)]
    for cluster_idx, label_idx in zip(y_pred, y_true):
        num_class_members_in_clusters[cluster_idx][label_idx] += 1

    prob_class_members_in_clusters = [
        [n/sum(num_class_members_in_cluster) for n in num_class_members_in_cluster]
            for num_class_members_in_cluster in num_class_members_in_clusters
    ]

    cluster_cohesiveness = [
        -sum([p*math.log2(p) for p in prob_class_members_in_cluster if p>0])
            for prob_class_members_in_cluster in prob_class_members_in_clusters
    ]

    return mean(cluster_cohesiveness)


def average_semantic_cohesiveness(y_true, y_pred):
    # TODO: check for each category of AudioSet which ones have more semantic cohesiveness
    num_clusters = max(y_pred) + 1
    num_labels = max(y_true) + 1

    num_class_members_in_clusters = [[0]*num_labels for _ in range(num_clusters)]
    for cluster_idx, label_idx in zip(y_pred, y_true):
        num_class_members_in_clusters[cluster_idx][label_idx] += 1

    prob_class_members_in_clusters = [
        [n/sum(num_class_members_in_cluster) for n in num_class_members_in_cluster]
            for num_class_members_in_cluster in num_class_members_in_clusters
    ]

    semantic_cohesiveness = [
        -sum([p*math.log2(p) for p in prob_class_member_in_clusters if p>0])
            for prob_class_member_in_clusters in zip(*prob_class_members_in_clusters)
    ]

    return mean(semantic_cohesiveness), semantic_cohesiveness


def add_sound_metadata_to_graph(dataset, metadata):
    for embedding_name, _ in EMBEDDING_FOLDERS.items():
        graph = json.load(open('web-visu/json/{}-{}.json'.format(dataset['name'],
                                                                 embedding_name), 'rb'))
        for node in graph['nodes']:
            try:
                node.update(metadata[str(node['sound_id'])])
            except:
                pass

        json.dump(graph, open('web-visu/json/{}-{}.json'.format(dataset['name'],
                                                                embedding_name), 'w'))


def add_clustering_info_for_web_visu(datasets):
    clustering_info = json.load(open(current_location+'/visualisation/web-visu/clustering_info.json'))
    datasets_names = [str(dataset['name']) for dataset in datasets]+clustering_info['datasets']
    clustering_info['datasets'] = list(set(datasets_names))
    clustering_info['features'] = [embedding_name for embedding_name, _ in EMBEDDING_FOLDERS.items()]
    clustering_info['datasets_names'].update({dataset['name'] : dataset['display_name'] for dataset in datasets})

    json.dump(clustering_info, open(current_location+'/visualisation/web-visu/clustering_info.json', 'w'))


def cluster_datasets(dataset_files, datasets):
    # # cleaning
    # remove_failed_embeddings(datasets)

    for dataset in datasets:
        # dataset = create_label_vector(dataset)
        dataset = load_features(dataset)
        dataset = cluster_dataset(dataset)

    # write_stats_datasets_xls(datasets)
    # write_stats_clusterings_xls(datasets)

    # add metadata to sounds in graph for web visualisation
    # metadata = json.load(open('json/sounds_metadata.json', 'rb'))
    for dataset_file, dataset in zip(dataset_files, datasets):
        # add_sound_metadata_to_graph(dataset, metadata)

        dataset_serializable = {}
        for k, v in dataset.items():
            if not k.startswith('X_'):
                if isinstance(v, np.int32):
                    dataset_serializable[k] = int(v)
                elif isinstance(v, np.ndarray):
                    dataset_serializable[k] = map(float, v)
                else:
                    dataset_serializable[k] = v
        #try:
        json.dump(dataset_serializable, open(current_location+'/visualisation/datasets_clustered/{}'.format(dataset_file), 'w'))
        #except:
        #    print("exception raised with {}".format(dataset_serializable))

    add_clustering_info_for_web_visu(datasets)
if __name__ == '__main__':
    # load datasets
    dataset_files = os.listdir('datasets')
    datasets = [json.load(open('datasets/'+f, 'rb')) for f in dataset_files]
    cluster_datasets(dataset_files, datasets)

