import os
import json
import random

current_location = "/data/contrastive-mir-learning"
mp3_folder = "/home/andres/projects/contrastive-mir-learning/mp3_cut"
# metadata
metadata = json.load(open(current_location+'/json/song_meta.json', 'rb'))
metadata_by_id = {obj['id']: obj for obj in metadata}

# existing files
#existing_files = os.listdir(current_location+'/data/embeddings/kakao_sample/embeddings_128_lr_0001_fusion_best/')
playlists_data = json.load(open(current_location+'/data/train.json'))
playlists_id_dict = {v['id']:i for i,v in enumerate(playlists_data)}
unseen_track_ids = {k:1 for k in json.load(open(current_location+'/json/unseen_ids.json', 'rb'))}


def create_random_dataset(name, num_tracks=1000, playlist=None):
    if playlist == None:
        position = random.randint(0, len(playlists_data))
    else:
        position = playlists_id_dict[int(playlist)]
    tracks = playlists_data[position]['songs']
    #mp3_tracks = []
    existing_ids = []
    for t in tracks:
        if os.path.isfile(os.path.join(mp3_folder,str(t)+".mp3")):
            existing_ids.append(t)

    if playlist == None and len(existing_ids) < 30:
        return create_random_dataset(name)

    unseen_random = []
    while len(unseen_random) < 150:
        random_tracks = random.sample(list(unseen_track_ids.keys()), 250)
        for t in random_tracks:
            if os.path.isfile(os.path.join(mp3_folder,str(t)+".mp3")) and t not in existing_ids:
                unseen_random.append(t)

    existing_ids += unseen_random
    #existing_ids = set([f.split('.')[0] for f in existing_files])
    random_ids = existing_ids #random.sample(existing_ids, num_tracks)
    track_metadata = {k: v for k, v in metadata_by_id.items() if k in random_ids}
    #print (track_metadata, random_ids)
    categories = {str(t_id): 'unseen' if t_id in unseen_track_ids else 'train' for t_id,_ in track_metadata.items()}
    for t in unseen_random:
        categories[str(t)] = 'random'

    dataset = {
        'name': playlists_data[position]['id'],
        'display_name': str(playlists_data[position]['id'])+' - '+playlists_data[position]['plylst_title'],
        'preview':{str(t_id): '/audio/'+str(t_id)+".mp3"  for t_id,_ in track_metadata.items()},
        'unseen': categories,
        'track_ids': [str(i) for i in random_ids],
        'labels': {str(t_id): v['song_gn_gnr_basket'] + v['song_gn_dtl_gnr_basket'] for t_id, v in track_metadata.items()},
        'track_names': {str(t_id): v['song_name'] for t_id, v in track_metadata.items()},
        'artist_names': {str(t_id): ', '.join(v['artist_name_basket']) for t_id, v in track_metadata.items()},
    }

    return dataset


if __name__ == '__main__':
    """
    Create random datasets
    """
    for k in range(10):
        dataset = create_random_dataset(k, )
        json.dump(dataset, open('datasets/{}.json'.format(k), 'w'))
