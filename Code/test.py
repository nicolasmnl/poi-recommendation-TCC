import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from gensim.models import Word2Vec

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings, TransformerModel
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss

device = torch.device("cuda:0")
torch.cuda.set_device(device) 

# word2vec = Word2Vec.load('./geo2vec_embeddings/austin-sl-tuple-geoc2vec0bin-wgt0.8pfp-concat-c.model')
word2vec = Word2Vec.load('./geo2vec_embeddings/austin-sl-tuple-n-itdl-0bin-wgt1.0-p.model')
cat_id_dict = pd.read_csv("./dataset/NYC/POI_catid_to_yelp_category.csv").set_index("POI_catid").to_dict()["yelp_category"]

def get_custom_embeddings(poi_catid):
    category = cat_id_dict[poi_catid].split(".")
    embeddings = []
    if (len(category) > 1):
        for cat in category:
            embeddings.append(word2vec.wv[cat])
        embed = np.stack(embeddings).mean(axis=0)
    else:
        embed = word2vec.wv[category[0]]
    
    return torch.tensor(embed)

    

def test(args):

    state_dict = torch.load(args.state_dict)

    args.save_dir = f"./results/{args.name}"

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in test data
    test_df = pd.read_csv(args.data_test)

    # Build POI graph (built from train)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # # Save ont-hot encoder
    # with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
    #     pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplacian matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_id2idx_dict = state_dict["poi_id2idx_dict"]

    # Cat id to index
    cat_id2idx_dict = state_dict["cat_id2idx_dict"]

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = state_dict["poi_idx2cat_idx_dict"]

    # User id to index
    user_id2idx_dict = state_dict["user_id2idx_dict"]

    # Print user-trajectories count
    traj_list = list(set(test_df['trajectory_id'].tolist()))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTest(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    test_dataset = TrajectoryDatasetTest(test_df)

    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    
    X = X.to(dtype=torch.float)
    A = A.to(dtype=torch.float)
    print("Shape do X")
    print(X.shape)
    print(X.shape[1])
    # X = X.to(device=args.device, dtype=torch.float)
    # A = A.to(device=args.device, dtype=torch.float)

    poi_embed_model = GCN(ninput=X.shape[1],
                          nhid=args.gcn_nhid,
                          noutput=args.poi_embed_dim,
                          dropout=args.gcn_dropout)
    poi_embed_model.load_state_dict(state_dict["poi_embed_state_dict"])
    print("Criou o poi embed model")

    # Node Attn Model
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    node_attn_model.load_state_dict(state_dict["node_attn_state_dict"])
    print("Criou o node_attn model")

    # %% Model2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
    user_embed_model.load_state_dict(state_dict["user_embed_state_dict"])
    print("Criou o user_embed model")

    # %% Model3: Time Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
    time_embed_model.load_state_dict(state_dict["time_embed_state_dict"])
    print("Criou o time_embed model")

    # %% Model4: Category embedding model
    # %% Model4: Category embedding model
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    pretrained_embeddings = np.array([get_custom_embeddings(poi_cat) for poi_cat in cat_ids])
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim, pretrained_embeddings)
    # if (args.use_embeddings):
        
    # else:
    #     cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)
    
    cat_embed_model.load_state_dict(state_dict["cat_embed_state_dict"])
    print("Criou o cat_embed model")
    

    # %% Model5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(args.user_embed_dim, args.poi_embed_dim)
    embed_fuse_model1.load_state_dict(state_dict["embed_fuse1_state_dict"])
    print("Criou o Fuse1_embed model")
                                      
    embed_fuse_model2 = FuseEmbeddings(args.time_embed_dim, args.cat_embed_dim)
    embed_fuse_model2.load_state_dict(state_dict["embed_fuse2_state_dict"])
    print("Criou o Fuse2_embed model")

    # %% Model6: Sequence model
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)
    seq_model.load_state_dict(state_dict["seq_model_state_dict"])

    print("Criou o seq_embed model")

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    print("Criou os criterios")

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings):
        # Parse sample
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]

        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        # input = torch.LongTensor([user_idx]).to(device=args.device)
        input = torch.LongTensor([user_idx])
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)

        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            # poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
            poi_embedding = torch.squeeze(poi_embedding)

            # Time to vector
            time_embedding = time_embed_model(
                # torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device)
                torch.tensor([input_seq_time[idx]], dtype=torch.float)
                )
                
            # time_embedding = torch.squeeze(time_embedding).to(device=args.device)
            time_embedding = torch.squeeze(time_embedding)

            # Category to embedding
            # cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_idx = torch.LongTensor([input_seq_cat[idx]])
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Fuse user+poi embeds
            fused_embedding1 = embed_fuse_model1(user_embedding, poi_embedding)
            fused_embedding2 = embed_fuse_model2(time_embedding, cat_embedding)

            # Concat time, cat after user+poi
            concat_embedding = torch.cat((fused_embedding1, fused_embedding2), dim=-1)

            # Save final embed
            input_seq_embed.append(concat_embedding)

        return input_seq_embed

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted
        
    test_batches_top1_acc_list = []
    test_batches_top5_acc_list = []
    test_batches_top10_acc_list = []
    test_batches_top20_acc_list = []
    test_batches_mAP20_list = []
    test_batches_mrr_list = []
    test_batches_loss_list = []
    test_batches_poi_loss_list = []
    test_batches_time_loss_list = []
    test_batches_cat_loss_list = []
    # src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
    src_mask = seq_model.generate_square_subsequent_mask(args.batch)
    for vb_idx, batch in enumerate(test_loader):
        if len(batch) != args.batch:
            # src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            src_mask = seq_model.generate_square_subsequent_mask(len(batch))

        # For padding
        batch_input_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_cat = []

        poi_embeddings = poi_embed_model(X, A)

        # Convert input seq to embeddings
        for sample in batch:
            traj_id = sample[0]
            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]
            input_seq_time = [each[1] for each in sample[1]]
            label_seq_time = [each[1] for each in sample[2]]
            label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
            input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings))
            batch_seq_embeds.append(input_seq_embed)
            batch_seq_lens.append(len(input_seq))
            batch_input_seqs.append(input_seq)
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))
            batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
            batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

        # Pad seqs for batch training
        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
        label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
        label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)

        # Feedforward
        # x = batch_padded.to(device=args.device, dtype=torch.float)
        x = batch_padded.to(dtype=torch.float)
        # y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
        y_poi = label_padded_poi.to(dtype=torch.long)
        
        # y_time = label_padded_time.to(device=args.device, dtype=torch.float)
        y_time = label_padded_time.to(dtype=torch.float)

        # y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
        y_cat = label_padded_cat.to(dtype=torch.long)
        y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

        # Graph Attention adjusted prob
        y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

        # Calculate loss
        loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
        loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
        loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
        loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

        # Performance measurement
        top1_acc = 0
        top5_acc = 0
        top10_acc = 0
        top20_acc = 0
        mAP20 = 0
        mrr = 0
        batch_label_pois = y_poi.detach().cpu().numpy()
        batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
        batch_pred_times = y_pred_time.detach().cpu().numpy()
        batch_pred_cats = y_pred_cat.detach().cpu().numpy()
        for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
            label_pois = label_pois[:seq_len]  # shape: (seq_len, )
            pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
            top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
            top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
            top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
            top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
            mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
            mrr += MRR_metric_last_timestep(label_pois, pred_pois)
        test_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
        test_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
        test_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
        test_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        test_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
        test_batches_mrr_list.append(mrr / len(batch_label_pois))
        test_batches_loss_list.append(loss.detach().cpu().numpy())
        test_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
        test_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
        test_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

        # Report testidation progress
        if (vb_idx % (args.batch * 2)) == 0:
            sample_idx = 0
            batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
            print(f'test_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                    f'test_move_loss:{np.mean(test_batches_loss_list):.2f} \n'
                    f'test_move_poi_loss:{np.mean(test_batches_poi_loss_list):.2f} \n'
                    f'test_move_time_loss:{np.mean(test_batches_time_loss_list):.2f} \n'
                    f'test_move_top1_acc:{np.mean(test_batches_top1_acc_list):.4f} \n'
                    f'test_move_top5_acc:{np.mean(test_batches_top5_acc_list):.4f} \n'
                    f'test_move_top10_acc:{np.mean(test_batches_top10_acc_list):.4f} \n'
                    f'test_move_top20_acc:{np.mean(test_batches_top20_acc_list):.4f} \n'
                    f'test_move_mAP20:{np.mean(test_batches_mAP20_list):.4f} \n'
                    f'test_move_MRR:{np.mean(test_batches_mrr_list):.4f} \n'
                    f'traj_id:{batch[sample_idx][0]}\n'
                    f'input_seq:{batch[sample_idx][1]}\n'
                    f'label_seq:{batch[sample_idx][2]}\n'
                    f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][2]]}\n'
                    f'pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                    f'label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n'
                    f'pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                    '=' * 100)
    # testid end --------------------------------------------------------------------------------------------------------
    
    mean_test_top1_acc = np.mean(test_batches_top1_acc_list)
    mean_test_top5_acc = np.mean(test_batches_top5_acc_list)
    mean_test_top10_acc = np.mean(test_batches_top10_acc_list)
    mean_test_top20_acc = np.mean(test_batches_top20_acc_list)
    mean_test_mAP20 = np.mean(test_batches_mAP20_list)
    mean_test_mrr = np.mean(test_batches_mrr_list)
    mean_test_loss = np.mean(test_batches_loss_list)
    mean_test_poi_loss = np.mean(test_batches_poi_loss_list)
    mean_test_time_loss = np.mean(test_batches_time_loss_list)
    mean_test_cat_loss = np.mean(test_batches_cat_loss_list)

    print(f"val_loss: {mean_test_loss:.4f}, "
            f"test_poi_loss: {mean_test_poi_loss:.4f}, "
            f"test_time_loss: {mean_test_time_loss:.4f}, "
            f"test_cat_loss: {mean_test_cat_loss:.4f}, "
            f"test_top1_acc:{mean_test_top1_acc:.4f}, "
            f"test_top5_acc:{mean_test_top5_acc:.4f}, "
            f"test_top10_acc:{mean_test_top10_acc:.4f}, "
            f"test_top20_acc:{mean_test_top20_acc:.4f}, "
            f"test_mAP20:{mean_test_mAP20:.4f}, "
            f"test_mrr:{mean_test_mrr:.4f}")

    with open(os.path.join(args.save_dir, 'metrics-test.txt'), "w") as f:
        print(f'test_top1_acc={mean_test_top1_acc}', file=f)
        print(f'test_top5_acc={mean_test_top5_acc}', file=f)
        print(f'test_top10_acc={mean_test_top10_acc}', file=f)
        print(f'test_top20_acc={mean_test_top20_acc}', file=f)
        print(f'test_mAP20={mean_test_mAP20}', file=f)
        print(f'test_mrr={mean_test_mrr}', file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    test(args)
