import sys

sys.path.append('.')

import torch
from utils.match import Annoy
import numpy as np
import argparse
from annoy import AnnoyIndex


#  部分观看书本不足的用户会被归为冷启动用户，不产生推荐
#  测试用例run     python ./scripts/RecItem.py '00b19313fb62cfc5797612dd8460bccb'    in terminal
def Recommend(ProjectFolderDir, InputUser):
    raw_id_map = np.load(ProjectFolderDir + 'data/processed/raw_id_maps.npy', allow_pickle=True)
    user_map = raw_id_map[0]
    item_map = raw_id_map[1]

    def getKey(dic, value):
        if value not in dic.values():
            return None
        for key in dic:
            if dic[key] == value:
                result = key
        return result


    UserLabel = getKey(user_map, InputUser)
    # 对于该用户进行TOPN召回

    user_embeddings = torch.load(ProjectFolderDir+'temp/user_embedding.pth')
    item_embeddings = torch.load(ProjectFolderDir+'temp/item_embedding.pth')

    annoy = AnnoyIndex(item_embeddings.shape[1], metric='angular')
    annoy.load(ProjectFolderDir + "temp/item.ann.index")

    item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test = np.load(
        ProjectFolderDir+'data/saved/data_process.npy', allow_pickle=True)
    UserIndex = user_set['PATRON_ID'].tolist().index(UserLabel)
    UserEmdedding = user_embeddings[UserIndex]

    topk = 100
    items_idx = annoy.get_nns_by_vector(UserEmdedding.tolist(), n=topk, search_k = -1)
    # items_idx, items_scores = annoy.query(v=UserEmdedding, n=topk)
    ItemLabel = item_set['ITEM_ID'][items_idx].tolist()

    RecItem = []
    for i in ItemLabel:
        RecItem.append(item_map[i])
    return RecItem


if __name__=='__main__':
        # 采集输入的用户信息
    parser = argparse.ArgumentParser(description='Input User')
    parser.add_argument('InputUser', type=str, help='User ID')
    args = parser.parse_args()
    RecItem = Recommend(ProjectFolderDir = './', InputUser = args.InputUser)
    print(RecItem)