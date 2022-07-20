import torch
import torch.nn as nn
import torch.nn.functional as F





class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels, source):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        user_idx = torch.argwhere(source).reshape(-1) # source = 1 인 것들의 idx
        shop_idx = torch.argwhere(source ^ 1).reshape(-1) # source = 0 인 것들의 idx

        user_emb = torch.index_select(embeddings, 0, user_idx)
        shop_emb = torch.index_select(embeddings, 0, shop_idx)

        user_labels = torch.index_select(labels, 0, user_idx)
        shop_labels = torch.index_select(labels, 0, shop_idx)

        pairwise_dist = _pairwise_distance(user_emb, shop_emb , squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(user_labels, shop_labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(user_labels, shop_labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True) # row마다 max인 dist 출력
            
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
            # invalid(not negative)에 max 더해서 min 취했을 때 나올 수 없도록 제외하는 작업
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True) # hardest negative = min dist + diff label

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)  # relu -> max(0,_)
            triplet_loss = torch.mean(triplet_loss)

            
        else:
            # pairwise_dist shape -> (user, shop)
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size) -> (user, shop, shop)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1) -> (user,shop,1)
            # and the 2nd (batch_size, 1, batch_size) -> (user, 1, shop)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(user_emb, shop_emb, squared=False, eps=1e-16):

    # (batch)*(batch)에서 (user)*(shop)의 dim으로 
    # x = embeddings (batch, emb_size) => x = (user, emb_size) * x.t = (emb_size, shop)

    # Compute the 2D matrix of distances between user embedding and shop embeddings
    u_sq = torch.matmul(user_emb, user_emb.t()).diag().unsqueeze(1)
    s_sq = torch.matmul(shop_emb, shop_emb.t()).diag()

    user_sq = u_sq * torch.ones(len(user_emb), len(shop_emb))
    shop_sq = s_sq * torch.ones(len(user_emb), len(shop_emb))

    user_shop = torch.matmul(user_emb, shop_emb.t())
    distances = user_sq + shop_sq - 2 * user_shop
    distances = F.relu(distances)

    if not squared:  # 0에 sqrt 취하기 위해 eps 더한 후 다시 0으로 만들어 줌
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(user_labels, shop_labels):  # mask dim = (user)*(shop), user
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if user_labels[i] == shop_labels[j]
    mask = torch.unsqueeze(user_labels, 1) == torch.unsqueeze(shop_labels, 0)


    return mask


def _get_anchor_negative_triplet_mask(user_labels, shop_labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(user_labels, 1) == torch.unsqueeze(shop_labels, 0)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(user_labels, shop_labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - j, k are distinct
        - labels[i] == labels[j] and labels[j] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[j] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask
    







