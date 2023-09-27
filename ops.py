import numpy as np
import torch
import torch.nn.functional as F



# def random_sampling(max_queries_sample, max_queries_possible, num_samples):
#     num_queries = torch.randint(low=0, high=max_queries_sample, size=(num_samples, ))
#     qh = int(np.sqrt(max_queries_possible))
#     mask = torch.zeros(num_samples, max_queries_possible)
#     mask_all = mask.reshape(num_samples, qh, qh)
#     mask = torch.zeros(num_samples, qh//2 * qh//2)
#     max_queries_possible_ = qh//2 * qh//2
#
#     for code_ind, num in enumerate(num_queries):
#         if num == 0:
#             continue
#         random_history = torch.multinomial(torch.ones(max_queries_possible_), num, replacement=False)
#         mask[code_ind, random_history.flatten()] = 1.0
#
#     mask = mask.reshape(num_samples, qh//2, qh//2)
#
#     # mask_all[:, qh//4:qh//4+qh//2, qh//4:qh//4+qh//2] = mask
#
#     mask_all[:, :qh // 4, :qh // 4] = mask[:, :qh // 4, :qh // 4]
#     mask_all[:, -qh // 4:, -qh // 4:] = mask[:, -qh // 4:, -qh // 4:]
#     mask = mask_all.reshape(num_samples, max_queries_possible)
#     return mask


def random_sampling(max_queries_sample, max_queries_possible, num_samples, empty=False, exact=False, return_ids=False):
    if exact:
        num_queries = [max_queries_sample]*num_samples
    else:
        num_queries = torch.randint(low=0, high=max_queries_sample, size=(num_samples,))

    mask = torch.zeros(num_samples, max_queries_possible)
    if empty: return mask

    # ids = torch.arange(max_queries_possible)
    histories = []
    for code_ind, num in enumerate(num_queries):
        if num == 0:
            continue
        random_history = torch.multinomial(torch.ones(max_queries_possible), num, replacement=False)
        mask[code_ind, random_history.flatten()] = 1.0
        if return_ids:
            # print(random_history)
            histories.append(random_history)

    if return_ids:
        if exact and len(histories) > 0:
            histories = torch.stack(histories)
        return histories
    return mask

def random_sampling_w_prior(max_queries_sample, max_queries_possible, num_samples, empty=False, exact=False, return_ids=False, case='patches', wh = None):
    if exact:
        num_queries = [max_queries_sample]*num_samples
    else:
        num_queries = torch.randint(low=0, high=max_queries_sample, size=(num_samples,))

    if case.startswith('patches'):
        probs = torch.ones(max_queries_possible)
        side = int(np.sqrt(max_queries_possible))
        discard = int(side//2)
        probs = probs.reshape(side, side)
        probs[:discard, :discard] = 0.1
        probs[:discard, -discard:] = 0.1
        probs[-discard:, -discard:] = 0.1
        probs[-discard:, :discard] = 0.1
        # assert probs[probs!=0].sum() >= max_queries_sample
    
    if case.startswith('attributes'):
        # TODO: maybe start at the center (max prob) and lower towards the sides
        assert isinstance(wh, tuple)
        probs = torch.ones(max_queries_possible)
        probs = probs.reshape(wh[0], wh[1]) # attr, obj
        for i in wh[0]:
            probs[i] /= 2**i
        # assert probs[probs != 0].sum() >= max_queries_sample

    mask = torch.zeros(num_samples, max_queries_possible)
    if empty: return mask

    # ids = torch.arange(max_queries_possible)
    histories = []
    for code_ind, num in enumerate(num_queries):
        if num == 0:
            continue
        random_history = torch.multinomial(torch.ones(max_queries_possible), num, replacement=False)
        mask[code_ind, random_history.flatten()] = 1.0
        if return_ids:
            # print(random_history)
            histories.append(random_history)

    if return_ids:
        if exact and len(histories) > 0:
            histories = torch.stack(histories)
        return histories
    return mask

def random_single_sampling(max_queries_possible, num_samples):
    mask = torch.zeros(num_samples, max_queries_possible)

    for code_ind in range(num_samples):
        random_history = torch.multinomial(torch.ones(max_queries_possible), 1, replacement=False)
        mask[code_ind, random_history.flatten()] = 1.0

    return mask

def adaptive_sampling(x, num_queries, querier, patch_size, max_queries):
    device = x.device
    N, C, H, W = x.shape

    mask = torch.zeros(N, (H - patch_size + 1)*(W - patch_size + 1)).to(device)
    final_mask = torch.zeros(N, (H - patch_size + 1)*(W - patch_size + 1)).to(device)
    patch_mask = torch.zeros((N, C, H, W)).to(device)
    final_patch_mask = torch.zeros((N, C, H, W)).to(device)
    sorted_indices = num_queries.argsort()
    counter = 0

    with torch.no_grad():
        for i in range(max_queries + 1):
            while (counter < N):
                batch_index = sorted_indices[counter]
                if i == num_queries[batch_index]:
                    final_mask[batch_index] = mask[batch_index]
                    final_patch_mask[batch_index] = patch_mask[batch_index]
                    counter += 1
                else:
                    break
            if counter == N:
                break
            query_vec, query_soft = querier(patch_mask, mask)
            mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
            patch_mask = update_masked_image(patch_mask, x, query_vec, patch_size)
    return final_mask, final_patch_mask

def get_labels(list, size):
    # We assume a squared image
    list = [list[0], list[1]*2/size - 1]
    return torch.cat(list, dim=1)

def average_gradients(grad_S, grad_q, split):
    grads = []
    for i in range(len(split)-1):
        grad = (grad_S[split[i]:split[i+1]].sum(0, keepdims=True) + grad_q[i:i+1]) / (split[i+1]-split[i]+1)
        grads.append(grad)
    return torch.cat(grads)

def expand_to_S(x, split):
    split_diff = split[1:] - split[:-1]
    all_x = []
    for x_i in x:
        all_x_i = []
        for bid, l in enumerate(split_diff):
            all_x_i.append(torch.repeat_interleave(x_i[bid:bid+1], int(l), 0))
        all_x.append(torch.cat(all_x_i))
    return all_x


def update_masked_image(masked_image, original_image, query_vec, patch_size):
    N, _, H, W = original_image.shape
    device = masked_image.device

    query_vec = query_vec.view(N, 1, (H - patch_size + 1), (W - patch_size + 1))

    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    # convoluting signal with kernel and applying padding
    mask = F.conv2d(query_vec, kernel, stride=1, padding=patch_size - 1, bias=None)
    output = mask * original_image                                           #get new patch
    modified_history = (1 - mask) * masked_image + output # TODO: Aixo esta malament si es superposen dos patches. Bueno per MNIST no. pero cutre.    #update and get new masked image
    # modified_history = torch.clamp(modified_history, min=-1.0, max=1.0)

    return modified_history

def get_patch_mask(mask, x, patch_size, null_val=0): # TODO: Make differentiable like update_masked_image
    patch_mask = torch.zeros(x.size()).to(x.device) + null_val

    if patch_size == 1:
        mask = mask.reshape(x.shape[0],-1,*x.shape[2:])
        mask[mask!=1] = 0
        out = x * mask
        nulls = (1 - mask)*null_val
        return out + nulls, None, None, None
    all_image_parts, all_image_part_coords = [], []
    acc_idx, batch_split = 0, [0]
    for batch_index in range(mask.size(0)):

        positive_indices = torch.where(mask[batch_index] == 1)[0]
        index_i = positive_indices // (x.size(3) - patch_size + 1) # TODO: Check
        index_j = positive_indices % (x.size(3) - patch_size + 1)

        image_parts = []
        image_part_coords = []
        for row in range(patch_size):
            for col in range(patch_size):
                part_of_image = x[batch_index, :, index_i + row, index_j + col]
                image_parts.append(part_of_image)
                if row == patch_size//2 + 1 and col == patch_size//2 + 1:
                    image_part_coords = \
                        (torch.stack([index_i + row, index_j + col]))
                patch_mask[batch_index, :, index_i + row, index_j + col] = part_of_image
        all_image_parts.append(torch.stack(image_parts).reshape(patch_size**2, -1))
        all_image_part_coords.append(image_part_coords.reshape(2, -1))
        acc_idx += part_of_image.shape[-1]
        batch_split.append(acc_idx)

    image_parts = torch.cat(all_image_parts, dim=-1).permute(1,0)
    image_part_coords = torch.cat(all_image_part_coords, dim=-1).permute(1,0)

    return patch_mask, image_parts, image_part_coords, torch.tensor(batch_split)

def get_single_query(original_image, query_vec, patch_size):
    N, _, H, W = original_image.shape
    device = original_image.device

    query_vec = query_vec.view(N, 1, (H - patch_size + 1), (W - patch_size + 1))

    kernel = torch.ones(1, 1, patch_size, patch_size, requires_grad=False).to(device)
    # convoluting signal with kernel and applying padding
    patch_mask = F.conv2d(query_vec, kernel, stride=1, padding=patch_size - 1, bias=None)

    output = patch_mask * original_image
    q_v = output.reshape(-1)[patch_mask.reshape(-1) == 1].reshape(-1, patch_size**2)
    # TODO: get position, create gradient, stack. q_ij = torch.where(patch_mask.reshape(N, -1) == 1)[0]
    q_ij = (query_vec==torch.max(query_vec)).nonzero()[:, -2:] + patch_size//2
    return q_v, q_ij


def compute_queries_needed(logits, threshold):
    """Compute the number of queries needed for each prediction.

    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)

    """
    assert 0 < threshold and threshold < 1, 'threshold should be between 0 and 1'
    n_samples, n_queries, _ = logits.shape
    device = logits.device

    # turn logits into probability and find queried prob.
    prob = F.softmax(logits, dim=2)
    prob_max = prob.amax(dim=2)

    # `decay` to multipled such that argmax finds
    #  the first nonzero that is above threshold.
    threshold_indicator = (prob_max >= threshold).float().to(device)
    decay = torch.linspace(10, 1, n_queries).unsqueeze(0).to(device)
    semantic_entropy = (threshold_indicator * decay).argmax(1)

    # `threshold_indicator`==0 is to check which
    # samples did not stop querying, hence indicator vector
    # is all zeros, preventing bug that yields argmax as 0.
    semantic_entropy[threshold_indicator.sum(1) == 0] = n_queries
    semantic_entropy[threshold_indicator.sum(1) != 0] += 1

    return semantic_entropy


