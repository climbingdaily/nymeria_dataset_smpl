import torch
from time import time
from .bvh_search import BVH

def find_collisions(human_verts, faces, max_collisions=8):
    triangles = human_verts[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=max_collisions)

    torch.cuda.synchronize()
    start = time.time()

    outputs = m(triangles)
    
    torch.cuda.synchronize()
    print('Elapsed time', time.time() - start)

    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]
    print(collisions.shape)