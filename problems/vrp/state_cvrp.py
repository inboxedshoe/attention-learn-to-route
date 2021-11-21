import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import math

class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def partial_finished(self):
        #if all the last visited are 0 then we finished route
        return not torch.all(self.visited_[:, :, -1])

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self, attention_neighborhood=0, pause_on_depots=False):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap


        #mask out any nodes not in k-nearest
        if attention_neighborhood !=0 :
            # neighbors = torch.topk(self.dist, 10, largest = False).indices 
            #select the distance vector for the node we are at now
            current_dis_vector = torch.index_select(self.dist,1,self.prev_a.reshape(-1))[:,0,:].unsqueeze(1)
            #we only want the current accessible top k nodes so offset any nodes previously masked
            #current_dis_vector.masked_fill_(mask_loc, float("1"))
            current_dis_vector[:,:,1:][mask_loc] = +math.inf

            #get the k nearest neighbors and then the cutoff point for every node's neighbors
            #neighbors_cutoff = torch.topk(self.dist, neighborhood_size, largest = False)[0][:,:,-1:]
            #neighbors_cutoff = torch.gather(neighbors_cutoff.reshape(mask_loc.shape[0],-1),-2,self.prev_a)
            neighbors_cutoff = torch.topk(current_dis_vector, attention_neighborhood, largest = False)[0][:,:,-1:].reshape(mask_loc.shape[0],-1)

            #set all distances greater than cutoff to true
            is_not_neighbor = current_dis_vector.squeeze() > neighbors_cutoff
            is_not_neighbor.unsqueeze_(1)

            #add the non-neighbors to the mask
            mask_loc = mask_loc | is_not_neighbor[:,:,1:]

            # is_not_neighbor = torch.ones(mask_loc.shape).to(exceeds_cap.dtype)
            # is_not_neighbor = is_not_neighbor.index_fill_(-1,neighbors,False)     


        if pause_on_depots:
            if not self.partial_finished():
                nodes_at_depot = (self.prev_a == 0).nonzero()[:, 0]
                mask_loc[nodes_at_depot, :] = True

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def get_neighborhood_mask(self, attention_neighborhood=20, static=True, encoder=True, include_state=False):
        if encoder:
            #for encoder: 1/True means don't attend
            #get distance matrix
            mask = torch.cdist(self.coords, self.coords, p=2)
            #get location of top k items
            _, idx = mask.topk(attention_neighborhood)
            #set all to zero
            mask.fill_(1)
            #set mask to keep
            mask[idx] = 0
            #depot should always be accessible
            mask[:,0,:] = 0
            mask[:,:,0] = 0

            mask = mask == 1

            return mask

    def construct_solutions(self, actions):
        return actions
