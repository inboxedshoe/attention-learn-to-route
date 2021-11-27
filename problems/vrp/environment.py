import torch

class AgentVRP():

    VEHICLE_CAPACITY = 1.0

    def __init__(self, input, device):
        depot = input["depot"]
        loc = input["loc"]

        self.batch_size, self.n_loc, _ = loc.shape  # (batch_size, n_nodes, 2)

        # Coordinates of depot + other nodes
        self.coords = torch.cat((depot[:, None, :], loc), -2)
        self.demand = input["demand"].to(torch.float32)

        # Indices of graphs in batch
        self.ids = torch.arange(start=0, end=self.batch_size, dtype=torch.int64, device=device)[:, None]

        # State
        self.prev_a = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=device)
        self.from_depot = self.prev_a == 0
        self.used_capacity = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=device)

        # Nodes that have been visited will be marked with 1
        self.visited = torch.zeros((self.batch_size, 1, self.n_loc + 1), dtype=torch.uint8, device=device)

        # Step counter
        self.i = torch.zeros(1, dtype=torch.int64, device=device)

        # Constant tensors for scatter update (in step method)
        self.step_updates = torch.ones((self.batch_size, 1), dtype=torch.uint8, device=device)  # (batch_size, 1)
        self.scatter_zeros = torch.zeros((self.batch_size, 1), dtype=torch.int64, device=device)  # (batch_size, 1)

        self.device = device

    @staticmethod
    def outer_pr(a, b):
        """Outer product of matrices
        """
        return torch.einsum('ki,kj->kij', a, b)

    def get_att_mask(self):
        """ Mask (batch_size, n_nodes, n_nodes) for attention encoder.
            We mask already visited nodes except depot
        """

        # We dont want to mask depot
        att_mask = torch.squeeze(self.visited.to(torch.float32), dim=-2)[:, 1:]  # [batch_size, 1, n_nodes] --> [batch_size, n_nodes-1]

        # Number of nodes in new instance after masking
        cur_num_nodes = self.n_loc + 1 - torch.reshape(torch.sum(att_mask, -1), (-1, 1))  # [batch_size, 1]

        att_mask = torch.cat([torch.zeros(att_mask.shape[0], 1, device=self.device), att_mask], dim=-1)

        ones_mask = torch.ones_like(att_mask, device=self.device)

        #Create square attention mask from row-like mask
        att_mask = AgentVRP.outer_pr(att_mask, ones_mask) \
                   + AgentVRP.outer_pr(ones_mask, att_mask) \
                   - AgentVRP.outer_pr(att_mask, att_mask)

        #att_mask = att_mask[:, None, :].repeat(8, att_mask.shape[-1], 1)

        return att_mask.to(torch.bool), cur_num_nodes

    def all_finished(self):
        """Checks if all games are finished
        """
        return torch.all(self.visited.to(torch.bool))

    def partial_finished(self):
        """Checks if partial solution for all graphs has been built, i.e. all agents came back to depot
        """
        return torch.all(self.from_depot) and self.i != 0

    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
        """

        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity
        exceeds_cap = self.demand + self.used_capacity > self.VEHICLE_CAPACITY

        # We mask nodes that are already visited or have too much demand
        # Also for dynamical model we stop agent at depot when it arrives there (for partial solution)
        mask_loc = visited_loc.to(torch.bool) | exceeds_cap[:, None, :] | (
                    (self.i > 0) & self.from_depot[:, None, :])

        # We can choose depot if 1) we are not in depot OR 2) all nodes are visited
        mask_depot = self.from_depot & (torch.sum(mask_loc == False, dim=-1) > 0)

        return torch.cat([mask_depot[:, :, None], mask_loc], dim=-1)

    def step(self, action):
        # Update current state
        selected = action[:, None]

        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # We have to shift indices by 1 since demand doesn't include depot
        # 0-index in demand corresponds to the FIRST node
        # selected_demand = torch.gather(self.demand, 0,
        #                                torch.cat([self.ids, torch.clamp(self.prev_a - 1, 0, self.n_loc - 1)],
        #                                          dim=1))[:, None]  # (batch_size, 1)

        selected_demand = torch.gather(self.demand, 0, torch.clamp(self.prev_a - 1, 0, self.n_loc - 1))  # (batch_size, 1)

        # We add current node capacity to used capacity and set it to zero if we return to the depot
        self.used_capacity = (self.used_capacity + selected_demand) * (1.0 - self.from_depot.to(torch.float32))

        # Update visited nodes (set 1 to visited nodes)
        idx = torch.cat([self.ids, self.scatter_zeros, self.prev_a], dim=-1).to(torch.int64)[:, None, :]  # (batch_size, 1, 3)
        self.visited = torch.scatter(self.visited, -1, self.prev_a.unsqueeze(1), 1)  # (batch_size, 1, n_nodes)

        self.i = self.i + 1

    @staticmethod
    def get_costs(dataset, pi):
        # Place nodes with coordinates in order of decoder tour
        loc_with_depot = torch.cat([dataset["depot"][:, None, :], dataset["loc"]], dim=1)  # (batch_size, n_nodes, 2)
        #d = torch.gather(loc_with_depot, 1, pi.to(torch.int64))
        #d = loc_with_depot[pi]
        d = batch_gather_vec(loc_with_depot, pi.to(torch.int64))

        # Calculation of total distance
        # Note: first element of pi is not depot, but the first selected node in the path
        return (torch.sum(torch.norm(d[:, 1:] - d[:, :-1], p=2, dim=2), dim=1)
                + torch.norm(d[:, 0] - dataset["depot"], p=2, dim=1)  # Distance from depot to first selected node
                + torch.norm(d[:, -1] - dataset["depot"], p=2, dim=1))  # Distance from last selected node (!=0 for graph with longest path) to depot


def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    flat_first = torch.reshape(
        tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(
        torch.arange(shape[0]).cuda() * shape[1],
        [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices + offset]
    return output