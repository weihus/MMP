import torch
import torch.nn as nn
from layers import SubViewEncoder, AttributeReconstruction, StructureReconstruction, MainViewEncoder
from utils import calculate_similarity, aggregate_views, calculate_similarity, custom_softmax, euclidean_distance
import torch.nn.functional as F
class MVGAD(nn.Module):
    def __init__(self, feat_size, hidden_dim, in_channels_views, device, suppression_factor, enhancement_factor):
        super(MVGAD, self).__init__()        
        # Initialize an embedding model for each view
        self.embedding_models = nn.ModuleList([SubViewEncoder(feat, hidden_dim).to(device) for feat in in_channels_views])
        
        # Initialize the original embedding model
        self.original_embedding_model = MainViewEncoder(feat_size, hidden_dim).to(device)
        
        # Initialize attribute and structure reconstruction models
        self.attr_reconstruction_model = AttributeReconstruction(feat_size, hidden_dim).to(device)
        self.struct_reconstruction_model = StructureReconstruction(hidden_dim).to(device)
        
        # Initialize similarities and weights as trainable parameters
        self.similarities = nn.Parameter(torch.Tensor(len(in_channels_views)))
        self.weights = nn.Parameter(torch.Tensor(len(in_channels_views)))
        # self.weights = torch.ones(len(in_channels_views))

        # self.similarities = nn.Parameter(F.softmax((torch.rand(len(in_channels_views)).uniform_(-0.5, 0.5))))
        # self.weights = self.similarities

        self.suppression_factor = suppression_factor
        self.enhancement_factor = enhancement_factor

        nn.init.uniform_(self.similarities)
        nn.init.uniform_(self.weights)
    
    def forward(self, x, edge_index, attrs):
        # Compute the original embedding without additional views
        original_embedding = self.original_embedding_model(x, edge_index)
        view_embeddings = []
        
        # Compute embeddings for each view
        for model, view_data in zip(self.embedding_models, attrs):
            view_embedding = model(view_data, edge_index)
            view_embeddings.append(view_embedding)
        
        # Calculate similarities using the initialized similarities
        similarities = []
        distances = []
        for view_embedding in view_embeddings:
            similarity = calculate_similarity(original_embedding, original_embedding+view_embedding)
            distance = euclidean_distance(original_embedding, original_embedding+view_embedding)
            similarities.append(similarity)
            distances.append(distance)

        # Convert similarities to a tensor and apply custom softmax to obtain weights
        self.similarities.data = torch.tensor(similarities)

        self.weights.data = custom_softmax(self.similarities, distances, self.suppression_factor, self.enhancement_factor)  # Update weights parameter

        # print(self.similarities.data, self.weights.data)
        # raise
        # Aggregate the view embeddings using the computed weights
        aggregated_embedding = aggregate_views(view_embeddings, self.weights)

        # Reconstruct attributes and structures from the aggregated embedding
        reconstructed_attrs = self.attr_reconstruction_model(aggregated_embedding, edge_index)
        reconstructed_structs = self.struct_reconstruction_model(aggregated_embedding, edge_index)
        
        # Return the reconstructed attributes, structures, original embedding, and aggregated embedding
        return reconstructed_attrs, reconstructed_structs, original_embedding, aggregated_embedding
