import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow.keras.backend as K


import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow.keras.backend as K




class MyHeavisideActivation(tf.keras.layers.Layer):
  def __init__(self, num_outputs, threshold=.5, **kwargs):
    super(MyHeavisideActivation, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.threshold = threshold

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return tf.cond(inputs > self.threshold, 
                   lambda: tf.add(tf.multiply(inputs,0), 1), # set to 1
                   lambda: tf.multiply(inputs, 0))           # set to 0


def pairwise_distanceR(point_cloud, mask):
    """Compute pairwise distance in the eta-phi plane for the point cloud.
    Uses the third dimension to find the zero-padded terms
    Args:
      point_cloud: tensor (batch_size, num_points, 2)
      IMPORTANT: The order should be (eta, phi)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape()[0]
    # make sure nothing bad happens in the extreme case in which you feed the network with just 1 example per mini-batch
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud = point_cloud[:, :, :2]  # Only use eta and phi, BxNx2
    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) #Bx2xN
    point_cloud_phi = point_cloud_transpose[:, 1:, :] #Bx1xN
    point_cloud_phi = tf.tile(point_cloud_phi, [1, point_cloud_phi.get_shape()[2], 1]) #BxNxN
    point_cloud_phi_transpose = tf.transpose(point_cloud_phi, perm=[0, 2, 1]) #BxNxN
    point_cloud_phi = tf.math.abs(point_cloud_phi - point_cloud_phi_transpose) #compute distance in the phi space for all the particles in the cloud (more precisely, its abs)
    is_biggerpi = tf.greater_equal(tf.abs(point_cloud_phi), np.pi) #is the abs greater than pi?
    point_cloud_phi_corr = tf.where(is_biggerpi, 2 * np.pi - point_cloud_phi, point_cloud_phi) #Correct if bigger than pi
    #build matrix of pairwise DeltaRs between particles
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)  # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1,
                                       keepdims=True)  # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    deltaR_matrix = point_cloud_square + point_cloud_square_transpose + point_cloud_inner #this matrix contains the squared, pairwise DRs between particles in the cloud
    deltaR_matrix = deltaR_matrix - tf.square(point_cloud_phi) #subtract non-corrected delta_phi squared part
    deltaR_matrix = deltaR_matrix + tf.square(point_cloud_phi_corr) #add corrected delta_phi squared part
    #Move zero-padded away
    point_shift = 1000*tf.expand_dims(mask,-1) #BxNx1
    point_shift_transpose = tf.transpose(point_shift,perm=[0, 2, 1]) #Bx1xN
    zero_mask = point_shift_transpose + point_shift #when adding tensors having a dimension equal to 1, tf tiles them to make them compatible for the sum
    zero_mask = tf.where(tf.equal(zero_mask, 2000), tf.zeros_like(zero_mask), zero_mask)
    return deltaR_matrix + zero_mask, zero_mask

def pairwise_distance(point_cloud, mask): 
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape()[0]
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)  # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1,
                                       keepdims=True)  # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose + mask


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    distances, nn_idx = tf.math.top_k(neg_adj, k=k)  # values, indices
    return nn_idx,distances


def get_neighbors(point_cloud, nn_idx, k=20):
    """Construct neighbors feature for each point
      Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int
      Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """    
    point_cloud = tf.squeeze(point_cloud, axis=-2)
    point_cloud_shape = tf.shape(point_cloud)
    point_cloud_shape_int = point_cloud.get_shape()
    batch_size = point_cloud_shape[0]
    num_points = point_cloud_shape[1]
    num_dims = point_cloud_shape_int[2]
    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    
    return point_cloud_neighbors


class AttnFeat(layers.Layer):

    def __init__(self, k=10, momentum=0.99, filters_C2DNB=32,
                 kernel_size_C2DNB=(1, 1), strides_C2DNB=(1, 1), padding_C2DNB='same',
                 activation_NoBias_C2DNB=tf.keras.activations.relu, activation=tf.nn.leaky_relu,
                 expand_dims=True, name='AttnFeat', **kwargs): 
        super(AttnFeat, self).__init__(name=name, **kwargs)
        self.k = k
        self.momentum = momentum
        self.activation = activation
        self.filters_C2DNB = filters_C2DNB
        self.kernel_size_C2DNB = kernel_size_C2DNB
        self.strides_C2DNB = strides_C2DNB
        self.padding_C2DNB = padding_C2DNB
        self.activation_NoBias_C2DNB = activation_NoBias_C2DNB
        self.expand_dims = expand_dims

        self.BatchNormNoBias = layers.BatchNormalization(momentum=self.momentum)

        self.Conv2DEdgeFeat = layers.Conv2D(filters=self.filters_C2DNB, kernel_size=[1, 1], strides=(1, 1),
                                            padding='valid')

        self.BatchNormEdgeFeat = layers.BatchNormalization(momentum=self.momentum)

        self.Conv2DSelfAtt = layers.Conv2D(filters=1, kernel_size=[1, 1], strides=(1, 1),
                                           padding='valid')

        self.BatchNormSelfAtt = layers.BatchNormalization(momentum=self.momentum)

        self.Conv2DNeighAtt = layers.Conv2D(filters=1, kernel_size=[1, 1], strides=(1, 1),
                                            padding='valid')

        self.BatchNormNeighAtt = layers.BatchNormalization(momentum=self.momentum)

        self.Conv2DNoBias = layers.Conv2D(filters=self.filters_C2DNB, kernel_size=self.kernel_size_C2DNB,
                                          strides=self.strides_C2DNB, padding=self.padding_C2DNB, use_bias=False,
                                          activation=self.activation_NoBias_C2DNB,
                                          kernel_initializer='glorot_uniform') 

    def call(self, inputs, training=None, **kwargs): 
        
        #Implement the operations described in the ABCNet paper (https://link.springer.com/article/10.1140/epjp/s13360-020-00497-3)

        nn_idx = kwargs['nn_idx']
        mask = kwargs['mask']        
        if self.expand_dims:
            inputs = tf.expand_dims(inputs, axis=-2)

        mask_neighbors = get_neighbors(tf.reshape(mask,(-1,tf.shape(mask)[1],1,1)),
                                       nn_idx=nn_idx,
                                       k=self.k)  # Group up the neighbors using the index passed on the arguments
        mask_neighbors = -10000*tf.transpose(mask_neighbors,(0,1,3,2))
        neighbors = get_neighbors(inputs, nn_idx=nn_idx,
                                  k=self.k)  # Group up the neighbors using the index passed on the arguments

        inputs_tiled = tf.tile(inputs, [1, 1, self.k, 1])
        edge_feature_pre = inputs_tiled - neighbors  # Make the edge features yij

        if 'deltaR' in kwargs:
            deltaR = kwargs['deltaR']
            edge_feature_pre = tf.concat([edge_feature_pre,tf.expand_dims(deltaR,-1)],-1)

        #Encode the points in cloud by a 1 CNN layer, the weights are learnable parameters of this filter
        new_feature = self.Conv2DNoBias(inputs)
        new_feature = self.BatchNormNoBias(new_feature)

        #Encode the edge features by a 1 CNN layer, the weights are learnable parameters of this filter
        edge_feature = self.Conv2DEdgeFeat(edge_feature_pre)
        edge_feature = self.BatchNormEdgeFeat(edge_feature)

        #Create self-coefficients by passing transformed points to 1 CNN layer with output size 1
        self_attention = self.Conv2DSelfAtt(new_feature)
        self_attention = self.BatchNormSelfAtt(self_attention)

        #Create local-coefficients by passing transformed edges to 1 CNN layer with output size 1
        neighbor_attention = self.Conv2DNeighAtt(edge_feature)
        neighbor_attention = self.BatchNormNeighAtt(neighbor_attention)
        
        #Finally, create attention coefficients by summing the previously created coefficients... 
        logits = self_attention + neighbor_attention
        logits = tf.transpose(logits, [0, 1, 3, 2])
        
        #... and by applying leaky-relu non-linearity. To align the attention coefficients, also apply softmax normalization
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits)+mask_neighbors)

        #Now, each point is associated with k attention coefficients. To compute a single attention coefficient per point, perform linear comibination...
        vals = tf.linalg.matmul(coefs, edge_feature)
        #... and apply non-linearity
        outputs = self.activation(vals)
        return outputs, self_attention, edge_feature


class GAPBlock(layers.Layer):

    def __init__(self, nheads=1, k=10, momentum=0.99, filters_C2DNB=32,
                 kernel_size_C2DNB=(1, 1), strides_C2DNB=(1, 1),
                 padding_C2DNB='same',
                 activation_NoBias_C2DNB=tf.keras.activations.relu,
                 activation=tf.keras.activations.relu,
                 expand_dims=True,
                 name='GAPBlock', **kwargs):  
        super(GAPBlock, self).__init__(name=name, **kwargs)
        self.k = k
        self.nheads = nheads
        self.momentum=momentum
        self.attn_feat_layers = []
        self.Name=name
        for i in range(nheads):
            self.attn_feat_layers.append(
                AttnFeat(k=self.k, momentum=self.momentum, filters_C2DNB=filters_C2DNB, kernel_size_C2DNB=kernel_size_C2DNB,
                         strides_C2DNB=strides_C2DNB, padding_C2DNB=padding_C2DNB, expand_dims=expand_dims,
                         activation_NoBias_C2DNB=activation_NoBias_C2DNB, activation=activation))

    def call(self, inputs, training=None, **kwargs): 
        nn_idx, layer_input,  mask = inputs 
        attns = []
        local_features = []
        for i in range(self.nheads):
            out, self_att, edge_feat = self.attn_feat_layers[i](inputs=layer_input,nn_idx=nn_idx, mask=mask,**kwargs)
            attns.append(out)  # This is the edge feature * att. coeff. activated by Leaky RELU, one per particle
            local_features.append(edge_feat)  # Those are the yij

        neighbors_features = tf.concat(attns, axis=-1)
        # neighbors_features = tf.concat([tf.expand_dims(point_cloud, axis=-2), neighbors_features], axis=-1) 
        locals_transform = tf.reduce_mean(tf.concat(local_features, axis=-1), axis=-2, keepdims=True)
        return tf.squeeze(neighbors_features, axis=2), tf.squeeze(locals_transform, axis=2), tf.squeeze(self_att, axis=2)


def ABCNet(npoint,nfeat=1,momentum=0.99):
    # Define the shapes of the multidimensionanl inputs for the pointcloud (per particle variables)
    # Always leave out the batchsize when specifying the shape
    inputs = Input(shape=(npoint,nfeat))
    k = 20 
    mask = tf.where(inputs[:,:,2]==0,K.ones_like(inputs[:,:,2]),K.zeros_like(inputs[:,:,2]))
    idx_list = list(range(nfeat))
    idx_list.pop(1)
    idx_list.pop(-1)
    adj_1, zero_matrix = pairwise_distanceR(inputs[:,:,:3], mask)
    nn_idx,dist = knn(adj_1, k=k)

    neighbors_features_1, graph_features_1, attention_features_1 = GAPBlock(k=k, filters_C2DNB=16, padding_C2DNB = 'valid', name='Gap1')((nn_idx, tf.gather(inputs,idx_list,axis=-1),mask),deltaR=dist)
    x = layers.Conv1D(filters = 64, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(neighbors_features_1)
    x = layers.Conv1D(filters = 64, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = layers.BatchNormalization(momentum=momentum)(x)
    x01=x
    
    # adj_2 = pairwise_distance(x, zero_matrix)
    # adj_2 = adj_1 #keep same pairs for faster computation
    
    neighbors_features_2, graph_features_2, attention_features_2 = GAPBlock(k=k, momentum=momentum, filters_C2DNB=32, padding_C2DNB = 'valid', name='Gap2')((nn_idx,x, mask))
    x = layers.Conv1D(filters = 128, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(neighbors_features_2)        
    #x = layers.BatchNormalization(momentum=momentum)(x)
    x = layers.Conv1D(filters = 128, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    x = layers.BatchNormalization(momentum=momentum)(x)
    x11 = x
    
    #perform aggregation. Aggregation is a concat tf.operation
    x = tf.concat([x01, x11, graph_features_1, graph_features_2], axis = -1)

    x = layers.Conv1D(filters = 256, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)        
    x = layers.BatchNormalization(momentum=momentum)(x)

    x_prime = x
    
    #Perform AveragePooling
    x = tf.reduce_mean(x, axis=1,keepdims=True)
    
    expand=tf.tile(x, [1, npoint, 1]) #after pooling, recover x tensor's second dimension by tiling

    x = tf.concat(values = [expand, x_prime], axis=-1)

    x = layers.Conv1D(filters = 128, kernel_size = 1, strides = 1, padding='valid', kernel_initializer='glorot_uniform', activation='relu')(x)
    # x = layers.Dropout(0.6)(x)
    # x = layers.BatchNormalization(momentum=momentum)(x)

    outputs = layers.Conv1D(filters = 1, kernel_size = 1, strides = 1,  padding='valid', kernel_initializer='glorot_uniform', activation='hard_sigmoid')(x)

    return inputs,outputs


def SWD(y_true, y_pred,nprojections=128):
    pu_pfs = y_true[:,:,:y_true.shape[2]//2]
    nopu_pfs = y_true[:,:,y_true.shape[2]//2:]

    charge_pu_mask = tf.cast(tf.expand_dims(tf.abs(pu_pfs[:,:,-1])>0,-1),tf.float32)
    charge_nopu_mask = tf.cast(tf.expand_dims(tf.abs(nopu_pfs[:,:,-1])>0,-1),tf.float32)    
    nopu_pfs = nopu_pfs[:,:,:4]
    pu_pfs = pu_pfs[:,:,:4]*y_pred



    def _getSWD(pu_pf,nopu_pf):    
        proj = tf.random.normal(shape=[tf.shape(pu_pf)[0],tf.shape(pu_pf)[2], nprojections])
        proj *= tf.math.rsqrt(tf.reduce_sum(tf.square(proj), 1, keepdims=True))

        p1 = tf.matmul(nopu_pf, proj) #BxNxNPROJ
        p2 = tf.matmul(pu_pf, proj) #BxNxNPROJ
        p1 = sort_rows(p1, tf.shape(pu_pf)[1])
        p2 = sort_rows(p2, tf.shape(pu_pf)[1])
        
        wdist = tf.reduce_mean(tf.square(p1 - p2),-1)
        return wdist
    
    def _getMET(particles):
        px = tf.abs(particles[:,:,2])*tf.math.cos(particles[:,:,1])
        py = tf.abs(particles[:,:,2])*tf.math.sin(particles[:,:,1])
        met = tf.stack([px,py],-1)
        # print(met)
        return met


    met_pu = tf.reduce_sum(_getMET(pu_pfs)*y_pred,1)
    met_nopu = tf.reduce_sum(_getMET(nopu_pfs),1)
    met_mse = tf.reduce_sum(tf.square(met_pu[:,:2] - met_nopu[:,:2]),-1)

    
    # nopu_pfs = tf.expand_dims(nopu_pfs[:,:,3],-1)
    # pu_pfs = tf.expand_dims(pu_pfs[:,:,3],-1)*y_pred
    
    # nopu_pfs = nopu_pfs
    # pu_pfs = pu_pfs*y_pred


#     wdist = _getSWD(pu_pfs,nopu_pfs)
#     notzero = tf.reduce_sum(tf.where(wdist>0,tf.ones_like(wdist),tf.zeros_like(wdist)))    
#     return 1e3*tf.reduce_sum(wdist)/tf.reduce_sum(notzero)
# # #+tf.reduce_mean(met_mse)

    wdist_charge = _getSWD(pu_pfs*charge_pu_mask,nopu_pfs*charge_nopu_mask)
    wdist_neutral = _getSWD(pu_pfs*tf.cast(charge_pu_mask==0,tf.float32),
                            nopu_pfs*tf.cast(charge_nopu_mask==0,tf.float32))
    wdist = _getSWD(pu_pfs,nopu_pfs)


    return 1e3*tf.reduce_mean(wdist_neutral) + 1e3*tf.reduce_mean(wdist_charge) + 1e3*tf.reduce_mean(wdist)

    
def sort_rows(matrix, num_rows):
    matrix_T = tf.transpose(matrix, [0,2,1])
    sorted_matrix_T,index_matrix = tf.math.top_k(matrix_T, num_rows)    
    return tf.transpose(sorted_matrix_T, [0,2, 1])
