import torch.nn as nn
from timm.models.layers import trunc_normal_




"""
This is a generic conv block based on the ConvNext paper it contains a depthwise convolution with groups = dim meaning that 
the channel will all be convoluted seperately, this improves the performance. The layers are combined by following up with 
two convolutions, one upscaling and one downscaling with kernel size 1. Essentially this recombines the layers that were
convolved seperately before. The activation function in between the up and downsample is a GELU. The norm is a regularization 
technique.
"""
class Conv_block(nn.Module):
    def __init__(self,dim,dConv_kernel_size=7):
        super().__init__()
        self.depth_conv = nn.Conv2d(dim,dim,kernel_size=dConv_kernel_size,padding=int((dConv_kernel_size-1)/2),groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.conv_1 = nn.Conv2d(dim,dim*4,kernel_size=1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(dim*4,dim,kernel_size=1)

    def forward(self,x):
        input = x
        x = self.depth_conv(x)
        x = self.norm(x)
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        return(x+input)


"""
Convolutional encoder based on the ConvNext paper. it repeats the Conv_Blocks depth times per layer and follows them with a
downscaling operation.
"""
class Encoder(nn.Module):
    def __init__(self, depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
        super().__init__()
        self.layers = nn.ModuleList()
      
        for layer_n,depth in enumerate(depths):
            for sublayer_n in range(depth):
                self.layers.append(Conv_block(dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:
                self.layers.append(nn.Conv2d(dims[layer_n],dims[layer_n+1],kernel_size= 2, stride = 2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)
    
"""
The convolutional decoder is used in the pretraining fase where it forms the autoencoder together with the convolutional encoder.
The goal of this is to teach the encoder how to create dense image embeddings without relying on the more unpredictable transformer.
"""
class Decoder(nn.Module):
    def __init__(self ,depths=[3, 3, 9, 3,3,1],dims=[96, 192, 384, 768,768,1536],dConv_kernel_size=7):
        super().__init__()
        self.depths = list(reversed(depths))
        self.dims = list(reversed(dims))
        self.layers = nn.ModuleList()
        for layer_n,depth in enumerate(self.depths):

            for _ in range(depth):
                self.layers.append(Conv_block(self.dims[layer_n],dConv_kernel_size))
            if layer_n < len(depths)-1:     
                self.layers.append(nn.ConvTranspose2d(self.dims[layer_n],self.dims[layer_n+1],kernel_size=2,stride=2))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return(x)


"""
The autoencoder binds the encoder and decoder together in the end this wasn't used after running some trials since pre training the 
encoder in this fashion didn't seem to affect the ViT training speed and accuracy too much.
"""
class AutoEncoder(nn.Module):
    def __init__(self,depths,dims, dConv_kernel_size=7):
        super().__init__()
        self.encoder = Encoder(depths=depths,dims = dims, dConv_kernel_size = dConv_kernel_size)
        self.decoder = Decoder(depths=depths,dims = dims, dConv_kernel_size = dConv_kernel_size)

    def forward(self,x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return(decoding)

    

"""
This is my implementation of the ATM mechanism as described in the SegViT paper, it is very similar to the original.
I just ommited the dropout since this was an extra hyperparameter to tune which there was really no time for, it
would theoretically help with potential overfitting.
"""
class ATM(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias =False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim// num_heads
        self.scale = qk_scale or head_dim ** -.5
        #create the linear transformations needed to create the Queries Keys and Values for the cross attention 
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)

    def forward(self,xq,xk,xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]
        
        """
        dividing the Queries Keys and Values over the heads, first by reshaping the data from B,N,C to B,N,Num_heads,C/Num_heads
        and then permuting the output to get the shape B,Num_heads,N,C/Num_heads
        """
        q = self.q(xq).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        """
        first dot product between query(B,Num_heads,Nq,C/Num_heads) and key(B,Num_heads,C/Num_heads,Nk)
        to obtain attn(B,Num_heads,Nq,Nk)
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        
        """
        second dot product between attn(B,Num_heads,Nq,Nk) and value(B,Num_heads,C/Num_heads,Nk)
        and transposing/reshaping to obtain query_Out(B,Nq,C)
        """
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        """
        returning q and the middle attention after the first dotproduct summed over the heads and then divided by the number of 
        heads to get the mean activation across the heads.
        """
        return x, attn_save.sum(dim=1) / self.num_heads
    



class Transformer_Decoder_Layer(nn.Module):
    
    def __init__(self,dim,num_heads=1,qkv_bias=False,feed_forward_dim = None):
        super().__init__()

        if feed_forward_dim == None:
            feed_forward_dim = dim*4

        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.multihead_attn =ATM(dim,num_heads,qkv_bias)
        self.norm2 = nn.LayerNorm(dim)

        #MLP
        self.linear1 = nn.Linear(dim,feed_forward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim,dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self,x,memory):
        x2 = self.self_attn(x,x,x)[0]
        x = x + self.norm1(x2)
        x2 , attn = self.multihead_attn(x,memory,memory)
        x = x + self.norm2(x2)

        #MLP
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x = x + x2
        x = self.norm3(x)
        
        return(x,attn)
    


"""
This is the final model used for the satellite image segmentation. It initializes the class and positional embeddings and feeds the data into the 
encoder and decoder to obtain the predictions for the classes and masks notice that there is no sigmoid for the attentions like in the paper.
this is important because we have to account for that in the training loop.
"""
class Conv_ATM(nn.Module):
    
    def __init__(self,encoder,dim,num_heads=8, n_classes = 2,qkv_bias=True, num_patches= 64, num_layers = 12, feed_forward_dim = None):
        
        super().__init__()
        self.num_layers = num_layers
        #assign encoder, this is not created internally so a pretrained encoder can be used.
        self.encoder = encoder
        self.num_patches = num_patches
        #this refers to the number of channels
        self.dim = dim

        #create the decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(Transformer_Decoder_Layer(dim,num_heads,qkv_bias,feed_forward_dim))

        self.decoder_norm = nn.LayerNorm(dim)
        
        #class embedding
        self.q = nn.Embedding(n_classes,dim)
        #positional embedding
        self.pos_emb = nn.Embedding(num_patches,dim)
        
        #initialize weights to a truncated normal with very low standard deviation to gradually increase influence of the attention mechanism
        trunc_normal_(self.q.weight,std = 0.05)
        trunc_normal_(self.pos_emb.weight,std = 0.05)
        
        #classifier consisting of a linear layer making sure to add a extra output for no class (background) and a softmax to obtain the probabilities
        self.classifier = nn.Sequential(nn.Linear(n_classes,n_classes+1),nn.Softmax(-1))
        
    def forward(self,x):
        
        #divide the input images into patches of 16 by 16 pixels
        batch_size = x.shape[0]
        x = x.unfold(1,16,16).unfold(2,16,16).flatten(0,2)
        #feed patches into encoder
        x =self.encoder(x).flatten(1)
        #reshape the featuremaps to the correct dimensions
        x = x.reshape(batch_size,self.num_patches,self.dim)
        
        x = self.decoder_norm(x)

        #add the positional embedding to the features
        x = x + self.pos_emb.weight
        
        attns = []
        #load the class embeddings we repeat it to fit the batch size
        q = self.q.weight.repeat(batch_size,1,1)
        
        #we sequentially pass it through the Transformer decoder layers, sidenote q is constantly updated by the layers
        for dec_layer in self.decoder_layers:
            q, attn = dec_layer(q,x)
            attns.append(attn)
        
        #make class predictions using the classifier
        class_prediction = self.classifier(q.permute(0,2,1))
        return(q,attns,class_prediction)








"""
The is the transformer encoder layer that i built, it is very similar to the decoder, it does however not use the ATM mechanism and
only employs a self attention node. In the final model this is not used for reasons mentioned in the report.
"""
class Transformer_Encoder_Layer(nn.Module):
    
    def __init__(self,dim,num_heads=1 ,feed_forward_dim = None):
        super().__init__()

        if feed_forward_dim == None:
            feed_forward_dim = dim*4
    
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        #MLP
        self.linear1 = nn.Linear(dim,feed_forward_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(feed_forward_dim,dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self,x):
        x2 = self.self_attn(x,x,x)[0]
        x = x + self.norm1(x2)
        
        #FeedForward
        x2 = self.linear1(x)
        x2 = self.activation(x2)
        x2 = self.linear2(x2)
        x = x + x2
        x = self.norm2(x)
        
        return(x)


"""
This is the ATM_ViT using the transformer encoder layer that i did not use for reasons mentioned in the paper.
"""
class ATM_ViT(nn.Module):
    
    def __init__(self,dim,num_heads=8, n_classes = 2,qkv_bias=False, num_patches= 64, num_layers = 12, feed_forward_dim = None , device = 'cpu'):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.encoder_layers.append(Transformer_Encoder_Layer(dim,num_heads,feed_forward_dim))
            self.decoder_layers.append(Transformer_Decoder_Layer(dim,num_heads,qkv_bias,feed_forward_dim))

        self.q = nn.Embedding(n_classes,dim)
        self.pos_emb = nn.Embedding(num_patches,dim)
        
        self.classifier = nn.Sequential(nn.Linear(n_classes,n_classes+1),nn.Softmax(-1))

    def forward(self,x):
        memory = []
        attns  = []

        #pos embedding
        x = x + self.pos_emb

        #encoder
        for enc_layer in self.encoder_layers:
            memory.append(enc_layer(x))

        memory.reverse()

        #decoder
        batch_size = x.shape[0]
        q = self.q.weight.repeat(batch_size,1,1)

        for memory_item,dec_layer in zip(memory,self.decoder_layers):
            q, attn = dec_layer(q,memory_item)
            attns.append(attn)
        
        class_prediction = self.classifier(q.permute(0,2,1))

        return(q,attns,class_prediction)