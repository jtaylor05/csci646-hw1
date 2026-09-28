from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

type Kernel = int | set[int]
type PerLayer = int | tuple[int] | list[int]
type Pools = type[nn.MaxPool2d | nn.AvgPool2d]

class CNNModel(nn.Module):
	
    def __init__(
            self, 
            out_channels: PerLayer, 
            kernel_size: PerLayer,
            stride: PerLayer,
            padding: PerLayer,
            output_size: PerLayer,
            /,*,
            classes: int = 10,
            use_dropout: bool =False, dropout: float = 0.2,
            use_pooling: bool =False, pool_class: Pools =nn.MaxPool2d, pool_kernel: Kernel = 3,
            use_batch_norm: bool =False,
            activation = None):
        super(CNNModel, self).__init__()

        if activation is None:
            activation = nn.ReLU
            
        size = None
        for i in (out_channels, kernel_size, stride, padding):
            if isinstance(i, int):
                continue
            elif size is not None and len(i) == size:
                continue
            else:
                if size is not None:
                    raise ValueError("All parameters must either be an 'int' or equal length iterables.")
                size = len(i)
                
        if size is None:
            size = 1
        
        if isinstance(out_channels, int): out_channels = [3] + [out_channels for _ in range(size)]
        else: out_channels = [3] + list(out_channels)
        
        if isinstance(kernel_size, int):  kernel_size  = [kernel_size  for _ in range(size)]
        if isinstance(stride, int):       stride       = [stride       for _ in range(size)]
        if isinstance(padding, int):      padding      = [padding      for _ in range(size)]
            
        layers = []
        outsize = 32
        for i in range(size):
            layers.append(nn.Conv2d(out_channels[i], out_channels[i+1], kernel_size[i], stride[i], padding[i]))
            
            #print(f"CONV: {int((outsize - kernel_size[i] + 2*padding[i]) / stride[i] + 1)}=({outsize} - {kernel_size[i]} + 2*{padding[i]}) / {stride[i]} + 1")
                        
            outsize = int((outsize - kernel_size[i] + 2*padding[i]) / stride[i] + 1)
                
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels[i+1]))
            
            layers.append(activation())
            
            if use_pooling:
                #print(f"POOL: {int((outsize - pool_kernel) / stride[i] + 1)}=({outsize} - {pool_kernel}) / {stride[i]} + 1")
                layers.append(pool_class(pool_kernel, 1, 0))
                outsize = int((outsize - pool_kernel) / 1 + 1)
            
            if use_dropout:
                layers.append(nn.Dropout(dropout))
                
        layers.append(nn.Flatten())
        
        #print(out_channels[-1], outsize)
        
        if isinstance(output_size, int): output_size = [output_size]
        else: output_size = list(output_size)
        output_size = [out_channels[-1] * outsize * outsize] + output_size
        
        for i in range(1, len(output_size)):
            layers.append(nn.Linear(output_size[i-1], output_size[i]))
            layers.append(activation())
            
        layers.append(nn.Linear(output_size[-1], classes))
        
        ## define CNN layers below
        self.conv = nn.Sequential(*layers)
        

    '''feed features to the model'''
    def forward(self, x): 
        
        result = self.conv(x)
        
        
        return result
        
		
		
	
		