import torch
import torch.nn as nn
import numpy as np
from .spline import *


class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device
    
    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward 
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cpu'):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        '''
        # scale base、scale sp：控制base function和spline function的幅度
        super(KANLayer, self).__init__()
        # size 
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        # 生成torch.size([size, num])的網格，每個網格的範圍為grid_range, ex: input_dim = 8, output_dim = 3, gird = torch.size([24, 5])
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        # grid、coef：能夠被optimizer識別並更新的參數，grid：(size, num)，coef：(size, num + k)
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size) # tensor[0,1,2,...,size-1], size = out_dim * in_dim
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
        self.device = device

    def forward(self, x):
        '''
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        '''
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        # 16512, 8 -> 24, 16512
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim) # 預留激活值，用於可視化(x軸)
        base = self.base_fun(x).permute(1, 0)  # torch.nn.SiLU(), shape (batch, size)
        y = coef2curve(x_eval=x, grid=self.grid[self.weight_sharing], coef=self.coef[self.weight_sharing], k=self.k, device=self.device)  # shape (size, batch)
        # 根據input x, 網格grid, spline function係數coef, degree k, 計算出y
        y = y.permute(1, 0)  # shape (batch, size)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)  # shape (batch, out_dim)
        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y, preacts, postacts, postspline
    
    def update_grid_from_samples_watermark_ver(self, x, embedding_key):
        '''
        在特定神经元上更新网格，同时添加浮水印信号。
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3, 3, steps=100)[:, None].to(device)
        >>> model.update_grid_from_samples_watermark_ver(x, embedding_key)
        >>> print(model.grid.data)
        '''
        # 提取水印参数
        i, j, watermark_func, amplitude, frequency, phase = embedding_key['i'], embedding_key['j'], embedding_key['watermark_func'], embedding_key['amplitude'], embedding_key['frequency'], embedding_key['phase']

        # 处理输入
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0] # 將x由小到大排好

        # 计算原始的输出评估
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)

        # 计算水印信号
        y_watermark = y_eval + amplitude * watermark_func(2 * np.pi * frequency * x_pos + phase).to(self.device)
        
        # 选择更新的神经元
        id = j * self.in_dim + i 
        y_eval[id] = y_watermark[id]

        # 自适应和均匀网格更新
        num_intervals = self.grid.shape[1] - 1
        ids = [int(batch / num_intervals * i) for i in range(num_intervals)] + [-1] # 基本上將 batch 均勻劃分為 num_intervals 部分，並計算每一部分的起始index。
        grid_adaptive = x_pos[:, ids] # 是從 x_pos 根據 ids 索引取出的值。這個操作基本上是從 x_pos 中選取特定的點
        margin = 0.01
        grid_uniform = torch.cat([
            grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a 
            for a in torch.linspace(0, 1, steps=self.grid.shape[1], device=self.device)
        ], dim=1) # 基於 grid_adaptive 的最小值和最大值之間均勻分布。是一個更均勻分佈的網格

        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # 更新系数
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)
        return [x_pos.cpu().detach().numpy(), y_eval.cpu().detach().numpy()]


    def update_grid_from_samples(self, x):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat([grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # grid_uniform：整個定義域的均勻分布；grid_adaptive：根據input的實際分佈生成的grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device) # (size, grid size + degree k), 不同spline basis function對應參數

    def watermark_verification(self, i, j, key_coef, key_grid, x):
        '''
        watermark verification
        
        Args:
        -----
            i : int
                input index
            j : int
                output index
            key_coef : 2D torch.float
                key coefficients
            key_grid : 2D torch.float
                key grid
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> key_coef, key_grid = model.save_curve(0, 0)
        >>> model.watermark_verification(0, 0, key_coef, key_grid, x)
        '''
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        y_eval_key = coef2curve(x_pos, key_grid, key_coef, self.k, device=self.device)
        id = j * self.in_dim + i 
        y_eval = y_eval[id]
        # print(f"\n\nDifference between the original and key sample: {torch.mean((y_eval - y_eval_key) **2).item()}")
        return torch.mean((y_eval - y_eval_key) **2).item()

    def get_coef_grid(self, i, j):
        '''
        save curve
        '''
        coef = self.coef[j * self.in_dim + i].cpu().detach().numpy()
        grid = self.grid[j * self.in_dim + i].cpu().detach().numpy()
        return coef, grid


    def initialize_grid_from_parent(self, parent, x):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device)
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1, spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1, spb.coef.shape[1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        '''
        lock activation functions to share parameters based on ids
        
        Args:
        -----
            ids : list
                list of ids of activation functions
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        '''
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        '''
        unlock activation functions
        
        Args:
        -----
            ids : list
                list of ids of activation functions
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=3, num=5, k=3)
        >>> model.lock([[0,0],[1,2],[2,1]]) # set (0,0),(1,2),(2,1) functions to be the same
        >>> print(model.weight_sharing.reshape(3,3))
        >>> model.unlock([[0,0],[1,2],[2,1]]) # unlock the locked functions
        >>> print(model.weight_sharing.reshape(3,3))
        tensor([[0, 1, 2],
                [3, 4, 0],
                [6, 0, 8]])
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        '''
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1
