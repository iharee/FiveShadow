"""Recovered FaceParsingV2 graph; corrected Scale and AVG-pooling fields."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from .reader import scalar, vector, int_vector, pooling_parameters
from .restoration import RecoveredGraph, numbers

from ..assets import asset_root
class ParsingGraph(RecoveredGraph):
    def __init__(self, root: Path | None = None):
        root = asset_root(root)
        self.graph=next(g for g in json.loads((root/'parsing_models/dunnite_networks.json').read_text())
                        if g['resource_id']=='2102828093')
        self.data=(root/'parsing_models/2102828093.decoded.bin').read_bytes()
        self.weights={}
        def floats(table, field):
            offset,count=vector(self.data,table,field)
            if offset is None:return np.empty(0,np.float32)
            return np.frombuffer(self.data,dtype='<f4',count=count,offset=offset).copy()
        for node in self.graph['nodes']:
            key=f"node_{node['index']:04d}"
            p=node['parameter_table_offset']
            if 'convolution' in node:
                c=node['convolution']
                if c['encoded_weight_table'] is not None:raise ValueError('Unexpected quantized parsing convolution')
                ic,oc,group=c['input_channels'],c['output_channels'],c['group'];kx,ky=c['kernel']
                shape=(ic,oc//group,ky,kx) if node['type_name']=='Deconvolution' else (oc,ic//group,ky,kx)
                self.weights[key+'_weight']=torch.from_numpy(floats(p,1).reshape(shape))
                self.weights[key+'_bias']=torch.from_numpy(floats(p,0))
            elif node['type_name']=='Const':
                dims=int_vector(self.data,p,0);dt=scalar(self.data,p,2)
                value=np.asarray(int_vector(self.data,p,6),np.int32) if dt==3 else floats(p,4)
                if dt==1 and vector(self.data,p,3)[0] is not None:raise ValueError('Unresolved quantized const')
                self.weights[key+'_value']=torch.from_numpy(value.reshape(dims))
            elif node['type_id']==70:
                self.weights[key+'_scale']=torch.from_numpy(floats(p,2))
                self.weights[key+'_offset']=torch.from_numpy(floats(p,1))

    def execute_node(self,node,args):
        kind=node['type_name'];p=node['parameter_table_offset'];key=f"node_{node['index']:04d}"
        def get(i,fmt='<i',default=0):return scalar(self.data,p,i,fmt,default)
        if node['type_id']==70:
            scale=self.weights[key+'_scale'];bias=self.weights[key+'_offset']
            value=args[0]*scale[None,:,None,None]
            return value+bias[None,:,None,None] if bias.numel() else value
        if node['type_id']==52:
            if get(3,'<B')!=1 or not get(5,'<B'):raise ValueError('Unresolved Pool3D parameters')
            return args[0].mean(dim=tuple(range(2,args[0].ndim)),keepdim=True)
        if node['type_id']==53:return torch.sigmoid(args[0])
        if kind in ('Unsqueeze','Squeeze') and len(args)>1:
            value=args[0]
            for axis in sorted(numbers(args[1]),reverse=kind=='Squeeze'):
                value=value.unsqueeze(axis) if kind=='Unsqueeze' else value.squeeze(axis)
            return value
        if kind=='StridedSlice' and get(7)==1:
            starts,ends,axes=map(numbers,args[1:4])
            steps=numbers(args[4]) if len(args)>4 else [1]*len(axes)
            slices=[slice(None)]*args[0].ndim
            for a,b,axis,step in zip(starts,ends,axes,steps):slices[axis]=slice(a,b,step)
            return args[0][tuple(slices)]
        if kind=='Interp':
            if get(5)!=2:raise ValueError('Unresolved resize type')
            if len(args)>1:
                sizes=numbers(args[1])
                if args[1].is_floating_point():
                    sizes=[round(args[0].shape[i+2]*float(args[1][i+2])) for i in range(2)]
                if len(sizes)==4:sizes=sizes[-2:]
                if len(sizes)!=2:raise ValueError(f'Unexpected resize input {sizes}')
            else:sizes=[get(4),get(3)]
            return F.interpolate(args[0],size=sizes,mode='bilinear',align_corners=bool(get(7,'<B')))
        if kind=='Reduction':
            axes=numbers(args[1]) if len(args)>1 else int_vector(self.data,p,3)
            if get(1,'<B')!=0:raise ValueError('Unresolved reduction op')
            return args[0].sum(dim=tuple(axes),keepdim=bool(get(0,'<B')))
        if kind=='BinaryOp' and node['binary_operation'] in (3,7):
            return args[0]/args[1]
        if kind=='Pooling':
            pool=pooling_parameters(self.data,p)
            if pool['pad_type'] or pool['global']:raise ValueError('Unresolved pooling padding/global')
            if pool['type']==0:
                return F.max_pool2d(args[0],pool['kernel'],pool['stride'],pool['padding'],ceil_mode=pool['ceil'])
            if pool['count_type'] not in (1,2):raise ValueError('Unresolved default average denominator')
            return F.avg_pool2d(args[0],pool['kernel'],pool['stride'],pool['padding'],
                                ceil_mode=pool['ceil'],count_include_pad=pool['count_type']==1)
        return super().execute_node(node,args)

    @torch.inference_mode()
    def parse(self,image):
        nodes=self.graph['nodes'];input_node=next(n for n in nodes if n['type_name']=='Input')
        values={input_node['outputs'][0]:torch.from_numpy(image.copy())}
        counts=Counter(t for n in nodes for t in n['inputs'])
        kept={self.graph['tensor_names'].index(n) for n in self.graph['output_names']}
        trace=[]
        for node in nodes:
            if node['type_name']=='Input':continue
            args=[values[t] for t in node['inputs']]
            try:result=self.execute_node(node,args)
            except Exception as error:
                raise RuntimeError(f"Node {node['index']} {node['type_name']}, shapes {[list(a.shape) for a in args]}: {error}") from error
            values[node['outputs'][0]]=result
            trace.append({'node':node['index'],'kind':node['type_name'],'shape':list(result.shape),
                          'min':float(result.min()),'max':float(result.max())})
            for t in node['inputs']:
                counts[t]-=1
                if not counts[t] and t not in kept:values.pop(t)
        return values[next(iter(kept))].numpy(),trace

