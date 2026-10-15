"""Recovered restoration graph executed independently through PyTorch CPU.

Parameter mappings follow the corrected AVG-pooling research implementation.
Model assets are supplied externally; no original application code is loaded.
"""
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .reader import int_vector, scalar, pooling_parameters

from ..assets import asset_root


def numbers(value):
    return [int(v) for v in value.reshape(-1).tolist()]


class RecoveredGraph:
    def __init__(self, model, root: Path | None = None):
        root = asset_root(root)
        self.graph = next(g for g in json.loads((root / 'dunnite_networks.json').read_text())
                          if g['resource_id'] == str(model))
        if str(model) == '1655581403':
            raise ValueError('Classifier has unresolved operations; use a restoration model.')
        self.data = (root / 'decoded_models' / f'{model}.decoded.bin').read_bytes()
        self.weights = {}
        with np.load(root / 'recovered_weights' / f'{model}.float.npz', allow_pickle=False) as arrays:
            self.weights = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
        self.output_ids = {self.graph['tensor_names'].index(name) for name in self.graph['output_names']}

    def execute_node(self, node, args):
        kind, table = node['type_name'], node['parameter_table_offset']
        key = f"node_{node['index']:04d}"
        def get(field, fmt='<i', default=0):
            return scalar(self.data, table, field, fmt, default)
        if kind == 'Const':
            return self.weights[key + '_value']
        if kind in ('Convolution', 'DepthwiseConvolution', 'Deconvolution'):
            conv = node['convolution']
            kw = dict(stride=tuple(reversed(conv['stride'])),
                      padding=tuple(reversed(conv['padding'])),
                      dilation=tuple(reversed(conv['dilation'])), groups=conv['group'])
            operation = F.conv_transpose2d if kind == 'Deconvolution' else F.conv2d
            result = operation(args[0], self.weights[key + '_weight'], self.weights[key + '_bias'], **kw)
            return F.relu(result) if conv['relu'] else result
        if kind == 'Permute':
            return args[0].permute(node['permutation'])
        if kind == 'Reshape':
            shape = numbers(args[1]) if len(args) == 2 else int_vector(self.data, table, 0)
            return args[0].reshape(shape)
        if kind == 'ConvertTensor':
            # Recovered models alternate custom NCHW(0) / packed NCHW(3).
            # CPU tensors here always use the same logical axes, with no pack.
            if {get(0, '<B'), get(1, '<B')} != {0, 3}:
                raise ValueError('Unresolved tensor format conversion')
            return args[0]
        if kind == 'LayerNorm':
            param = node['layer_norm']
            if param['axis'] != [-1] or param['group'] != 1:
                raise ValueError('Unsupported LayerNorm axes/group')
            return F.layer_norm(args[0], (args[0].shape[-1],), self.weights[key + '_gamma'],
                                self.weights[key + '_beta'], param['epsilon'])
        if kind == 'MatMul':
            left, right = args[:2]
            if get(1, '<B'):
                left = left.transpose(-1, -2)
            if get(2, '<B'):
                right = right.transpose(-1, -2)
            value = torch.matmul(left, right)
            return value + args[2] if len(args) == 3 else value
        if kind == 'BinaryOp':
            op = node['binary_operation']
            if get(0) not in (0, 1):
                raise ValueError('Unresolved fused BinaryOp activation')
            if op == 0:
                result = args[0] + args[1]
            elif op == 1:
                result = args[0] - args[1]
            elif op == 2:
                result = args[0] * args[1]
            elif op == 19:
                result = torch.remainder(args[0], args[1])
            else:
                raise ValueError(f'Unsupported BinaryOp {op}')
            return F.relu(result) if get(0) == 1 else result
        if kind == 'Shape':
            return torch.tensor(list(args[0].shape), dtype=torch.int32)
        if kind == 'Rank':
            return torch.tensor(args[0].ndim, dtype=torch.int32)
        if kind in ('Unsqueeze', 'Squeeze'):
            axes = int_vector(self.data, table, 0)
            result = args[0]
            if kind == 'Squeeze' and not axes:
                return result.squeeze()
            for axis in sorted(axes, reverse=kind == 'Squeeze'):
                result = result.unsqueeze(axis) if kind == 'Unsqueeze' else result.squeeze(axis)
            return result
        if kind == 'StridedSlice':
            # Only all-zero slicing masks occur in the recovered restoration graphs.
            if any(get(i) for i in range(1, 8)):
                raise ValueError('Unsupported StridedSlice masks')
            begin, end, stride = map(numbers, args[1:4])
            return args[0][tuple(slice(a, b, c) for a, b, c in zip(begin, end, stride))]
        if kind == 'SliceTf':
            begin, size = map(numbers, args[1:3])
            return args[0][tuple(slice(a, None if b < 0 else a + b) for a, b in zip(begin, size))]
        if kind == 'GatherV2':
            axis = int(args[2].item()) if len(args) > 2 else 0
            axis %= args[0].ndim
            indices = args[1].to(torch.int64)
            result = torch.index_select(args[0], axis, indices.reshape(-1))
            shape = (*args[0].shape[:axis], *indices.shape, *args[0].shape[axis + 1:])
            return result.reshape(shape)
        if kind == 'Reduction':
            if get(1, '<B') != 6:
                raise ValueError('Unsupported Reduction operation')
            axes = int_vector(self.data, table, 3)
            if axes:
                raise ValueError('Unresolved explicit reduction axes')
            return args[0].prod().to(args[0].dtype)
        if kind == 'Pack':
            return torch.stack(args, dim=get(0))
        if kind == 'Concat':
            return torch.cat(args, dim=node['axis'])
        if kind == 'Softmax':
            return torch.softmax(args[0], dim=node['axis'])
        if kind == 'UnaryOp' and node['unary_operation'] == 32:
            return F.gelu(args[0], approximate='tanh')
        if kind == 'DepthToSpace':
            if get(1, '<B') != 1:
                raise ValueError('Unsupported DepthToSpace mode')
            return F.pixel_shuffle(args[0], get(0))
        if kind == 'ReLU':
            return F.leaky_relu(args[0], negative_slope=node['negative_slope'])
        if kind == 'Clip':
            return torch.clamp(args[0], min=node['clip']['min'], max=node['clip']['max'])
        if kind == 'Pooling':
            # Native creator 1015e8260 selects AVG for custom slot 7 == 1.
            # Earlier versions incorrectly used slot 7 as ceil and ran MAX.
            # The restoration pools have type=AVG, ceil=slot10, no padding.
            pool = pooling_parameters(self.data, table)
            if pool['pad_type'] or pool['global'] or any(pool['padding']):
                raise ValueError('Unresolved Pooling parameters')
            operation = {0: F.max_pool2d, 1: F.avg_pool2d}[pool['type']]
            return operation(args[0], pool['kernel'], stride=pool['stride'], ceil_mode=pool['ceil'])
        if kind == 'Interp':
            # The two recovered operations have resizeType=2, half-pixel enabled,
            # alignCorners disabled, and dynamic scale input [1,1,2,2].
            if get(5) != 2 or not get(6, '<B') or get(7, '<B'):
                raise ValueError('Unresolved Interp settings')
            scale = args[1].reshape(-1).tolist()
            size = tuple(int(args[0].shape[i + 2] * scale[i + 2]) for i in range(2))
            return F.interpolate(args[0], size=size, mode='bilinear', align_corners=False)
        raise ValueError(f'Unsupported node {kind}')

    @torch.inference_mode()
    def run(self, image, mask, capture_alpha2=False):
        nodes = self.graph['nodes']
        kept_ids = set(self.output_ids)
        if capture_alpha2:
            delta2 = self.graph['tensor_names'].index('2405607714')
            producer = next(n for n in nodes if delta2 in n['outputs'])
            alpha2_id = producer['inputs'][0]
            kept_ids.add(alpha2_id)
        counts = Counter(t for n in nodes for t in n['inputs'])
        values, trace = {}, []
        inputs = [n for n in nodes if n['type_name'] == 'Input']
        for node, array in zip(inputs, [image, mask], strict=True):
            if list(array.shape) != node['dims']:
                raise ValueError(f"Input shape {array.shape} != {node['dims']}")
            values[node['outputs'][0]] = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
        for node in nodes:
            if node['type_name'] == 'Input':
                continue
            args = [values[t] for t in node['inputs']]
            start = time.monotonic()
            try:
                result = self.execute_node(node, args)
            except Exception as error:
                raise RuntimeError(f"Node {node['index']} {node['type_name']}; input shapes "
                                   f"{[list(a.shape) for a in args]}: {error}") from error
            if len(node['outputs']) != 1:
                raise ValueError('Unsupported multi-output operation')
            values[node['outputs'][0]] = result
            trace.append({'node': node['index'], 'kind': node['type_name'],
                          'shape': list(result.shape), 'seconds': time.monotonic() - start})
            for tensor in node['inputs']:
                counts[tensor] -= 1
                if counts[tensor] == 0 and tensor not in kept_ids:
                    values.pop(tensor)
        result = {name: values[self.graph['tensor_names'].index(name)].numpy()
                  for name in self.graph['output_names']}
        if capture_alpha2:
            result['diagnostic_A2'] = values[alpha2_id].numpy()
        return result, trace

