"""Minimal bounds-aware FlatBuffers reader for dunnite graph records."""
import struct

def integer(data, offset, fmt='<I'):
    return struct.unpack_from(fmt, data, offset)[0]

def fields(data, table):
    vtable = table - integer(data, table, '<i')
    length = integer(data, vtable, '<H')
    size = integer(data, vtable + 2, '<H')
    if not 4 <= length <= 256 or length & 1:
        raise ValueError(f'Invalid vtable length at {vtable}')
    return vtable, size, [integer(data, vtable + offset, '<H') for offset in range(4, length, 2)]

def location(data, table, index):
    _, _, offsets = fields(data, table)
    if index >= len(offsets) or offsets[index] == 0:
        return None
    return table + offsets[index]

def scalar(data, table, index, fmt='<I', default=0):
    pos = location(data, table, index)
    return integer(data, pos, fmt) if pos is not None else default

def vector(data, table, index):
    pos = location(data, table, index)
    if pos is None:
        return None, 0
    pos += integer(data, pos)
    count = integer(data, pos)
    if pos + 4 > len(data):
        raise ValueError('Vector out of bounds')
    return pos + 4, count

def int_vector(data, table, index):
    pos, count = vector(data, table, index)
    if pos is None:
        return []
    if pos + 4 * count > len(data):
        raise ValueError('Int vector out of bounds')
    return list(struct.unpack_from(f'<{count}i', data, pos))

def table_target(data, table, index):
    pos = location(data, table, index)
    return pos + integer(data, pos) if pos is not None else None

def string(data, pos):
    pos += integer(data, pos)
    count = integer(data, pos)
    if pos + 4 + count > len(data):
        raise ValueError('String out of bounds')
    return data[pos + 4:pos + 4 + count].decode('utf-8')

def string_vector(data, table, index):
    pos, count = vector(data, table, index)
    return [string(data, pos + i * 4) for i in range(count)] if pos is not None else []

def pooling_parameters(data, table):
    """Native Pool table, verified at 1015e8260, 10166e880, 10166f5b0.

    vtable+0x12 (slot7) selects AVG/MAX; +0x14 (slot8) pad mode;
    +0x16 (slot9) global; +0x1a (slot11) passes count mode to kernel.
    Slot10 is ceilModel. Pool11=1 was previously misread as ceil/type.
    """
    return {'kernel': (scalar(data,table,1),scalar(data,table,0)),
            'stride': (scalar(data,table,3),scalar(data,table,2)),
            'padding': (scalar(data,table,5),scalar(data,table,4)),
            'pads': int_vector(data,table,6),
            'type': scalar(data,table,7,'<B'),
            'pad_type': scalar(data,table,8,'<B'),
            'global': bool(scalar(data,table,9,'<B')),
            'ceil': bool(scalar(data,table,10,'<B')),
            'count_type': scalar(data,table,11,'<B'),
            'data_type': scalar(data,table,12)}
