__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2023-10-18"

import base64
import functools
import hashlib
from typing import Set, Optional, NamedTuple, Dict, OrderedDict, Union, Type, Tuple
import dataclasses

import numpy as np


HashableType = Union[Tuple, NamedTuple, Set, Dict, OrderedDict, Type, np.ndarray, str, int, float, bool, bytes]
CanonicType = Union[Tuple, NamedTuple, str, int, float, bool, bytes]
HashType = str
DEFAULT_PRECISION_FOR_FLOATS_ROUND = 7


def base64_deterministic_hash(
        obj: HashableType,
        precision_for_floats_round: Optional[int] = DEFAULT_PRECISION_FOR_FLOATS_ROUND,
        base_hasher = hashlib.blake2b,
) -> HashType:
    """
    Calculates a deterministic hash for the given object.
    Agnostic to small numerical floating representation errors (up to some defined) by rounding floats to a pre-defined
    precision. Traverses the potentially structured input object and converts the structure to a deterministically
    serializable one. For unordered collections sorts the items (if dict - by key) before the hashing. The obtained
    hash is deterministic in the sense that the same object would have the exactly same hash value across different
    executions and different machines.
    :param obj: An object to hash. Can be structured with collections like list, tuple, dict, set, dataclasses or numpy
                arrays. Apart from the structure, should contain primitive hashable types (like int, str, bytes, bool,
                float)
    :param precision_for_floats_round: Number of digits to consider for a floating value. If set to `None` keeps the
                                       value as-is without rounding.
    :return: base64 hash string; eg: `Wp2HTUdK-mqEpLKdP96_X4v0cTk=`
    """
    # Note that, for example, a simple python's native hash(string) result is execution-dependent. However, using
    # hashlib's hashing strategy creates an execution-independent reproducible hashes.
    # To create the hash, we first convert the object to a more-primitive representation (converting all collections to
    # sorted tuples and having only primitive non-structured types), and then create a bytes-array representation from
    # this structure using `repr()` (which is deterministic across machines), as hashlib's hashing strategies require
    # a bytes-array as input to digest.
    obj = canonize_structured_obj(obj, precision_for_floats_round=precision_for_floats_round)
    repr_as_bytes = repr(obj).encode(encoding='utf-8')
    # The created hash (hashlib's output) is a bytes-array. We use base64 encoding to create a stringified
    # representation of this hash (to be used as human-readable keys in DB or file-names).
    return base64.urlsafe_b64encode(base_hasher(repr_as_bytes).digest()).decode('ascii')


class CanonicCollection(NamedTuple):
    orig_type_name: str
    as_tuple: tuple


def canonize_structured_obj(
        obj: HashableType,
        precision_for_floats_round: Optional[int] = DEFAULT_PRECISION_FOR_FLOATS_ROUND,
) -> CanonicType:
    """
    Traverses the object recursively, sorting unordered collections, and transforming dataclass objects to tuple
    of (field-name, value) pairs ordered by the field name. Additionally, converts numpy arrays into tuples of the
    inner items (then recursively convert the inner items of course). Raises exception on encountering other types.
    Floating values are being determinized by rounding into a fixed given numerical precision. Assumes that there are
    no cyclic pointers in the structures.
    """

    def sort_compare_fn(a: CanonicType, b: CanonicType) -> int:
        if type(a) is not type(b):
            return -1 if type(a).__name__ < type(b).__name__ else 1
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return len(a) < len(b)
            for item_a, item_b in zip(a, b):
                inner_cmp = sort_compare_fn(item_a, item_b)
                if inner_cmp != 0:
                    return inner_cmp
            return 0
        return -1 if a < b else 1

    visited_objects_ids: Set[int] = set()

    # We use the following aux inner function is to allow sharing variables in local scope during the entire recursion.
    def recursive_various_containers_to_tuples(cur_obj: HashableType) -> CanonicType:
        # We use the following aux inner function to allow returning the converted object in multiple return statements,
        # without repeating the code of removal from `visited_objects_ids` before every return statement.
        def convert_object(item: HashableType) -> CanonicType:
            if isinstance(item, (list, tuple)):
                return tuple(recursive_various_containers_to_tuples(inner_obj) for inner_obj in item)
            if isinstance(item, set):
                item = [recursive_various_containers_to_tuples(inner_obj) for inner_obj in item]
                item.sort(key=functools.cmp_to_key(sort_compare_fn))
                return tuple(item)
            if dataclasses.is_dataclass(item):
                item = dataclasses.asdict(item)
            if isinstance(item, dict):
                item = [
                    (recursive_various_containers_to_tuples(k), recursive_various_containers_to_tuples(v))
                    for k, v in item.items()]
                item.sort(key=functools.cmp_to_key(lambda a, b: sort_compare_fn(a[0], b[0])))
                return tuple(item)
            if isinstance(item, np.ndarray):
                if item.ndim == 0:
                    return recursive_various_containers_to_tuples(item.item())
                else:
                    # We canonize a numpy array by converting it to a pair of its shape and its values flattened.
                    shape_and_flattened_arr = (
                        tuple(item.shape),
                        tuple(recursive_various_containers_to_tuples(inner_obj) for inner_obj in item))
                    return shape_and_flattened_arr
            if isinstance(item, (float, np.float, np.float32, np.float64)) or \
                    (np.isscalar(item) and hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.floating)):
                if precision_for_floats_round is None:
                    return float(item)
                if not np.isfinite(item):
                    return item
                return int(round(item, ndigits=precision_for_floats_round) * (10 ** precision_for_floats_round))
            if item is None:
                return None
            ALLOWED_PRIMITIVE_TYPES = (str, np.str, int, np.int, np.int32, np.int64, bool, np.bool, bytes, np.string_)
            if not (isinstance(item, ALLOWED_PRIMITIVE_TYPES) or
                    (np.isscalar(item) and hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.integer))):
                raise ValueError(f'Hashing does not support type `{type(item)}`.')
            return item

        if id(cur_obj) in visited_objects_ids:
            # TODO: Add support for cyclic referencing!
            raise ValueError(f'{canonize_structured_obj.__name__}() does not support cyclic referencing.')
        visited_objects_ids.add(id(cur_obj))
        converted_object = convert_object(cur_obj)
        # Add the orig typing info
        if isinstance(converted_object, tuple):
            if hasattr(cur_obj, '__class__'):
                orig_type_name = cur_obj.__class__.__name__
            elif hasattr(type(cur_obj), '__name__'):
                orig_type_name = type(cur_obj).__name__
            else:
                orig_type_name = str(type(cur_obj))
            converted_object = CanonicCollection(orig_type_name=orig_type_name, as_tuple=converted_object)
        visited_objects_ids.remove(id(cur_obj))
        return converted_object

    tupled_deterministically_ordered_representation = recursive_various_containers_to_tuples(obj)
    return tupled_deterministically_ordered_representation
