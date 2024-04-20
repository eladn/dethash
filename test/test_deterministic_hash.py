__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2023-10-18"

import dataclasses
import inspect
import itertools
import os
import pathlib
import subprocess
import tempfile
from typing import Any, Optional, Dict

import numpy as np

from src.deterministic_hash import base64_deterministic_hash


def test_deterministic_base64_hash__numpy_non_finite():
    distinct_items = [
        '',
        None,
        np.nan,
        np.inf,
        -np.inf,
        0,
        1,
        'np.nan'
        'np.inf'
        '-np.inf'
    ]
    hashes = [
        base64_deterministic_hash(item)
        for item in distinct_items
    ]
    assert len(set(hashes)) == len(hashes)  # verify that they're all different


def test_deterministic_base64_hash__numpy_arrays():
    hashes_per_seed: Dict[int, str] = {}
    for seed in range(10):
        for iter_nr in range(1000):
            rng = np.random.RandomState(seed=seed)
            cur_arr = rng.random(100)
            cur_arr_hash = base64_deterministic_hash(cur_arr)
            if iter_nr == 0:
                assert all(cur_arr_hash != other_hash for other_hash in hashes_per_seed.values())
                hashes_per_seed[seed] = cur_arr_hash
            else:
                assert hashes_per_seed[seed] == cur_arr_hash


def test_deterministic_base64_hash__numpy_arrays__multiple_python_processes():
    hashes_per_seed: Dict[int, str] = {}
    for seed in range(3):
        for iter_nr in range(3):
            cur_arr_hash = get_hash_from_other_python_process(seed)
            if iter_nr == 0:
                assert all(cur_arr_hash != other_hash for other_hash in hashes_per_seed.values())
                hashes_per_seed[seed] = cur_arr_hash
            else:
                assert hashes_per_seed[seed] == cur_arr_hash


@dataclasses.dataclass
class MockDataclass:
    field_a: Any
    field_b: Any
    field_c: Any


class RandomFieldValueChanger:
    def __init__(self):
        self._last_field_idx_in_cur_iteration = -1
        self._tot_nr_fields: Optional[int] = None
        self._field_idx_to_change_in_cur_iteration = -1
        self._nr_changed_fields_in_cur_iter = 0

    def finished(self) -> bool:
        return self._tot_nr_fields is not None and self._field_idx_to_change_in_cur_iteration == self._tot_nr_fields

    def finished_iterating_object(self):
        if self._tot_nr_fields is None:
            self._tot_nr_fields = self._last_field_idx_in_cur_iteration + 1
            assert self._nr_changed_fields_in_cur_iter == 0
        else:
            assert self._tot_nr_fields == self._last_field_idx_in_cur_iteration + 1
            assert self._nr_changed_fields_in_cur_iter == 1
        self._last_field_idx_in_cur_iteration = -1
        self._field_idx_to_change_in_cur_iteration += 1
        self._nr_changed_fields_in_cur_iter = 0

    def conditional_change(self, init_val) -> Any:
        MAX_ALLOWED_EPS_FOR_FLOATS = 0.0000001
        self._last_field_idx_in_cur_iteration += 1  # now this counter has the cur field idx
        if not self._should_change_cur_field():
            return init_val
        self._nr_changed_fields_in_cur_iter += 1
        if isinstance(init_val, set):
            if None in init_val:
                return init_val - {None}
            return {None} | init_val
        if isinstance(init_val, tuple):
            if None in init_val:
                return tuple(item for item in init_val if item is not None)
            return (None,) + init_val
        if isinstance(init_val, list):
            if None in init_val:
                return [item for item in init_val if item is not None]
            return [None] + init_val
        if isinstance(init_val, int):
            return init_val + 1
        if isinstance(init_val, float):
            return init_val - MAX_ALLOWED_EPS_FOR_FLOATS
        if isinstance(init_val, str):
            return init_val + '_'
        if init_val is None:
            return 0
        if isinstance(init_val, np.ndarray):
            if init_val.ndim == 0:
                return np.array(init_val.item() + MAX_ALLOWED_EPS_FOR_FLOATS)
            init_val = np.array(init_val)
            init_val[self._last_field_idx_in_cur_iteration % len(init_val)] += MAX_ALLOWED_EPS_FOR_FLOATS
            return init_val

    def _should_change_cur_field(self) -> bool:
        return self._last_field_idx_in_cur_iteration == self._field_idx_to_change_in_cur_iteration


def test_deterministic_base64_hash__structured_obj():
    hashes_per_changed_field: Dict[int, str] = {}
    for cur_iter in range(3):
        cc = RandomFieldValueChanger()
        for changed_field_idx in itertools.count():
            if cc.finished():
                break
            rng = np.random.RandomState(seed=0)
            cur_structured_obj = {
                cc.conditional_change('a'):
                    MockDataclass(
                        field_a=cc.conditional_change(rng.random(10)),
                        field_b=[
                            cc.conditional_change(()),
                            cc.conditional_change('some str'),
                            (cc.conditional_change([]), cc.conditional_change([]))],
                        field_c=cc.conditional_change(None)),
                cc.conditional_change('b'): None,
                cc.conditional_change('c'): (
                    cc.conditional_change('strr'),
                    [
                        cc.conditional_change(1234),
                        cc.conditional_change(6.23421321),
                        cc.conditional_change(None),
                        cc.conditional_change('something'),
                        cc.conditional_change(())
                    ]),
                cc.conditional_change('d'): {
                    cc.conditional_change(1),
                    cc.conditional_change('u'),
                    cc.conditional_change(None),
                    cc.conditional_change(()),
                    (cc.conditional_change(5), cc.conditional_change(5))}}
            cc.finished_iterating_object()
            cur_arr_hash = base64_deterministic_hash(cur_structured_obj)
            if cur_iter == 0:
                assert changed_field_idx not in hashes_per_changed_field
                assert all(cur_arr_hash != other_hash for other_hash in hashes_per_changed_field.values())
                hashes_per_changed_field[changed_field_idx] = cur_arr_hash
            else:
                assert hashes_per_changed_field[changed_field_idx] == cur_arr_hash


def get_hash_from_other_python_process(rng_seed: int):
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tmp_file_path = os.path.join(tmp_dir_path, 'res.txt')
        cwd = pathlib.Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), f'..{os.sep}')).resolve()
        import_rel_path = os.path.relpath(
            pathlib.Path(inspect.getfile(base64_deterministic_hash)).expanduser().resolve(),
            cwd,
        )
        import_rel_path = import_rel_path.replace(os.sep, '.')
        assert import_rel_path[-len('.py'):] == '.py'
        import_rel_path = import_rel_path[:-len('.py')]
        hash_fn = base64_deterministic_hash.__name__
        escaped_file_path = tmp_file_path.replace("\\", "\\\\")
        code = f'from {import_rel_path} import {hash_fn}; ' \
               f'import numpy as np; ' \
               f'rng = np.random.RandomState(seed={rng_seed}); ' \
               f'arr = rng.random(100); ' \
               f'f = open(\'{escaped_file_path}\', \'w\'); ' \
               f'f.write({hash_fn}(arr)); ' \
               f'f.close()'
        cmd = ['python', '-c', f'{code}']
        run_res = subprocess.run(cmd, cwd=cwd)
        assert run_res.returncode == 0
        with open(tmp_file_path, 'r') as tmp_file:
            tmp_file.seek(0)
            return tmp_file.read()


def test_deterministic_base64_hash_unordered_literal_dict():
    hash1 = base64_deterministic_hash({'a': 1, 'c': 3, 'b': 2})
    hash2 = base64_deterministic_hash({'c': 3, 'b': 2, 'a': 1})
    assert hash1 == hash2


def test_deterministic_base64_hash_unordered_set():
    rng = np.random.RandomState(seed=0)
    set_hash = None
    for perm_idx in range(10):
        perm = rng.permutation(100)
        cur_set = set()
        for item in perm:
            cur_set.add(item)
        cur_set_hash = base64_deterministic_hash(cur_set)
        if perm_idx == 0:
            set_hash = cur_set_hash
        assert set_hash == cur_set_hash


def test_deterministic_base64_hash_unordered_dict():
    rng = np.random.RandomState(seed=0)
    dict_hash = None
    for perm_idx in range(10):
        perm = rng.permutation(100)
        cur_dict = {}
        for item in perm:
            cur_dict[item] = item
        cur_dict_hash = base64_deterministic_hash(cur_dict)
        if perm_idx == 0:
            dict_hash = cur_dict_hash
        assert dict_hash == cur_dict_hash


if __name__ == '__main__':
    test_deterministic_base64_hash__numpy_non_finite()
    test_deterministic_base64_hash__numpy_arrays()
    test_deterministic_base64_hash__numpy_arrays__multiple_python_processes()
    test_deterministic_base64_hash__structured_obj()
    test_deterministic_base64_hash_unordered_literal_dict()
    test_deterministic_base64_hash_unordered_set()
    test_deterministic_base64_hash_unordered_dict()
