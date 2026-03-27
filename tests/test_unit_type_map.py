import numpy as np

from src.starcraft.utils import unit_type_map


def test_unit_type_map_constants_and_known_ids():
    assert unit_type_map.EMPTY_INDEX == 0
    assert unit_type_map.UNKNOWN_INDEX == 1
    assert unit_type_map.NUM_UNIT_TYPES == len(unit_type_map.SC2_ID_TO_INDEX) + 2

    for sc2_id in (48, 73, 105, 135, 146):
        assert unit_type_map.SC2_ID_TO_INDEX[sc2_id] > unit_type_map.UNKNOWN_INDEX


def test_unknown_ids_map_to_unknown_index():
    raw_ids = np.array([-1, 5000], dtype=np.int64)

    mapped = unit_type_map.remap_unit_type(raw_ids)

    assert np.array_equal(
        mapped,
        np.array(
            [unit_type_map.UNKNOWN_INDEX, unit_type_map.UNKNOWN_INDEX],
            dtype=np.int64,
        ),
    )


def test_remap_unit_type_preserves_shape_and_dtype():
    raw_ids = np.array([[48, 5000], [135, -1]], dtype=np.int32)

    mapped = unit_type_map.remap_unit_type(raw_ids)

    assert mapped.shape == raw_ids.shape
    assert mapped.dtype == np.int64
    assert np.array_equal(
        mapped,
        np.array(
            [
                [
                    unit_type_map.SC2_ID_TO_INDEX[48],
                    unit_type_map.UNKNOWN_INDEX,
                ],
                [
                    unit_type_map.SC2_ID_TO_INDEX[135],
                    unit_type_map.UNKNOWN_INDEX,
                ],
            ],
            dtype=np.int64,
        ),
    )
