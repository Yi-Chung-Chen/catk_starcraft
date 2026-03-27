import numpy as np

from src.starcraft.utils import unit_type_map


def test_unit_type_map_constants_and_known_ids():
    assert not hasattr(unit_type_map, "EMPTY_INDEX")
    assert unit_type_map.UNKNOWN_INDEX == 0
    assert unit_type_map.NUM_UNIT_TYPES == len(unit_type_map.SC2_ID_TO_INDEX) + 1

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


def test_moving_unit_type_examples_and_edge_cases():
    for sc2_id in (
        48, 45, 73, 105, 32, 33, 734, 500, 136, 1911, 1912, 36, 85, 7,
        12, 13, 14, 15, 16, 17, 125, 139, 140,
    ):
        assert unit_type_map.is_moving_unit_type(sc2_id)
        assert not unit_type_map.is_static_unit_type(sc2_id)


def test_static_unit_type_examples_and_edge_cases():
    for sc2_id in (
        18, 21, 59, 86, 146, 342, 149, 58, 130, 142, 1913, 31, 11, 830,
        135, 151, 103, 8, 113, 687, 501, 892, 128, 150, 824,
    ):
        assert unit_type_map.is_static_unit_type(sc2_id)
        assert not unit_type_map.is_moving_unit_type(sc2_id)


def test_unknown_ids_are_neither_moving_nor_static():
    for sc2_id in (-1, 5000):
        assert not unit_type_map.is_moving_unit_type(sc2_id)
        assert not unit_type_map.is_static_unit_type(sc2_id)


def test_moving_and_static_sets_do_not_overlap():
    assert not (
        unit_type_map.MOVING_UNIT_TYPE_IDS & unit_type_map.STATIC_UNIT_TYPE_IDS
    )


def test_named_lookup_helpers():
    assert unit_type_map.get_unit_name(48) == "Marine"
    assert unit_type_map.get_unit_race(48) == "Terran"
    assert unit_type_map.describe_unit_type(48) == "Terran.Marine"
    assert unit_type_map.describe_unit_type(135) == "Protoss.ForceField"
    assert unit_type_map.describe_unit_type(5000) == "Unknown(5000)"
