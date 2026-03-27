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


def test_moving_unit_type_examples_and_edge_cases():
    for sc2_id in (
        48,    # Marine
        45,    # SCV
        73,    # Zealot
        105,   # Zergling
        32,    # SiegeTankSieged
        33,    # SiegeTank
        734,   # LiberatorAG
        500,   # WidowMineBurrowed
        136,   # WarpPrismPhasing
        1911,  # ObserverSurveillanceMode
        1912,  # OverseerOversightMode
        36,    # CommandCenterFlying
        85,    # Interceptor
        7,     # InfestedTerran
        12,    # Changeling
        13,    # ChangelingZealot
        14,    # ChangelingMarineShield
        15,    # ChangelingMarine
        16,    # ChangelingZerglingWings
        17,    # ChangelingZergling
        125,   # QueenBurrowed
        139,   # SpineCrawlerUprooted
        140,   # SporeCrawlerUprooted
    ):
        assert unit_type_map.is_moving_unit_type(sc2_id)
        assert not unit_type_map.is_static_unit_type(sc2_id)


def test_static_unit_type_examples_and_edge_cases():
    for sc2_id in (
        18,    # CommandCenter
        21,    # Barracks
        59,    # Nexus
        86,    # Hatchery
        146,   # MineralField
        342,   # VespeneGeyser
        149,   # XelNagaTower
        58,    # Nuke
        130,   # PlanetaryFortress
        142,   # NydusCanal
        1913,  # RepairDrone
        31,    # AutoTurret
        11,    # PointDefenseDrone
        830,   # KD8Charge
        135,   # ForceField
        151,   # Larva
        103,   # Cocoon
        8,     # BanelingCocoon
        113,   # BroodLordCocoon
        687,   # RavagerCocoon
        501,   # LurkerCocoon
        892,   # OverlordTransportCocoon
        128,   # OverseerCocoon
        150,   # InfestedTerranCocoon
        824,   # ParasiticBombDummy
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
