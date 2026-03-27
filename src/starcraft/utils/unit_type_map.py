"""Global mapping from SC2 unit_type_id to contiguous indices.

Indices:
    0 = Empty (no unit)
    1 = Unknown (unmapped unit type)
    2..NUM_UNIT_TYPES-1 = known unit types, sorted by raw SC2 API ID

Source: pysc2.lib.units from the installed PySC2 package.
Reference: https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/units.py
This static ID table was generated from PySC2 to avoid a runtime dependency on
`pysc2` in this repo. It intentionally follows PySC2's unit classification,
including SC2 ID 135 = Protoss.ForceField.
"""

import numpy as np

# Static SC2 API unit type IDs vendored from PySC2, grouped to match the
# upstream pysc2.lib.units layout for easier manual comparison.
_NEUTRAL_IDS = [
    146, 147, 149, 321, 322, 324, 330, 335, 336, 341, 342, 343, 344, 350,
    364, 365, 371, 372, 373, 376, 377, 472, 473, 474, 475, 483, 485, 486,
    487, 490, 517, 518, 559, 560, 561, 562, 563, 564, 588, 589, 590, 591,
    608, 609, 610, 612, 628, 629, 630, 638, 639, 640, 641, 642, 643, 648,
    649, 651, 661, 662, 663, 664, 665, 666, 796, 797, 877, 880, 881, 884,
    885, 886, 887, 1904, 1908, 1957, 1958, 1961,
]

_PROTOSS_IDS = [
    4, 10, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 133, 135, 136, 141, 311,
    488, 495, 496, 694, 732, 733, 801, 894, 1910, 1911, 1955,
]

_TERRAN_IDS = [
    5, 6, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 130, 132, 134, 144, 145, 268, 484, 498,
    500, 689, 691, 692, 734, 830, 1913, 1960,
]

_ZERG_IDS = [
    7, 8, 9, 12, 13, 14, 15, 16, 17, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 125, 126, 127,
    128, 129, 131, 137, 138, 139, 140, 142, 143, 150, 151, 289, 489, 493,
    494, 499, 501, 502, 503, 504, 687, 688, 690, 693, 824, 892, 893, 1912,
    1956,
]

EMPTY_INDEX = 0
UNKNOWN_INDEX = 1

_ALL_SC2_IDS = sorted(set(_NEUTRAL_IDS + _PROTOSS_IDS + _TERRAN_IDS + _ZERG_IDS))

SC2_ID_TO_INDEX: dict[int, int] = {
    sc2_id: idx + 2 for idx, sc2_id in enumerate(_ALL_SC2_IDS)
}

NUM_UNIT_TYPES: int = len(_ALL_SC2_IDS) + 2  # +2 for Empty and Unknown


def remap_unit_type(raw_ids: np.ndarray) -> np.ndarray:
    """Map raw SC2 unit_type_id values to contiguous indices.

    Args:
        raw_ids: Array of raw SC2 unit type IDs (any shape).

    Returns:
        Array of same shape with contiguous indices.
        Unknown IDs map to UNKNOWN_INDEX (1).
    """
    out = np.full_like(raw_ids, fill_value=UNKNOWN_INDEX, dtype=np.int64)
    for sc2_id, idx in SC2_ID_TO_INDEX.items():
        out[raw_ids == sc2_id] = idx
    return out
