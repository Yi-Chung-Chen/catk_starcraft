"""Global mapping from SC2 unit_type_id to contiguous indices.

Indices:
    0 = Empty (no unit)
    1 = Unknown (unmapped unit type)
    2..NUM_UNIT_TYPES-1 = known unit types, sorted by raw SC2 API ID

Source: pysc2.lib.units from the installed PySC2 package.
Reference: https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/units.py
This static metadata was generated from PySC2 to avoid a runtime dependency on
`pysc2` in this repo. It intentionally follows PySC2's unit classification,
including SC2 ID 135 = Protoss.ForceField.
"""

import numpy as np


def _ids(unit_specs):
    return [unit_id for _, unit_id in unit_specs]


# Static SC2 API unit type IDs vendored from PySC2, grouped to match the
# upstream pysc2.lib.units layout for easier manual comparison.
_NEUTRAL_UNITS = [
    ("BattleStationMineralField", 886),
    ("BattleStationMineralField750", 887),
    ("CarrionBird", 322),
    ("CleaningBot", 612),
    ("CollapsibleRockTower", 609),
    ("CollapsibleRockTowerDebris", 490),
    ("CollapsibleRockTowerDebrisRampLeft", 518),
    ("CollapsibleRockTowerDebrisRampRight", 517),
    ("CollapsibleRockTowerDiagonal", 588),
    ("CollapsibleRockTowerPushUnit", 561),
    ("CollapsibleRockTowerPushUnitRampLeft", 564),
    ("CollapsibleRockTowerPushUnitRampRight", 563),
    ("CollapsibleRockTowerRampLeft", 664),
    ("CollapsibleRockTowerRampRight", 663),
    ("CollapsibleTerranTower", 610),
    ("CollapsibleTerranTowerDebris", 485),
    ("CollapsibleTerranTowerDiagonal", 589),
    ("CollapsibleTerranTowerPushUnit", 562),
    ("CollapsibleTerranTowerPushUnitRampLeft", 559),
    ("CollapsibleTerranTowerPushUnitRampRight", 560),
    ("CollapsibleTerranTowerRampLeft", 590),
    ("CollapsibleTerranTowerRampRight", 591),
    ("Crabeetle", 662),
    ("Debris2x2NonConjoined", 475),
    ("DebrisRampLeft", 486),
    ("DebrisRampRight", 487),
    ("DestructibleBillboardTall", 350),
    ("DestructibleCityDebris4x4", 628),
    ("DestructibleCityDebris6x6", 629),
    ("DestructibleCityDebrisHugeDiagonalBLUR", 630),
    ("DestructibleDebris4x4", 364),
    ("DestructibleDebris6x6", 365),
    ("DestructibleDebrisRampDiagonalHugeBLUR", 377),
    ("DestructibleDebrisRampDiagonalHugeULBR", 376),
    ("DestructibleIce4x4", 648),
    ("DestructibleIce6x6", 649),
    ("DestructibleIceDiagonalHugeBLUR", 651),
    ("DestructibleRampDiagonalHugeBLUR", 373),
    ("DestructibleRampDiagonalHugeULBR", 372),
    ("DestructibleRock6x6", 371),
    ("DestructibleRockEx14x4", 638),
    ("DestructibleRockEx16x6", 639),
    ("DestructibleRockEx1DiagonalHugeBLUR", 641),
    ("DestructibleRockEx1DiagonalHugeULBR", 640),
    ("DestructibleRockEx1HorizontalHuge", 643),
    ("DestructibleRockEx1VerticalHuge", 642),
    ("Dog", 336),
    ("InhibitorZoneMedium", 1958),
    ("InhibitorZoneSmall", 1957),
    ("KarakFemale", 324),
    ("LabBot", 661),
    ("LabMineralField", 665),
    ("LabMineralField750", 666),
    ("Lyote", 321),
    ("MineralField", 341),
    ("MineralField450", 1961),
    ("MineralField750", 483),
    ("ProtossVespeneGeyser", 608),
    ("PurifierMineralField", 884),
    ("PurifierMineralField750", 885),
    ("PurifierRichMineralField", 796),
    ("PurifierRichMineralField750", 797),
    ("PurifierVespeneGeyser", 880),
    ("ReptileCrate", 877),
    ("RichMineralField", 146),
    ("RichMineralField750", 147),
    ("RichVespeneGeyser", 344),
    ("Scantipede", 335),
    ("ShakurasVespeneGeyser", 881),
    ("SpacePlatformGeyser", 343),
    ("UnbuildableBricksDestructible", 473),
    ("UnbuildablePlatesDestructible", 474),
    ("UnbuildableRocksDestructible", 472),
    ("UtilityBot", 330),
    ("VespeneGeyser", 342),
    ("XelNagaDestructibleBlocker8NE", 1904),
    ("XelNagaDestructibleBlocker8SW", 1908),
    ("XelNagaTower", 149),
]

_PROTOSS_UNITS = [
    ("Adept", 311),
    ("AdeptPhaseShift", 801),
    ("Archon", 141),
    ("Assimilator", 61),
    ("AssimilatorRich", 1955),
    ("Carrier", 79),
    ("Colossus", 4),
    ("CyberneticsCore", 72),
    ("DarkShrine", 69),
    ("DarkTemplar", 76),
    ("Disruptor", 694),
    ("DisruptorPhased", 733),
    ("FleetBeacon", 64),
    ("ForceField", 135),
    ("Forge", 63),
    ("Gateway", 62),
    ("HighTemplar", 75),
    ("Immortal", 83),
    ("Interceptor", 85),
    ("Mothership", 10),
    ("MothershipCore", 488),
    ("Nexus", 59),
    ("Observer", 82),
    ("ObserverSurveillanceMode", 1911),
    ("Oracle", 495),
    ("Phoenix", 78),
    ("PhotonCannon", 66),
    ("Probe", 84),
    ("Pylon", 60),
    ("PylonOvercharged", 894),
    ("RoboticsBay", 70),
    ("RoboticsFacility", 71),
    ("Sentry", 77),
    ("ShieldBattery", 1910),
    ("Stalker", 74),
    ("Stargate", 67),
    ("StasisTrap", 732),
    ("Tempest", 496),
    ("TemplarArchive", 68),
    ("TwilightCouncil", 65),
    ("VoidRay", 80),
    ("WarpGate", 133),
    ("WarpPrism", 81),
    ("WarpPrismPhasing", 136),
    ("Zealot", 73),
]

_TERRAN_UNITS = [
    ("Armory", 29),
    ("AutoTurret", 31),
    ("Banshee", 55),
    ("Barracks", 21),
    ("BarracksFlying", 46),
    ("BarracksReactor", 38),
    ("BarracksTechLab", 37),
    ("Battlecruiser", 57),
    ("Bunker", 24),
    ("CommandCenter", 18),
    ("CommandCenterFlying", 36),
    ("Cyclone", 692),
    ("EngineeringBay", 22),
    ("Factory", 27),
    ("FactoryFlying", 43),
    ("FactoryReactor", 40),
    ("FactoryTechLab", 39),
    ("FusionCore", 30),
    ("Ghost", 50),
    ("GhostAcademy", 26),
    ("GhostAlternate", 144),
    ("GhostNova", 145),
    ("Hellion", 53),
    ("Hellbat", 484),
    ("KD8Charge", 830),
    ("Liberator", 689),
    ("LiberatorAG", 734),
    ("MULE", 268),
    ("Marauder", 51),
    ("Marine", 48),
    ("Medivac", 54),
    ("MissileTurret", 23),
    ("Nuke", 58),
    ("OrbitalCommand", 132),
    ("OrbitalCommandFlying", 134),
    ("PlanetaryFortress", 130),
    ("PointDefenseDrone", 11),
    ("Raven", 56),
    ("Reactor", 6),
    ("Reaper", 49),
    ("Refinery", 20),
    ("RefineryRich", 1960),
    ("RepairDrone", 1913),
    ("SCV", 45),
    ("SensorTower", 25),
    ("SiegeTank", 33),
    ("SiegeTankSieged", 32),
    ("Starport", 28),
    ("StarportFlying", 44),
    ("StarportReactor", 42),
    ("StarportTechLab", 41),
    ("SupplyDepot", 19),
    ("SupplyDepotLowered", 47),
    ("TechLab", 5),
    ("Thor", 52),
    ("ThorHighImpactMode", 691),
    ("VikingAssault", 34),
    ("VikingFighter", 35),
    ("WidowMine", 498),
    ("WidowMineBurrowed", 500),
]

_ZERG_UNITS = [
    ("Baneling", 9),
    ("BanelingBurrowed", 115),
    ("BanelingCocoon", 8),
    ("BanelingNest", 96),
    ("BroodLord", 114),
    ("BroodLordCocoon", 113),
    ("Broodling", 289),
    ("BroodlingEscort", 143),
    ("Changeling", 12),
    ("ChangelingMarine", 15),
    ("ChangelingMarineShield", 14),
    ("ChangelingZealot", 13),
    ("ChangelingZergling", 17),
    ("ChangelingZerglingWings", 16),
    ("Cocoon", 103),
    ("Corruptor", 112),
    ("CreepTumor", 87),
    ("CreepTumorBurrowed", 137),
    ("CreepTumorQueen", 138),
    ("Drone", 104),
    ("DroneBurrowed", 116),
    ("EvolutionChamber", 90),
    ("Extractor", 88),
    ("ExtractorRich", 1956),
    ("GreaterSpire", 102),
    ("Hatchery", 86),
    ("Hive", 101),
    ("Hydralisk", 107),
    ("HydraliskBurrowed", 117),
    ("HydraliskDen", 91),
    ("InfestationPit", 94),
    ("InfestedTerran", 7),
    ("InfestedTerranBurrowed", 120),
    ("InfestedTerranCocoon", 150),
    ("Infestor", 111),
    ("InfestorBurrowed", 127),
    ("Lair", 100),
    ("Larva", 151),
    ("Locust", 489),
    ("LocustFlying", 693),
    ("Lurker", 502),
    ("LurkerBurrowed", 503),
    ("LurkerCocoon", 501),
    ("LurkerDen", 504),
    ("Mutalisk", 108),
    ("NydusCanal", 142),
    ("NydusNetwork", 95),
    ("Overlord", 106),
    ("OverlordTransport", 893),
    ("OverlordTransportCocoon", 892),
    ("Overseer", 129),
    ("OverseerCocoon", 128),
    ("OverseerOversightMode", 1912),
    ("ParasiticBombDummy", 824),
    ("Queen", 126),
    ("QueenBurrowed", 125),
    ("Ravager", 688),
    ("RavagerBurrowed", 690),
    ("RavagerCocoon", 687),
    ("Roach", 110),
    ("RoachBurrowed", 118),
    ("RoachWarren", 97),
    ("SpawningPool", 89),
    ("SpineCrawler", 98),
    ("SpineCrawlerUprooted", 139),
    ("Spire", 92),
    ("SporeCrawler", 99),
    ("SporeCrawlerUprooted", 140),
    ("SwarmHost", 494),
    ("SwarmHostBurrowed", 493),
    ("Ultralisk", 109),
    ("UltraliskBurrowed", 131),
    ("UltraliskCavern", 93),
    ("Viper", 499),
    ("Zergling", 105),
    ("ZerglingBurrowed", 119),
]

# Repo-maintained semantic metadata built on top of the vendored PySC2 IDs.
# This is not a direct PySC2 API export. Important edge cases:
# - ForceField is static.
# - Flying Terran structures are moving, but PlanetaryFortress is static.
# - Uprooted Spine/Spore Crawlers are moving.
# - Mode variants of fundamentally mobile units stay moving.
# - NydusCanal is static.
_MOVING_TERRAN_UNITS = [
    ("SiegeTankSieged", 32),
    ("SiegeTank", 33),
    ("VikingAssault", 34),
    ("VikingFighter", 35),
    ("CommandCenterFlying", 36),
    ("FactoryFlying", 43),
    ("StarportFlying", 44),
    ("SCV", 45),
    ("BarracksFlying", 46),
    ("Marine", 48),
    ("Reaper", 49),
    ("Ghost", 50),
    ("Marauder", 51),
    ("Thor", 52),
    ("Hellion", 53),
    ("Medivac", 54),
    ("Banshee", 55),
    ("Raven", 56),
    ("Battlecruiser", 57),
    ("OrbitalCommandFlying", 134),
    ("GhostAlternate", 144),
    ("GhostNova", 145),
    ("MULE", 268),
    ("Hellbat", 484),
    ("WidowMine", 498),
    ("WidowMineBurrowed", 500),
    ("Liberator", 689),
    ("ThorHighImpactMode", 691),
    ("Cyclone", 692),
    ("LiberatorAG", 734),
]

_MOVING_PROTOSS_UNITS = [
    ("Colossus", 4),
    ("Mothership", 10),
    ("Zealot", 73),
    ("Stalker", 74),
    ("HighTemplar", 75),
    ("DarkTemplar", 76),
    ("Sentry", 77),
    ("Phoenix", 78),
    ("Carrier", 79),
    ("VoidRay", 80),
    ("WarpPrism", 81),
    ("Observer", 82),
    ("Immortal", 83),
    ("Probe", 84),
    ("Interceptor", 85),
    ("WarpPrismPhasing", 136),
    ("Archon", 141),
    ("Adept", 311),
    ("MothershipCore", 488),
    ("Oracle", 495),
    ("Tempest", 496),
    ("Disruptor", 694),
    ("DisruptorPhased", 733),
    ("AdeptPhaseShift", 801),
    ("ObserverSurveillanceMode", 1911),
]

_MOVING_ZERG_UNITS = [
    ("InfestedTerran", 7),
    ("Baneling", 9),
    ("Changeling", 12),
    ("ChangelingZealot", 13),
    ("ChangelingMarineShield", 14),
    ("ChangelingMarine", 15),
    ("ChangelingZerglingWings", 16),
    ("ChangelingZergling", 17),
    ("Drone", 104),
    ("Zergling", 105),
    ("Overlord", 106),
    ("Hydralisk", 107),
    ("Mutalisk", 108),
    ("Ultralisk", 109),
    ("Roach", 110),
    ("Infestor", 111),
    ("Corruptor", 112),
    ("BroodLord", 114),
    ("BanelingBurrowed", 115),
    ("DroneBurrowed", 116),
    ("HydraliskBurrowed", 117),
    ("RoachBurrowed", 118),
    ("ZerglingBurrowed", 119),
    ("InfestedTerranBurrowed", 120),
    ("QueenBurrowed", 125),
    ("Queen", 126),
    ("InfestorBurrowed", 127),
    ("Overseer", 129),
    ("UltraliskBurrowed", 131),
    ("SpineCrawlerUprooted", 139),
    ("SporeCrawlerUprooted", 140),
    ("BroodlingEscort", 143),
    ("Broodling", 289),
    ("Locust", 489),
    ("SwarmHostBurrowed", 493),
    ("SwarmHost", 494),
    ("Viper", 499),
    ("Lurker", 502),
    ("LurkerBurrowed", 503),
    ("Ravager", 688),
    ("RavagerBurrowed", 690),
    ("LocustFlying", 693),
    ("OverlordTransport", 893),
    ("OverseerOversightMode", 1912),
]

_STATIC_TERRAN_UNITS = [
    ("TechLab", 5),
    ("Reactor", 6),
    ("PointDefenseDrone", 11),
    ("CommandCenter", 18),
    ("SupplyDepot", 19),
    ("Refinery", 20),
    ("Barracks", 21),
    ("EngineeringBay", 22),
    ("MissileTurret", 23),
    ("Bunker", 24),
    ("SensorTower", 25),
    ("GhostAcademy", 26),
    ("Factory", 27),
    ("Starport", 28),
    ("Armory", 29),
    ("FusionCore", 30),
    ("AutoTurret", 31),
    ("BarracksTechLab", 37),
    ("BarracksReactor", 38),
    ("FactoryTechLab", 39),
    ("FactoryReactor", 40),
    ("StarportTechLab", 41),
    ("StarportReactor", 42),
    ("SupplyDepotLowered", 47),
    ("Nuke", 58),
    ("PlanetaryFortress", 130),
    ("OrbitalCommand", 132),
    ("KD8Charge", 830),
    ("RepairDrone", 1913),
    ("RefineryRich", 1960),
]

_STATIC_PROTOSS_UNITS = [
    ("Nexus", 59),
    ("Pylon", 60),
    ("Assimilator", 61),
    ("Gateway", 62),
    ("Forge", 63),
    ("FleetBeacon", 64),
    ("TwilightCouncil", 65),
    ("PhotonCannon", 66),
    ("Stargate", 67),
    ("TemplarArchive", 68),
    ("DarkShrine", 69),
    ("RoboticsBay", 70),
    ("RoboticsFacility", 71),
    ("CyberneticsCore", 72),
    ("WarpGate", 133),
    ("ForceField", 135),
    ("StasisTrap", 732),
    ("PylonOvercharged", 894),
    ("ShieldBattery", 1910),
    ("AssimilatorRich", 1955),
]

_STATIC_ZERG_UNITS = [
    ("BanelingCocoon", 8),
    ("Hatchery", 86),
    ("CreepTumor", 87),
    ("Extractor", 88),
    ("SpawningPool", 89),
    ("EvolutionChamber", 90),
    ("HydraliskDen", 91),
    ("Spire", 92),
    ("UltraliskCavern", 93),
    ("InfestationPit", 94),
    ("NydusNetwork", 95),
    ("BanelingNest", 96),
    ("RoachWarren", 97),
    ("SpineCrawler", 98),
    ("SporeCrawler", 99),
    ("Lair", 100),
    ("Hive", 101),
    ("GreaterSpire", 102),
    ("Cocoon", 103),
    ("BroodLordCocoon", 113),
    ("OverseerCocoon", 128),
    ("CreepTumorBurrowed", 137),
    ("CreepTumorQueen", 138),
    ("NydusCanal", 142),
    ("InfestedTerranCocoon", 150),
    ("Larva", 151),
    ("LurkerCocoon", 501),
    ("LurkerDen", 504),
    ("RavagerCocoon", 687),
    ("ParasiticBombDummy", 824),
    ("OverlordTransportCocoon", 892),
    ("ExtractorRich", 1956),
]

_STATIC_NEUTRAL_UNITS = list(_NEUTRAL_UNITS)

_NEUTRAL_IDS = _ids(_NEUTRAL_UNITS)
_PROTOSS_IDS = _ids(_PROTOSS_UNITS)
_TERRAN_IDS = _ids(_TERRAN_UNITS)
_ZERG_IDS = _ids(_ZERG_UNITS)

EMPTY_INDEX = 0
UNKNOWN_INDEX = 1

_ALL_SC2_IDS = sorted(set(_NEUTRAL_IDS + _PROTOSS_IDS + _TERRAN_IDS + _ZERG_IDS))

MOVING_UNIT_TYPE_IDS = frozenset(
    _ids(_MOVING_TERRAN_UNITS) + _ids(_MOVING_PROTOSS_UNITS) + _ids(_MOVING_ZERG_UNITS)
)
STATIC_UNIT_TYPE_IDS = frozenset(
    _ids(_STATIC_TERRAN_UNITS)
    + _ids(_STATIC_PROTOSS_UNITS)
    + _ids(_STATIC_ZERG_UNITS)
    + _ids(_STATIC_NEUTRAL_UNITS)
)

UNIT_ID_TO_NAME = {
    unit_id: name
    for units in (_NEUTRAL_UNITS, _PROTOSS_UNITS, _TERRAN_UNITS, _ZERG_UNITS)
    for name, unit_id in units
}
UNIT_ID_TO_RACE = {
    unit_id: race
    for race, units in (
        ("Neutral", _NEUTRAL_UNITS),
        ("Protoss", _PROTOSS_UNITS),
        ("Terran", _TERRAN_UNITS),
        ("Zerg", _ZERG_UNITS),
    )
    for _, unit_id in units
}

SC2_ID_TO_INDEX: dict[int, int] = {
    sc2_id: idx + 2 for idx, sc2_id in enumerate(_ALL_SC2_IDS)
}

NUM_UNIT_TYPES: int = len(_ALL_SC2_IDS) + 2  # +2 for Empty and Unknown


def remap_unit_type(raw_ids: np.ndarray) -> np.ndarray:
    """Map raw SC2 unit_type_id values to contiguous indices."""
    out = np.full_like(raw_ids, fill_value=UNKNOWN_INDEX, dtype=np.int64)
    for sc2_id, idx in SC2_ID_TO_INDEX.items():
        out[raw_ids == sc2_id] = idx
    return out


def is_moving_unit_type(unit_type_id: int) -> bool:
    return unit_type_id in MOVING_UNIT_TYPE_IDS


def is_static_unit_type(unit_type_id: int) -> bool:
    return unit_type_id in STATIC_UNIT_TYPE_IDS


def get_unit_name(unit_type_id: int) -> str | None:
    return UNIT_ID_TO_NAME.get(unit_type_id)


def get_unit_race(unit_type_id: int) -> str | None:
    return UNIT_ID_TO_RACE.get(unit_type_id)


def describe_unit_type(unit_type_id: int) -> str:
    name = get_unit_name(unit_type_id)
    race = get_unit_race(unit_type_id)
    if name is None or race is None:
        return f"Unknown({unit_type_id})"
    return f"{race}.{name}"
