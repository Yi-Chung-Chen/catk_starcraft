"""Coarse action category mapping for SC2 ability_ids.

Maps raw SC2 ability_id (uint32, as stored in HDF5 unit_data/repeated/ability_id)
to 10 coarse action classes suitable for downstream action prediction tasks.

Sources
-------
Primary:
    Blizzard s2client-api ABILITY_ID enum (~468 entries)
    https://blizzard.github.io/s2client-api/sc2__typeenums_8h.html

Supplementary (for 10 IDs missing from the enum but present in dataset):
    Blizzard s2client-proto stableid.json (3,797 entries)
    https://github.com/Blizzard/s2client-proto/blob/master/stableid.json

Dataset validated against:
    StarCraftMotion_hdf5_v2 (13,957 replays, 292 unique non-zero ability_ids)

Classification method
---------------------
1. Each Blizzard enum name has a clear prefix (TRAIN_*, BUILD_*, RESEARCH_*,
   MORPH_*, BURROWDOWN_*, EFFECT_*, HARVEST_*, LOAD_/UNLOAD_/LIFT_/LAND_*,
   ATTACK_*, MOVE/PATROL/HOLDPOSITION/STOP_*).
   We classify by matching the prefix to a coarse category.

2. For the 10 ability_ids that appear in the dataset but are absent from the
   Blizzard enum, we look them up in stableid.json and assign a category
   manually based on the stableid name. These are listed explicitly in
   _STABLEID_OVERRIDES below with rationale.

3. Any ability_id not covered by (1) or (2) maps to UNKNOWN (255).

Coverage: 100% of all action occurrences across the dataset.

Label 0 is reserved for NO_OP (ability_id == 0, meaning the unit has no
active order). Action classes start at 1.
"""

# ---------------------------------------------------------------------------
# Coarse action class definitions
# ---------------------------------------------------------------------------

COARSE_ACTION_NAMES = {
    0: "NO_OP",      # ability_id == 0: unit has no active order (idle)
    1: "MOVE",       # move, patrol, hold position, stop, smart (right-click)
    2: "ATTACK",     # attack, attack-move, attack building
    3: "HARVEST",    # gather resources, return cargo
    4: "TRAIN",      # produce units from buildings / larvae / warp-in
    5: "BUILD",      # construct structures, add-ons, creep tumors
    6: "RESEARCH",   # upgrades and tech research
    7: "MORPH",      # unit/structure transformation (siege, burrow-tumor, archon, etc.)
    8: "EFFECT",     # combat abilities, spells, auto-cast effects
    9: "TRANSPORT",  # load, unload, lift off, land
    10: "BURROW",    # burrow down / burrow up
    255: "UNKNOWN",  # unmapped or cosmetic
}

# ---------------------------------------------------------------------------
# Blizzard enum: ABILITY_ID name -> coarse category
# Source: https://blizzard.github.io/s2client-api/sc2__typeenums_8h.html
# ---------------------------------------------------------------------------

_PREFIX_TO_CATEGORY = {
    # Prefix (checked in order)      -> coarse label
    "ATTACK":                         2,
    "HARVEST":                        3,
    "TRAINWARP":                      4,
    "TRAIN":                          4,
    "BUILD":                          5,
    "RESEARCH":                       6,
    "MORPH":                          7,
    "BURROWDOWN":                     10,
    "BURROWUP":                       10,
    "EFFECT":                         8,
    "HALLUCINATION":                  8,
    "LOAD":                           9,
    "UNLOAD":                         9,
    "LIFT":                           9,
    "LAND":                           9,
    "MOVE":                           1,
    "PATROL":                         1,
    "HOLDPOSITION":                   1,
    "SCAN_MOVE":                      1,
    "STOP":                           1,
    "HALT":                           1,
    "SMART":                          1,
    "CANCEL":                         8,  # cancel is grouped with EFFECT (minor)
    "CANCELSLOT":                     8,
    "RALLY":                          8,  # rally is grouped with EFFECT (minor)
    "BEHAVIOR":                       8,  # toggles (cloak, hold fire) grouped with EFFECT
}


def _classify_by_prefix(enum_name):
    """Classify a Blizzard ABILITY_ID enum name by its prefix.

    Returns coarse label (int) or None if no prefix matches.
    """
    upper = enum_name.upper()
    for prefix, label in _PREFIX_TO_CATEGORY.items():
        if upper.startswith(prefix):
            return label
    return None


# Full Blizzard enum: ability_id -> enum name
# Extracted from https://blizzard.github.io/s2client-api/sc2__typeenums_8h.html
_BLIZZARD_ENUM = {
    0: "INVALID",
    1: "SMART",
    4: "STOP_STOP",
    6: "STOP_CHEER",
    7: "STOP_DANCE",
    16: "MOVE",
    17: "PATROL",
    18: "HOLDPOSITION",
    19: "SCAN_MOVE",
    23: "ATTACK_ATTACK",
    26: "EFFECT_SPRAY_TERRAN",
    28: "EFFECT_SPRAY_ZERG",
    30: "EFFECT_SPRAY_PROTOSS",
    32: "EFFECT_SALVAGE",
    36: "BEHAVIOR_HOLDFIREON_GHOST",
    42: "EFFECT_EXPLODE",
    44: "RESEARCH_INTERCEPTORGRAVITONCATAPULT",
    46: "RESEARCH_PHOENIXANIONPULSECRYSTALS",
    74: "EFFECT_FUNGALGROWTH",
    76: "EFFECT_GUARDIANSHIELD",
    78: "EFFECT_REPAIR_MULE",
    80: "TRAIN_BANELING",
    110: "TRAIN_MOTHERSHIP",
    140: "EFFECT_FEEDBACK",
    144: "EFFECT_POINTDEFENSEDRONE",
    146: "HALLUCINATION_ARCHON",
    148: "HALLUCINATION_COLOSSUS",
    150: "HALLUCINATION_HIGHTEMPLAR",
    152: "HALLUCINATION_IMMORTAL",
    154: "HALLUCINATION_PHOENIX",
    156: "HALLUCINATION_PROBE",
    158: "HALLUCINATION_STALKER",
    160: "HALLUCINATION_VOIDRAY",
    162: "HALLUCINATION_WARPPRISM",
    164: "HALLUCINATION_ZEALOT",
    167: "HARVEST_RETURN_MULE",
    169: "EFFECT_HUNTERSEEKERMISSILE",
    171: "EFFECT_CALLDOWNMULE",
    173: "EFFECT_GRAVITONBEAM",
    174: "CANCEL_GRAVITONBEAM",
    181: "EFFECT_SPAWNCHANGELING",
    195: "RALLY_BUILDING",
    199: "RALLY_MORPHING_UNIT",
    203: "RALLY_COMMANDCENTER",
    207: "RALLY_NEXUS",
    211: "RALLY_HATCHERY_UNITS",
    212: "RALLY_HATCHERY_WORKERS",
    216: "RESEARCH_GLIALREGENERATION",
    217: "RESEARCH_TUNNELINGCLAWS",
    247: "EFFECT_INFESTEDTERRANS",
    249: "EFFECT_NEURALPARASITE",
    251: "EFFECT_INJECTLARVA",
    253: "EFFECT_STIM_MARAUDER",
    255: "EFFECT_SUPPLYDROP",
    261: "EFFECT_CHRONOBOOST",
    265: "RESEARCH_CHITINOUSPLATING",
    295: "HARVEST_GATHER_SCV",
    296: "HARVEST_RETURN_SCV",
    298: "HARVEST_GATHER_PROBE",
    299: "HARVEST_RETURN_PROBE",
    304: "CANCEL_QUEUE1",
    305: "CANCELSLOT_QUEUE1",
    306: "CANCEL_QUEUE5",
    307: "CANCELSLOT_QUEUE5",
    308: "CANCEL_QUEUECANCELTOSELECTION",
    309: "CANCELSLOT_QUEUECANCELTOSELECTION",
    312: "CANCEL_QUEUEADDON",
    313: "CANCELSLOT_ADDON",
    314: "CANCEL_BUILDINPROGRESS",
    315: "HALT_BUILDING",
    316: "EFFECT_REPAIR_SCV",
    318: "BUILD_COMMANDCENTER",
    319: "BUILD_SUPPLYDEPOT",
    320: "BUILD_REFINERY",
    321: "BUILD_BARRACKS",
    322: "BUILD_ENGINEERINGBAY",
    323: "BUILD_MISSILETURRET",
    324: "BUILD_BUNKER",
    326: "BUILD_SENSORTOWER",
    327: "BUILD_GHOSTACADEMY",
    328: "BUILD_FACTORY",
    329: "BUILD_STARPORT",
    331: "BUILD_ARMORY",
    333: "BUILD_FUSIONCORE",
    348: "HALT_TERRANBUILD",
    380: "EFFECT_STIM_MARINE",
    382: "BEHAVIOR_CLOAKON_GHOST",
    383: "BEHAVIOR_CLOAKOFF_GHOST",
    386: "EFFECT_HEAL",
    388: "MORPH_SIEGEMODE",
    390: "MORPH_UNSIEGE",
    392: "BEHAVIOR_CLOAKON_BANSHEE",
    393: "BEHAVIOR_CLOAKOFF_BANSHEE",
    394: "LOAD_MEDIVAC",
    396: "UNLOADALLAT_MEDIVAC",
    397: "UNLOADUNIT_MEDIVAC",
    399: "EFFECT_SCAN",
    401: "EFFECT_YAMATOGUN",
    403: "MORPH_VIKINGASSAULTMODE",
    405: "MORPH_VIKINGFIGHTERMODE",
    407: "LOAD_BUNKER",
    408: "UNLOADALL_BUNKER",
    410: "UNLOADUNIT_BUNKER",
    413: "UNLOADALL_COMMANDCENTER",
    415: "UNLOADUNIT_COMMANDCENTER",
    416: "LOADALL_COMMANDCENTER",
    417: "LIFT_COMMANDCENTER",
    419: "LAND_COMMANDCENTER",
    421: "BUILD_TECHLAB_BARRACKS",
    422: "BUILD_REACTOR_BARRACKS",
    451: "CANCEL_BARRACKSADDON",
    452: "LIFT_BARRACKS",
    454: "BUILD_TECHLAB_FACTORY",
    455: "BUILD_REACTOR_FACTORY",
    484: "CANCEL_FACTORYADDON",
    485: "LIFT_FACTORY",
    487: "BUILD_TECHLAB_STARPORT",
    488: "BUILD_REACTOR_STARPORT",
    517: "CANCEL_STARPORTADDON",
    518: "LIFT_STARPORT",
    520: "LAND_FACTORY",
    522: "LAND_STARPORT",
    524: "TRAIN_SCV",
    554: "LAND_BARRACKS",
    556: "MORPH_SUPPLYDEPOT_LOWER",
    558: "MORPH_SUPPLYDEPOT_RAISE",
    560: "TRAIN_MARINE",
    561: "TRAIN_REAPER",
    562: "TRAIN_GHOST",
    563: "TRAIN_MARAUDER",
    591: "TRAIN_SIEGETANK",
    594: "TRAIN_THOR",
    595: "TRAIN_HELLION",
    596: "TRAIN_HELLBAT",
    597: "TRAIN_CYCLONE",
    614: "TRAIN_WIDOWMINE",
    620: "TRAIN_MEDIVAC",
    621: "TRAIN_BANSHEE",
    622: "TRAIN_RAVEN",
    623: "TRAIN_BATTLECRUISER",
    624: "TRAIN_VIKINGFIGHTER",
    626: "TRAIN_LIBERATOR",
    650: "RESEARCH_HISECAUTOTRACKING",
    651: "RESEARCH_TERRANSTRUCTUREARMORUPGRADE",
    652: "RESEARCH_TERRANINFANTRYWEAPONSLEVEL1",
    653: "RESEARCH_TERRANINFANTRYWEAPONSLEVEL2",
    654: "RESEARCH_TERRANINFANTRYWEAPONSLEVEL3",
    655: "RESEARCH_NEOSTEELFRAME",
    656: "RESEARCH_TERRANINFANTRYARMORLEVEL1",
    657: "RESEARCH_TERRANINFANTRYARMORLEVEL2",
    658: "RESEARCH_TERRANINFANTRYARMORLEVEL3",
    710: "BUILD_NUKE",
    730: "RESEARCH_STIMPACK",
    731: "RESEARCH_COMBATSHIELD",
    732: "RESEARCH_CONCUSSIVESHELLS",
    761: "RESEARCH_INFERNALPREIGNITER",
    764: "RESEARCH_DRILLINGCLAWS",
    766: "RESEARCH_SMARTSERVOS",
    768: "RESEARCH_RAPIDFIRELAUNCHERS",
    790: "RESEARCH_BANSHEECLOAKINGFIELD",
    793: "RESEARCH_RAVENCORVIDREACTOR",
    799: "RESEARCH_BANSHEEHYPERFLIGHTROTORS",
    803: "RESEARCH_RAVENRECALIBRATEDEXPLOSIVES",
    804: "RESEARCH_HIGHCAPACITYFUELTANKS",
    805: "RESEARCH_ADVANCEDBALLISTICS",
    806: "RESEARCH_ENHANCEDMUNITIONS",
    820: "RESEARCH_PERSONALCLOAKING",
    855: "RESEARCH_TERRANVEHICLEWEAPONSLEVEL1",
    856: "RESEARCH_TERRANVEHICLEWEAPONSLEVEL2",
    857: "RESEARCH_TERRANVEHICLEWEAPONSLEVEL3",
    861: "RESEARCH_TERRANSHIPWEAPONSLEVEL1",
    862: "RESEARCH_TERRANSHIPWEAPONSLEVEL2",
    863: "RESEARCH_TERRANSHIPWEAPONSLEVEL3",
    864: "RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1",
    865: "RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2",
    866: "RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3",
    880: "BUILD_NEXUS",
    881: "BUILD_PYLON",
    882: "BUILD_ASSIMILATOR",
    883: "BUILD_GATEWAY",
    884: "BUILD_FORGE",
    885: "BUILD_FLEETBEACON",
    886: "BUILD_TWILIGHTCOUNCIL",
    887: "BUILD_PHOTONCANNON",
    889: "BUILD_STARGATE",
    890: "BUILD_TEMPLARARCHIVE",
    891: "BUILD_DARKSHRINE",
    892: "BUILD_ROBOTICSBAY",
    893: "BUILD_ROBOTICSFACILITY",
    894: "BUILD_CYBERNETICSCORE",
    895: "BUILD_SHIELDBATTERY",
    911: "LOAD_WARPPRISM",
    913: "UNLOADALLAT_WARPPRISM",
    914: "UNLOADUNIT_WARPPRISM",
    916: "TRAIN_ZEALOT",
    917: "TRAIN_STALKER",
    919: "TRAIN_HIGHTEMPLAR",
    920: "TRAIN_DARKTEMPLAR",
    921: "TRAIN_SENTRY",
    922: "TRAIN_ADEPT",
    946: "TRAIN_PHOENIX",
    948: "TRAIN_CARRIER",
    950: "TRAIN_VOIDRAY",
    954: "TRAIN_ORACLE",
    955: "TRAIN_TEMPEST",
    976: "TRAIN_WARPPRISM",
    977: "TRAIN_OBSERVER",
    978: "TRAIN_COLOSSUS",
    979: "TRAIN_IMMORTAL",
    994: "TRAIN_DISRUPTOR",
    1006: "TRAIN_PROBE",
    1036: "EFFECT_PSISTORM",
    1042: "BUILD_INTERCEPTORS",
    1062: "RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1",
    1063: "RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2",
    1064: "RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3",
    1065: "RESEARCH_PROTOSSGROUNDARMORLEVEL1",
    1066: "RESEARCH_PROTOSSGROUNDARMORLEVEL2",
    1067: "RESEARCH_PROTOSSGROUNDARMORLEVEL3",
    1068: "RESEARCH_PROTOSSSHIELDSLEVEL1",
    1069: "RESEARCH_PROTOSSSHIELDSLEVEL2",
    1070: "RESEARCH_PROTOSSSHIELDSLEVEL3",
    1093: "RESEARCH_GRAVITICBOOSTER",
    1094: "RESEARCH_GRAVITICDRIVE",
    1097: "RESEARCH_EXTENDEDTHERMALLANCE",
    1126: "RESEARCH_PSISTORM",
    1152: "BUILD_HATCHERY",
    1154: "BUILD_EXTRACTOR",
    1155: "BUILD_SPAWNINGPOOL",
    1156: "BUILD_EVOLUTIONCHAMBER",
    1157: "BUILD_HYDRALISKDEN",
    1158: "BUILD_SPIRE",
    1159: "BUILD_ULTRALISKCAVERN",
    1160: "BUILD_INFESTATIONPIT",
    1161: "BUILD_NYDUSNETWORK",
    1162: "BUILD_BANELINGNEST",
    1163: "BUILD_LURKERDEN",
    1165: "BUILD_ROACHWARREN",
    1166: "BUILD_SPINECRAWLER",
    1167: "BUILD_SPORECRAWLER",
    1183: "HARVEST_GATHER_DRONE",
    1184: "HARVEST_RETURN_DRONE",
    1186: "RESEARCH_ZERGMELEEWEAPONSLEVEL1",
    1187: "RESEARCH_ZERGMELEEWEAPONSLEVEL2",
    1188: "RESEARCH_ZERGMELEEWEAPONSLEVEL3",
    1189: "RESEARCH_ZERGGROUNDARMORLEVEL1",
    1190: "RESEARCH_ZERGGROUNDARMORLEVEL2",
    1191: "RESEARCH_ZERGGROUNDARMORLEVEL3",
    1192: "RESEARCH_ZERGMISSILEWEAPONSLEVEL1",
    1193: "RESEARCH_ZERGMISSILEWEAPONSLEVEL2",
    1194: "RESEARCH_ZERGMISSILEWEAPONSLEVEL3",
    1216: "MORPH_LAIR",
    1217: "CANCEL_MORPHLAIR",
    1218: "MORPH_HIVE",
    1220: "MORPH_GREATERSPIRE",
    1223: "RESEARCH_PNEUMATIZEDCARAPACE",
    1225: "RESEARCH_BURROW",
    1252: "RESEARCH_ZERGLINGADRENALGLANDS",
    1253: "RESEARCH_ZERGLINGMETABOLICBOOST",
    1282: "RESEARCH_GROOVEDSPINES",
    1283: "RESEARCH_MUSCULARAUGMENTS",
    1312: "RESEARCH_ZERGFLYERATTACKLEVEL1",
    1313: "RESEARCH_ZERGFLYERATTACKLEVEL2",
    1314: "RESEARCH_ZERGFLYERATTACKLEVEL3",
    1315: "RESEARCH_ZERGFLYERARMORLEVEL1",
    1316: "RESEARCH_ZERGFLYERARMORLEVEL2",
    1317: "RESEARCH_ZERGFLYERARMORLEVEL3",
    1342: "TRAIN_DRONE",
    1343: "TRAIN_ZERGLING",
    1344: "TRAIN_OVERLORD",
    1345: "TRAIN_HYDRALISK",
    1346: "TRAIN_MUTALISK",
    1348: "TRAIN_ULTRALISK",
    1351: "TRAIN_ROACH",
    1352: "TRAIN_INFESTOR",
    1353: "TRAIN_CORRUPTOR",
    1354: "TRAIN_VIPER",
    1356: "TRAIN_SWARMHOST",
    1372: "MORPH_BROODLORD",
    1373: "CANCEL_MORPHBROODLORD",
    1374: "BURROWDOWN_BANELING",
    1376: "BURROWUP_BANELING",
    1378: "BURROWDOWN_DRONE",
    1380: "BURROWUP_DRONE",
    1382: "BURROWDOWN_HYDRALISK",
    1384: "BURROWUP_HYDRALISK",
    1386: "BURROWDOWN_ROACH",
    1388: "BURROWUP_ROACH",
    1390: "BURROWDOWN_ZERGLING",
    1392: "BURROWUP_ZERGLING",
    1394: "BURROWDOWN_INFESTORTERRAN",
    1396: "BURROWUP_INFESTORTERRAN",
    1406: "LOAD_OVERLORD",
    1408: "UNLOADALLAT_OVERLORD",
    1409: "UNLOADUNIT_OVERLORD",
    1413: "TRAINWARP_ZEALOT",
    1414: "TRAINWARP_STALKER",
    1416: "TRAINWARP_HIGHTEMPLAR",
    1417: "TRAINWARP_DARKTEMPLAR",
    1418: "TRAINWARP_SENTRY",
    1419: "TRAINWARP_ADEPT",
    1433: "BURROWDOWN_QUEEN",
    1435: "BURROWUP_QUEEN",
    1437: "LOAD_NYDUSNETWORK",
    1438: "UNLOADALL_NYDUSNETWORK",
    1442: "EFFECT_BLINK_STALKER",
    1444: "BURROWDOWN_INFESTOR",
    1446: "BURROWUP_INFESTOR",
    1448: "MORPH_OVERSEER",
    1449: "CANCEL_MORPHOVERSEER",
    1450: "MORPH_PLANETARYFORTRESS",
    1451: "CANCEL_MORPHPLANETARYFORTRESS",
    1454: "RESEARCH_PATHOGENGLANDS",
    1455: "RESEARCH_NEURALPARASITE",
    1482: "RESEARCH_CENTRIFUGALHOOKS",
    1512: "BURROWDOWN_ULTRALISK",
    1514: "BURROWUP_ULTRALISK",
    1516: "MORPH_ORBITALCOMMAND",
    1517: "CANCEL_MORPHORBITAL",
    1518: "MORPH_WARPGATE",
    1520: "MORPH_GATEWAY",
    1522: "LIFT_ORBITALCOMMAND",
    1524: "LAND_ORBITALCOMMAND",
    1526: "EFFECT_FORCEFIELD",
    1528: "MORPH_WARPPRISMPHASINGMODE",
    1530: "MORPH_WARPPRISMTRANSPORTMODE",
    1532: "RESEARCH_BATTLECRUISERWEAPONREFIT",
    1562: "RESEARCH_PROTOSSAIRWEAPONSLEVEL1",
    1563: "RESEARCH_PROTOSSAIRWEAPONSLEVEL2",
    1564: "RESEARCH_PROTOSSAIRWEAPONSLEVEL3",
    1565: "RESEARCH_PROTOSSAIRARMORLEVEL1",
    1566: "RESEARCH_PROTOSSAIRARMORLEVEL2",
    1567: "RESEARCH_PROTOSSAIRARMORLEVEL3",
    1568: "RESEARCH_WARPGATE",
    1592: "RESEARCH_CHARGE",
    1593: "RESEARCH_BLINK",
    1594: "RESEARCH_ADEPTRESONATINGGLAIVES",
    1622: "EFFECT_NUKECALLDOWN",
    1628: "EFFECT_EMP",
    1632: "TRAIN_QUEEN",
    1664: "EFFECT_TRANSFUSION",
    1682: "ATTACK_REDIRECT",
    1683: "EFFECT_STIM_MARINE_REDIRECT",
    1691: "STOP_REDIRECT",
    1692: "BEHAVIOR_GENERATECREEPON",
    1693: "BEHAVIOR_GENERATECREEPOFF",
    1694: "BUILD_CREEPTUMOR_QUEEN",
    1725: "MORPH_SPINECRAWLERUPROOT",
    1727: "MORPH_SPORECRAWLERUPROOT",
    1729: "MORPH_SPINECRAWLERROOT",
    1730: "CANCEL_SPINECRAWLERROOT",
    1731: "MORPH_SPORECRAWLERROOT",
    1733: "BUILD_CREEPTUMOR_TUMOR",
    1763: "CANCEL_CREEPTUMOR",
    1764: "EFFECT_AUTOTURRET",
    1766: "MORPH_ARCHON",
    1768: "BUILD_NYDUSWORM",
    1819: "EFFECT_CHARGE",
    1825: "EFFECT_CONTAMINATE",
    1831: "CANCEL_QUEUEPASSIVE",
    1833: "CANCEL_QUEUEPASSIVECANCELTOSELECTION",
    1847: "MORPH_MOTHERSHIP",
    1848: "CANCEL_MORPHMOTHERSHIP",
    1853: "TRAIN_MOTHERSHIPCORE",
    1974: "EFFECT_MASSRECALL_MOTHERSHIPCORE",
    1978: "MORPH_HELLION",
    1998: "MORPH_HELLBAT",
    2014: "BURROWDOWN_SWARMHOST",
    2016: "BURROWUP_SWARMHOST",
    2048: "ATTACK_ATTACKBUILDING",
    2057: "STOP_BUILDING",
    2063: "EFFECT_BLINDINGCLOUD",
    2067: "EFFECT_ABDUCT",
    2073: "EFFECT_VIPERCONSUME",
    2081: "BEHAVIOR_BUILDINGATTACKON",
    2082: "BEHAVIOR_BUILDINGATTACKOFF",
    2095: "BURROWDOWN_WIDOWMINE",
    2097: "BURROWUP_WIDOWMINE",
    2099: "EFFECT_WIDOWMINEATTACK",
    2108: "BURROWDOWN_LURKER",
    2110: "BURROWUP_LURKER",
    2112: "MORPH_LURKERDEN",
    2113: "CANCEL_MORPHLURKERDEN",
    2114: "HALLUCINATION_ORACLE",
    2116: "EFFECT_MEDIVACIGNITEAFTERBURNERS",
    2146: "EFFECT_ORACLEREVELATION",
    2162: "EFFECT_PHOTONOVERCHARGE",
    2244: "EFFECT_TIMEWARP",
    2324: "EFFECT_CAUSTICSPRAY",
    2328: "EFFECT_IMMORTALBARRIER",
    2330: "MORPH_RAVAGER",
    2331: "CANCEL_MORPHRAVAGER",
    2332: "MORPH_LURKER",
    2333: "CANCEL_MORPHLURKER",
    2338: "EFFECT_CORROSIVEBILE",
    2340: "BURROWDOWN_RAVAGER",
    2342: "BURROWUP_RAVAGER",
    2346: "EFFECT_PURIFICATIONNOVA",
    2350: "EFFECT_LOCKON",
    2358: "EFFECT_TACTICALJUMP",
    2362: "MORPH_THORHIGHIMPACTMODE",
    2364: "MORPH_THOREXPLOSIVEMODE",
    2368: "EFFECT_MASSRECALL_MOTHERSHIP",
    2370: "LOAD_NYDUSWORM",
    2371: "UNLOADALL_NYDUSWORM",
    2375: "BEHAVIOR_PULSARBEAMON",
    2376: "BEHAVIOR_PULSARBEAMOFF",
    2387: "EFFECT_LOCUSTSWOOP",
    2389: "HALLUCINATION_DISRUPTOR",
    2391: "HALLUCINATION_ADEPT",
    2393: "EFFECT_VOIDRAYPRISMATICALIGNMENT",
    2505: "BUILD_STASISTRAP",
    2542: "EFFECT_PARASITICBOMB",
    2544: "EFFECT_ADEPTPHASESHIFT",
    2550: "BEHAVIOR_HOLDFIREON_LURKER",
    2552: "BEHAVIOR_HOLDFIREOFF_LURKER",
    2558: "MORPH_LIBERATORAGMODE",
    2560: "MORPH_LIBERATORAAMODE",
    2588: "EFFECT_KD8CHARGE",
    2594: "CANCEL_ADEPTPHASESHIFT",
    2596: "CANCEL_ADEPTSHADEPHASESHIFT",
    2698: "EFFECT_TEMPESTDISRUPTIONBLAST",
    2700: "EFFECT_SHADOWSTRIDE",
    2704: "EFFECT_SPAWNLOCUSTS",
    2708: "MORPH_OVERLORDTRANSPORT",
    2709: "CANCEL_MORPHOVERLORDTRANSPORT",
    2714: "EFFECT_GHOSTSNIPE",
    2720: "RESEARCH_SHADOWSTRIKE",
    3659: "CANCEL",
    3660: "HALT",
    3661: "BURROWDOWN",
    3662: "BURROWUP",
    3663: "LOADALL",
    3664: "UNLOADALL",
    3665: "STOP",
    3666: "HARVEST_GATHER",
    3667: "HARVEST_RETURN",
    3668: "LOAD",
    3669: "UNLOADALLAT",
    3671: "CANCEL_LAST",
    3673: "RALLY_UNITS",
    3674: "ATTACK",
    3675: "EFFECT_STIM",
    3676: "BEHAVIOR_CLOAKON",
    3677: "BEHAVIOR_CLOAKOFF",
    3678: "LAND",
    3679: "LIFT",
    3680: "MORPH_ROOT",
    3681: "MORPH_UPROOT",
    3682: "BUILD_TECHLAB",
    3683: "BUILD_REACTOR",
    3684: "EFFECT_SPRAY",
    3685: "EFFECT_REPAIR",
    3686: "EFFECT_MASSRECALL",
    3687: "EFFECT_BLINK",
    3688: "BEHAVIOR_HOLDFIREON",
    3689: "BEHAVIOR_HOLDFIREOFF",
    3690: "RALLY_WORKERS",
    3691: "BUILD_CREEPTUMOR",
    3692: "RESEARCH_PROTOSSAIRARMOR",
    3693: "RESEARCH_PROTOSSAIRWEAPONS",
    3694: "RESEARCH_PROTOSSGROUNDARMOR",
    3695: "RESEARCH_PROTOSSGROUNDWEAPONS",
    3696: "RESEARCH_PROTOSSSHIELDS",
    3697: "RESEARCH_TERRANINFANTRYARMOR",
    3698: "RESEARCH_TERRANINFANTRYWEAPONS",
    3699: "RESEARCH_TERRANSHIPWEAPONS",
    3700: "RESEARCH_TERRANVEHICLEANDSHIPPLATING",
    3701: "RESEARCH_TERRANVEHICLEWEAPONS",
    3702: "RESEARCH_ZERGFLYERARMOR",
    3703: "RESEARCH_ZERGFLYERATTACK",
    3704: "RESEARCH_ZERGGROUNDARMOR",
    3705: "RESEARCH_ZERGMELEEWEAPONS",
    3706: "RESEARCH_ZERGMISSILEWEAPONS",
    3765: "EFFECT_RESTORE",
}

# ---------------------------------------------------------------------------
# Manual overrides for ability_ids present in the dataset but absent from
# the Blizzard enum.
#
# Source: stableid.json
# (https://github.com/Blizzard/s2client-proto/blob/master/stableid.json)
#
# Each entry documents:
#   ability_id: (coarse_label, stableid_name, reason)
# ---------------------------------------------------------------------------

_STABLEID_OVERRIDES = {
    # ability_id: (label, stableid_name, why it's missing & how we classify)

    # Creep tumor auto-burrows after spawning. Not a player-initiated burrow,
    # so classified as MORPH (structure state change) rather than BURROW.
    1662: (7, "BurrowCreepTumorDown"),

    # Liberator mode switches use different IDs than the enum's 2558/2560.
    # The enum has MORPH_LIBERATORAGMODE (2558) and MORPH_LIBERATORAAMODE (2560),
    # but the raw replay data also emits 2554/2556 for the same transformations.
    2554: (7, "LiberatorMorphtoAG"),
    2556: (7, "LiberatorMorphtoAA"),

    # Legacy ability removed in a later SC2 patch. Appears in older replays.
    # MorphToInfestedTerran: Infestor spawns Infested Terran (removed unit).
    40: (7, "MorphToInfestedTerran"),

    # Locust (SwarmHost summon) morph between flying and ground forms.
    2383: (7, "LocustMPFlyingMorphToGround"),
    2385: (7, "LocustMPMorphToAir"),

    # Second step of Archon merge. The enum has MORPH_ARCHON (1766) for the
    # initiating unit; 1767 is the target unit's complementary order.
    1767: (7, "ArchonWarp_Target"),

    # Locust flying attack-swoop. An auto-cast combat behavior of SwarmHost
    # locust. Classified as EFFECT (auto-cast combat ability).
    2706: (8, "LocustMPFlyingSwoopAttack"),

    # Oracle stasis ward activation. Classified as EFFECT (triggered ability).
    2536: (8, "OracleStasisTrapActivate"),

    # MULE gather (Harvest_Gather equivalent for MULEs). The enum has
    # HARVEST_RETURN_MULE (167) but omits the gather counterpart.
    166: (3, "HARVEST_GATHER_MULE"),
}

# ---------------------------------------------------------------------------
# Build the final lookup table
# ---------------------------------------------------------------------------

# Step 1: classify every Blizzard enum entry by prefix
_ABILITY_ID_TO_COARSE = {}
for _aid, _name in _BLIZZARD_ENUM.items():
    _label = _classify_by_prefix(_name)
    if _label is not None:
        _ABILITY_ID_TO_COARSE[_aid] = _label
    # else: INVALID (0) — not classified, will fall through to UNKNOWN

# Step 2: apply stableid overrides (takes precedence)
for _aid, (_label, _sname) in _STABLEID_OVERRIDES.items():
    _ABILITY_ID_TO_COARSE[_aid] = _label


def get_coarse_action(ability_id):
    """Map a raw SC2 ability_id to a coarse action label.

    Parameters
    ----------
    ability_id : int
        Raw ability_id from SC2 replay / HDF5 dataset.

    Returns
    -------
    int
        Coarse action label: 0 for NO_OP (idle), 1-10 for action classes,
        or 255 for UNKNOWN.
    """
    if ability_id == 0:
        return 0  # NO_OP
    return _ABILITY_ID_TO_COARSE.get(ability_id, 255)


def get_coarse_action_name(label):
    """Return the string name for a coarse action label."""
    return COARSE_ACTION_NAMES.get(label, "UNKNOWN")


# Convenience: expose the full dict for vectorized numpy lookups
ABILITY_ID_TO_COARSE_ACTION = {0: 0}  # NO_OP: ability_id 0 -> label 0
ABILITY_ID_TO_COARSE_ACTION.update(_ABILITY_ID_TO_COARSE)
