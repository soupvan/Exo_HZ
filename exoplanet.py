# FULL PATH TO MY CODE:
# /homes/metogra/sdillon1/AOSC247/Final Project/exoplanet.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import re

# Unit conversion constants used when loading the EU catalog, to get everything in Earth units
MJUP_TO_MEARTH = 317.828   # 1 Jupiter mass = 317.828 Earth masses
RJUP_TO_REARTH = 11.2089   # 1 Jupiter radius = 11.2089 Earth radii

# Bond albedo = the fraction of starlight reflected by a planet
# T_EQ_EARTH = Earths equilibrium temperature without greenhouse effect
# T_SURF_EARTH = Earths temp w/ greenhouse in real life
# GH_FACTOR = the ratio to scale equilibrium temp to estimated surface temp
BOND_ALBEDO  = 0.30
T_EQ_EARTH   = 255.0
T_SURF_EARTH = 288.0
GH_FACTOR    = T_SURF_EARTH / T_EQ_EARTH


# calculate stellar luminosity from radius and temperature
# R_sun=1, T_sun=5778 K
# Gives log10(L/L_sun)
def lum_from_RT(R_rsun, T_K):
  return 2 * np.log10(R_rsun) + 4 * np.log10(T_K / 5778.0)


# Calculate bulk density (g/cm^3) from mass and radius in Earth units
# Converts Earth masses to grams and Earth radii to centimeters
def density_from_MR(m_earth, r_earth):
  M_g  = m_earth * 5.972e27        # grams
  R_cm = r_earth * 6.371e8         # cm
  return M_g / (4.0 / 3.0 * np.pi * R_cm ** 3)


# 1.) LOAD NASA EXOPLANET ARCHIVE DATA =============================================================================
#"#" for comments and the controversial flag column marks planets whose existence is disputed
print("1.)")

NASA_FILE = "PS_2026.04.27_16.42.32.csv"
raw_nasa = pd.read_csv(NASA_FILE, comment="#", low_memory=False)

# If the controversial flag column exists, label planet as Unconfirmed
# Assume all NASA planets are Confirmed but i had to fix this later for KOI exoplanets
if "pl_controv_flag" in raw_nasa.columns:
  nasa_conf = raw_nasa["pl_controv_flag"].apply(lambda x: "Unconfirmed" if x == 1 else "Confirmed")
else:
  nasa_conf = pd.Series(["Confirmed"] * len(raw_nasa))

# Select only the columns we need and rename and standardize them if they need to be
nasa = pd.DataFrame({
  "pl_name":     raw_nasa["pl_name"],
  "hostname":    raw_nasa["hostname"],
  "pl_masse":    raw_nasa["pl_masse"],
  "pl_rade":     raw_nasa["pl_rade"],
  "pl_orbsmax":  raw_nasa["pl_orbsmax"],
  "pl_eqt":      raw_nasa["pl_eqt"],
  "pl_dens":     raw_nasa.get("pl_dens"),
  "st_teff":     raw_nasa["st_teff"],
  "st_lum":      raw_nasa["st_lum"],
  "st_spectype": raw_nasa["st_spectype"],
  "source":      "NASA",
  "confirmed":   nasa_conf,
})

print(f"[NASA Data] Loaded {len(nasa):,} planets from {nasa['hostname'].nunique():,} host stars")

# 2.) LOAD EXOPLANET.EU DATA =============================================================================
print("2.)")

EU_FILE = "exoplanet.eu_catalog_02-05-26_23_02_41.csv"
raw_eu = pd.read_csv(EU_FILE, low_memory=False)

# Convert Jupiter masses/radii to Earth units so everything matches
eu_masse = raw_eu["mass"]   * MJUP_TO_MEARTH
eu_rade  = raw_eu["radius"] * RJUP_TO_REARTH

# Find log luminosity where stellar radius and temperature are available and greater than zero
valid_mask = (
  raw_eu["star_radius"].notna() & raw_eu["star_teff"].notna() &
  (raw_eu["star_radius"] > 0)   & (raw_eu["star_teff"] > 0)
)

# np.where evaluates both branches before selecting them so bad values would reach log10 even if valid_mask would discard them
# This gave me a RuntimeWarning so .where(valid_mask, 1.0) replaces invalid rows with 1.0 before log10 sees them
# Challenge
eu_st_lum = np.where(
  valid_mask,
  lum_from_RT(
    raw_eu["star_radius"].where(valid_mask, 1.0),
    raw_eu["star_teff"].where(valid_mask, 1.0),
  ),
  np.nan,
)

# Find bulk density when both mass and radius data is available, a lot of blanks in all this data
eu_dens = np.where(
  eu_masse.notna() & eu_rade.notna(),
  density_from_MR(eu_masse, eu_rade),
  np.nan,
)

# Confirmed vs unconfirmed planets
# everything that isn't confirmed (candidate, unconfirmed, retracted, etc) is unconfirmed
eu_conf = raw_eu["planet_status"].apply(
  lambda x: "Confirmed" if str(x).strip() == "Confirmed" else "Unconfirmed"
)

# EU Data in the same format as the NASA Data so I can put them together
eu = pd.DataFrame({
  "pl_name":     raw_eu["name"],
  "hostname":    raw_eu["star_name"],
  "pl_masse":    eu_masse,
  "pl_rade":     eu_rade,
  "pl_orbsmax":  raw_eu["semi_major_axis"],
  "pl_eqt":      raw_eu["temp_calculated"],
  "pl_dens":     eu_dens,
  "st_teff":     raw_eu["star_teff"],
  "st_lum":      pd.Series(eu_st_lum, dtype=float),
  "st_spectype": raw_eu["star_sp_type"],
  "source":      "EU",
  "confirmed":   eu_conf,
})

print(f"[EU Data] Loaded {len(eu):,} planets from {eu['hostname'].nunique():,} host stars")

# 3.) COMBINE BOTH DATA SETS =============================================================================
# NASA data takes priority so we remove EU planets already in NASA by name match
# If the EU catalog marks a planet Unconfirmed that NASA has as Confirmed update the planet entry to Unconfirmed
# better safe then sorry if there is new information disputing the planets confirmed-ness
print("3.)")

# Set of lowercase NASA planet names for confirmation testing
nasa_names = set(nasa["pl_name"].str.strip().str.lower())

# Find EU rows whose planet name already exists in NASA
eu_dupes   = eu["pl_name"].str.strip().str.lower().isin(nasa_names)

# Keep only EU planets NOT already in NASA
eu_unique  = eu[~eu_dupes].copy()

# In the list of EU duplicates, find those the EU flags as Unconfirmed
# and put that label onto the NASA data, cofirm it matches up
eu_unconfirmed_names = set(
  eu[eu_dupes & (eu["confirmed"] == "Unconfirmed")]["pl_name"]
  .str.strip().str.lower()
)

if eu_unconfirmed_names:
  mask = nasa["pl_name"].str.strip().str.lower().isin(eu_unconfirmed_names)
  nasa.loc[mask, "confirmed"] = "Unconfirmed"

# NASA + unique EU rows into one big Data set
df = pd.concat([nasa, eu_unique], ignore_index=True)

# I had to go back and fix this later but for some reason I think the EU data marks KOI planets as confirmed
# even though they're not so i had to fix that retroactively
# KOI names are candidate designations by definition and not confirmed planet names. 
# Any planet still named with a KOI prefix should be treated as Unconfirmed regardless of what EU data set says
KOI_PATTERN = re.compile(r"^KOI", re.IGNORECASE)

count = 0

for i, name in df["pl_name"].items():
  if KOI_PATTERN.match(str(name)) and df.at[i, "confirmed"] == "Confirmed":
    df.at[i, "confirmed"] = "Unconfirmed"
    count += 1
if count > 0:
  print(f"{count} KOI-named planets corrected to Unconfirmed")
n_eu_added = len(eu_unique)

print(f"\n[Both data sets] {eu_dupes.sum():,} EU planets already in NASA catalog skipped")
print(f"[Both data sets] {n_eu_added:,} unique EU planets added")
print(f"[Both data sets] Combined = {len(df):,} planet and star data entries (some duplicates) from {df['hostname'].nunique():,} host stars total!!!!")

# 4.) GET COLUMNS FOR PROCESSING AND PLOTTING =================================================================
print("4.)")

# Convert log10 luminosity to (L/L_sun)
# Habitable zone boundaries scales with sqrt(L)
df["st_lum_lin"] = 10.0 ** df["st_lum"]

# Get the first letter of the spectral type string to get the class
# O, B, A, F, G, K, M, and Anything else = Unknown
def get_spec_class(s):
  s = str(s).strip()
  return s[0] if len(s) > 0 and s[0] in "OBAFGKM" else "Unknown"

df["spec_class"] = df["st_spectype"].apply(get_spec_class)

# Classify each planet by bulk density (g/cm^3) into categories:
#   Terrestrial: rho > 3.9   
#   Sub-Neptune: 1.5 - 3.9   
#   Neptune-like: 0.5 - 1.5   
#   Gas Giant: rho <= 0.5  
def classify_planet(rho):
  if pd.isna(rho):  return "Unknown"
  elif rho > 3.9:   return "Terrestrial"
  elif rho > 1.5:   return "Sub-Neptune"
  elif rho > 0.5:   return "Neptune-like"
  else:             return "Gas Giant"

df["planet_type"] = df["pl_dens"].apply(classify_planet)

print("\n Classification:")
print(df["planet_type"].value_counts().to_string())
print(f"\n({df['planet_type'].eq('Unknown').sum():,} planets lack data to find density)\n")

# Habitable zone inner and outer boundaries for each planet's host star
# a_HZ ~ sqrt(L/L_sun)
# Inner edge: 0.95 AU at L=1
# Outer edge: 1.70 AU at L=1
df["hz_inner"] = 0.95 * np.sqrt(df["st_lum_lin"])
df["hz_outer"] = 1.70 * np.sqrt(df["st_lum_lin"])

# Find each planet's position relative to its host star's habitable zone
# based on the semi-major axis vs the HZ boundaries above
def get_hz_status(row):
  a, hi, ho = row["pl_orbsmax"], row["hz_inner"], row["hz_outer"]
  if pd.isna(a) or pd.isna(hi) or pd.isna(ho): return "Unknown"
  elif a < hi:  return "Too Hot"
  elif a > ho:  return "Too Cold"
  else:         return "In HZ"

df["hz_status"] = df.apply(get_hz_status, axis=1)

# 5.) EARTH SIMILARITY INDEX (ESI) =============================================================================
# The ESI is a composite score comparing a planet to Earth across three properties: radius, bulk density, and equilibrium temperature
# Each property contributes with a different weight to the ESI scale
# A perfect ESI of 1.0 means identical to Earth in radius, density, and equialibrium temp
# Assume at least 2/3 of the properties must be available to compute an ESI for this
print("5.)")

# Earth reference values for each property used in the ESI
EARTH_REF = {
"pl_rade": 1.00, # 6,371 km
"pl_dens": 5.51, # 5.51 g/cm^3
"pl_eqt": 255.0  # 255 k 2/out atmosphere
}

# Exponent weights for each property in the ESI calculation
ESI_WEIGHTS = {
"pl_rade": 0.57, 
"pl_dens": 1.07, 
"pl_eqt": 5.58
}

# Sum of weights used to normalize the exponents in the total formula
W_TOTAL = sum(ESI_WEIGHTS.values())


def compute_esi(row):
  score, n = 1.0, 0
  for col, w in ESI_WEIGHTS.items():
    xi, x0 = row[col], EARTH_REF[col]
    if pd.isna(xi): continue
    # Each factor is a similarity term 1 when xi=x0 its decreasing as they diverge <- figured out from online
    # The exponent w/W_TOTAL makes it proportional to its weight
    score *= (1.0 - abs(xi - x0) / (xi + x0)) ** (w / W_TOTAL)
    n += 1
  # Return NaN if <2 properties were available so not enough data :(
  return score if n >= 2 else np.nan

df["ESI"] = df.apply(compute_esi, axis=1)

# TERRESTRIAL PLANET HZ FILTER =============================================================================
# Little data on density so if this is unavailbe fall back to radius data for planet classification
# - If density is missing then fall back to radius < 1.6 R_Earth
# - If both are missing then exclude the planet

def is_terrestrial_candidate(row):
  if not pd.isna(row["pl_dens"]):
    return row["pl_dens"] > 3.9
  elif not pd.isna(row["pl_rade"]):
    return row["pl_rade"] < 1.6
  return False

df["is_terrestrial"] = df.apply(is_terrestrial_candidate, axis=1)

terr_hz = df[df["is_terrestrial"] & (df["hz_status"] == "In HZ")]
print(f" Terrestrial planets in the habitable zone = {len(terr_hz)}")
print(f" density classified = {(terr_hz['planet_type'] == 'Terrestrial').sum()}")
print(f" radius back up = {(terr_hz['planet_type'] == 'Unknown').sum()}")

# Had to fix this later on cause the plot was fitting 14 exoplanets but it was giving me 15 esi ranking so there was just one floating esi number
top15 = terr_hz.dropna(subset=["ESI"]).nlargest(15, "ESI").drop_duplicates(subset="pl_name", keep="first") 

print("="*70)
print(f"Top 5 most Earth like exoplanets")
print(top15[["pl_name", "pl_rade", "pl_dens", "pl_eqt", "ESI", "source", "confirmed"]].head().to_string(index=False))

n_conf   = top15["confirmed"].eq("Confirmed").sum()
n_unconf = top15["confirmed"].eq("Unconfirmed").sum()
print("="*70)
print(f"\nOf the top 15, {n_conf} = Confirmed, {n_unconf} = Unconfirmed \n")

# 6.) H-R DIAGRAM OF EXOPLANET HOST STARS ===================================================================
# Plots effective temperature (x-axis, reversed) vs luminosity (y-axis)
print("6.)")

fig, ax = plt.subplots()

# Drop rows missing temperature or luminosity, and deduplicate by host star
# Each star is only plotted once even if it hosts multiple planets like trappist
hr = df.dropna(subset=["st_teff", "st_lum_lin"]).drop_duplicates("hostname")

# Plot each spectral class as a separate scatter series so they get differetn colors
for sclass in ["O", "B", "A", "F", "G", "K", "M"]:
  grp = hr[hr["spec_class"] == sclass]
  if len(grp) == 0: continue
  ax.scatter(grp["st_teff"], grp["st_lum_lin"], s=10, alpha=0.5, label=f"Type {sclass}")

# Plot stars with unknown or unclassified spectral types in gray :(
unknown = hr[hr["spec_class"] == "Unknown"]
ax.scatter(unknown["st_teff"], unknown["st_lum_lin"], s=6, alpha=0.3, color="gray")

# Mark the Sun as a gold star symbol :D  T=5778 K, L=1 L_sun
ax.scatter([5778], [1.0], marker="*", s=100, color="gold", edgecolors="black", linewidths=0.5, zorder=5, label="Sun")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(12000, 2500)
ax.set_ylim(0.001, 3500)
ax.invert_xaxis()
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Luminosity (L / L_sun)")
ax.set_title("H-R Diagram of Exoplanet Host Stars\n(NASA & EU Data)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# PLANET MASS-RADIUS DIAGRAM =============================================================================
# Shows the relationship between planet mass and radius, colored by planet type
# derived from bulk density with different classification cut offs and unknowns 

fig, ax = plt.subplots()

# Drop rows where either mass or radius is missing
mr = df.dropna(subset=["pl_masse", "pl_rade"])

# Plot each density-classified planet type as a separate scatter series
for ptype in ["Terrestrial", "Sub-Neptune", "Neptune-like", "Gas Giant", "Unknown"]:
  grp = mr[mr["planet_type"] == ptype]
  if len(grp) == 0: continue
  ax.scatter(grp["pl_masse"], grp["pl_rade"], s=8, alpha=0.5, label=ptype)

# Classification cutoffs
m_range = np.logspace(-1, 4, 300)
for rho, label in [
  (0.5,  "0.5 g/cm^3  (Gas Giant / Neptune-like)"),
  (1.5,  "1.5 g/cm^3  (Neptune-like / Sub-Neptune)"),
  (3.9,  "3.9 g/cm^3  (Sub-Neptune / Terrestrial)"),
  (5.51, "5.5 g/cm^3  (Earth)"),
]:
  r_cm    = (3 * m_range * 5.972e27 / (4 * np.pi * rho)) ** (1.0 / 3.0)
  r_earth = r_cm / 6.371e8
  ax.plot(m_range, r_earth, "k--", lw=0.8, alpha=0.5)
  ax.text(m_range[-40], r_earth[-40] * 1.06, label, fontsize=7)

# Mark Earth at (1, 1) for reference
ax.scatter([1], [1], marker="o", s=100, color="#4f4cb0", edgecolors="#8eb057", linewidths=1.5, zorder=5, label="Earth")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 10000)
ax.set_ylim(0.4, 40)
ax.set_xlabel("Planet Mass (M_Earth)")
ax.set_ylabel("Planet Radius (R_Earth)")
ax.set_title("Planet Type Classification From Density\n(NASA & EU Data)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# 7.) HABITABLE ZONE PLOT =============================================================================
# Shows orbital distance vs stellar luminosity for all planets.
# The green shaded regions are the habitable zone where liquid water could exist on the surface of a rocky planet.
# Optimistic HZ (0.95 - 1.70 AU) lighter green
# Conservative HZ (0.99 - 1.67 AU) darker green
# Orange stars highlight terrestrial planets that fall within the HZ
print("7.)")

fig, ax = plt.subplots()

# Kill rows missing orbital distance or stellar luminosity
hz_plot  = df.dropna(subset=["pl_orbsmax", "st_lum_lin"])

# Range of luminosities (y-axis)
lum_arr  = np.logspace(-3, 2, 400)

# Optimistic habitable zone 
ax.fill_betweenx(lum_arr, 0.95 * np.sqrt(lum_arr), 1.70 * np.sqrt(lum_arr), alpha=0.15, color="green", label="Optimistic HZ")

# Conservative habitable zone 
ax.fill_betweenx(lum_arr, 0.99 * np.sqrt(lum_arr), 1.67 * np.sqrt(lum_arr), alpha=0.35, color="green", label="Conservative HZ")

# Plot all planets colored by type
for ptype in ["Terrestrial", "Sub-Neptune", "Neptune-like", "Gas Giant"]:
  grp = hz_plot[hz_plot["planet_type"] == ptype]
  ax.scatter(grp["pl_orbsmax"], grp["st_lum_lin"], s=6, alpha=0.4, label=ptype)

# Highlight terrestrial planets that are inside the HZ with a larger orange star
if len(terr_hz) > 0:
  ax.scatter(terr_hz["pl_orbsmax"], terr_hz["st_lum_lin"], s=50, marker="*", color="orange", edgecolors="gray", linewidth = 0.5,  zorder=5, label="Terrestrial in HZ")

# Mark Earth at (1.0 AU, 1.0 L_sun)
ax.scatter([1.0], [1.0], marker="*", s=100, color="gold", edgecolors="black", linewidths=0.5, zorder=6, label="Earth")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.005,20)
ax.set_ylim(0.001, 100)
ax.set_xlabel("Semi-Major Axis (AU)")
ax.set_ylabel("Luminosity (L / L_sun)")
ax.set_title("Habitable Zone\n" "(NASA & EU Data)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# 8.) ESI RANKING BAR CHART (TOP 15 MOST EARTH LIKE) ===========================================================
# Blue bars = Confirmed, Red bars = Unconfirmed
print("8.)")

fig, ax = plt.subplots()
if len(top15) == 0:
  print("No terrestrial HZ planets with enough data to find ESI")
else:
  plot_data  = top15.sort_values("ESI")
  bar_colors = ["blue" if c == "Confirmed" else "red" for c in plot_data["confirmed"]]
  bars = ax.barh(plot_data["pl_name"], plot_data["ESI"], color=bar_colors, edgecolor="white")

  for bar, (_, row) in zip(bars, plot_data.iterrows()):
    ax.text(row["ESI"] + 0.001, bar.get_y() + bar.get_height() / 2, f"{row['ESI']:.3f}", va="center", fontsize=8)

  ax.axvline(1.0, color="black", lw=1, ls="--", label="Earth (ESI = 1.0)")

  # Make legend
  ax.scatter([], [], color="blue", label="Confirmed")
  ax.scatter([], [], color="red",  label="Unconfirmed")
  ax.legend(fontsize=8, loc="lower right")

  ax.set_xlabel("Earth Similarity Index (ESI)")
  ax.set_title("Most Earth like Exoplanets\n(Terrestrial planets in the Habitable Zone ranked on ESI scale)")
  ax.set_xlim(0.9, 1.01)
  ax.grid(axis="x", alpha=0.3)
  plt.tight_layout()
  plt.show()

# 9.) TRAPPIST-1 e JWST TRANSMISSION SPECTRUM ==================================================================
# Loads observed transit depth vs wavelength data and corrects it using TRAPPIST-1 b observations
# Find molecular absorption windows too and check if the corrected trappist-1 e data got that
print("9.)")

# File paths for the spectral data and the index table
SPEC_FILE   = "table_TRAPPIST-1-e-Espinoza-et-al.-2025 (1).csv"
SPEC_FILE_B = "table_TRAPPIST-1-b-Rathcke-et-al.-2025 (1).csv"
INDEX_FILE  = "table.csv"

# Wavelength windows (in microns) where key atmospheric molecules/atoms absorb <- found online
# A dip or excess in transit depth within these windows can indicate the presence of that molecules in the planet's potential atmosphere
# Detections here  should be interpreted with some discretion caues its so far away with such little data

MOL_WINDOWS = {
  # Water vapor -> three bands
  "H2O 1.4 um":  (1.35, 1.50),
  "H2O 1.9 um":  (1.75, 2.05),
  "H2O 2.7 um":  (2.55, 2.90),
  # Carbon dioxide
  "CO2 4.3 um":  (4.10, 4.50),
  # Methane -> could imply nitrogen atmosphere like earth
  "CH4 3.3 um":  (3.10, 3.50),
  # Carbon monoxide
  "CO  4.7 um":  (4.60, 4.80),
  # Oxygen molecule
  "O2  0.76 um": (0.750, 0.775),
}

# Continuum reference window: a wavelength region with no strong molecular features <- from internet
# can be though of as space background noise
CONT_WINDOW = (1.15, 1.30)

# Convert transit depth values to fractional (Rp/Rs)^2 depending on if that data was stored as ppm, percent, or already fractional or whatever

def to_fractional(arr):
  mx = np.nanmax(np.abs(arr))
  if mx > 100:
    return arr * 1e-6,  "ppm -> fractional"
  elif mx > 0.1:          # catches percent values like 0.5-0.9%
    return arr / 100.0, "percent -> fractional"
  else:
    return arr.copy(),  "already fractional"
# Load the index file and verify that the TRAPPIST-1 b dataset we are using matches the expected wavelength ranges we need to normalize the TRAPPIST-1 e data
index = pd.read_csv(INDEX_FILE)
index.columns = [c.strip() for c in index.columns]
b_meta = index[
  (index["PL_NAME"].str.strip().str.upper() == "TRAPPIST-1 B") &
  (index["SPEC_TYPE"].str.strip().str.lower() == "transmission") &
  (index["AUTHORS"].str.contains("Rathcke", na=False)) 
]

row = b_meta.iloc[0]
print("Index confirms TRAPPIST-1 b reference data")
print(f"Wavelength  = {row['MINWAVELNG']} - {row['MAXWAVELNG']} um")
print(f"Data points = {row['NUM_DATAPOINTS']}")

# Initialize variables used later for atmospheric classification
inferred_features = {}
h2o_signal        = 0.0
co2_signal        = 0.0
ch4_signal        = 0.0
o2_signal         = 0.0
co_signal         = 0.0

# Load TRAPPIST-1 b spectrum
raw_spec_b = pd.read_csv(SPEC_FILE_B, comment="#", low_memory=False)
raw_spec_b.columns = [c.strip().lower().replace(" ", "_") for c in raw_spec_b.columns]
cols_b = list(raw_spec_b.columns)

wave_col_b  = "centralwavelng"
depth_col_b = "pl_trandep"

# Fix and convert units
spec_b  = raw_spec_b.dropna(subset=[wave_col_b, depth_col_b]).sort_values(wave_col_b).reset_index(drop=True)
wave_b  = spec_b[wave_col_b].values.astype(float)
depth_b, b_units = to_fractional(spec_b[depth_col_b].values.astype(float))

print("=" * 70)
print(f" TRAPPIST-1 b (Rathcke et al. 2025) = {len(wave_b)} points, {wave_b.min():.4f}-{wave_b.max():.4f} um")
print(f" Depth units detected = {b_units}")
print(f" Depth range (frac) = {depth_b.min():.6f} - {depth_b.max():.6f}")

# Load TRAPPIST-1 e spectrum 
raw_spec = pd.read_csv(SPEC_FILE, comment="#", low_memory=False)

raw_spec.columns = [c.strip().lower().replace(" ", "_") for c in raw_spec.columns]
cols = list(raw_spec.columns)

wave_col  = "centralwavelng"
depth_col = "pl_trandep"
err_col   = "pl_trandeperr1"

print(f" Wavelength column = {wave_col}")
print(f" Depth column = {depth_col}")
print(f" No column = {err_col if err_col else 'none found'}")

# If the file contains multiple planets, filter to only TRAPPIST-1 e rows
# If no planet name column is found or no matching rows, use the whole file
name_col = next(
  (c for c in ["plntname", "pl_name", "planet", "name", "planet_name"] if c in cols), None
)

if name_col:
  mask_t1e = raw_spec[name_col].astype(str).str.strip().str.upper() == "TRAPPIST-1 E"
  spec_t1e = raw_spec[mask_t1e].copy()
  if len(spec_t1e) == 0:
    spec_t1e = raw_spec.copy()
else:
  spec_t1e = raw_spec.copy()

# Fix and convert units for TRAPPIST-1 e
spec_t1e  = spec_t1e.dropna(subset=[wave_col, depth_col]).sort_values(wave_col).reset_index(drop=True)
wave      = spec_t1e[wave_col].values.astype(float)
depth_raw = spec_t1e[depth_col].values.astype(float)
errs_raw  = spec_t1e[err_col].values.astype(float) if err_col else None

depth_frac, e_units = to_fractional(depth_raw)
errs_frac, _        = to_fractional(errs_raw) if errs_raw is not None else (None, None)
print("TRAPPIST-1 e:")
print(f"Depth units detected = {e_units}")

print(f"\n  Wavelength range = {wave.min():.4f} - {wave.max():.4f} um")
print(f"  Data points = {len(wave)}")
print(f"  Depth range = {depth_frac.min():.6f} - {depth_frac.max():.6f}")

# 10.) Stellar contamination correction =============================================================================
# Turn the TRAPPIST-1 b transit depth onto the same wavelength grid as TRAPPIST-1 e, then subtract it
print("10.)")

from scipy.interpolate import interp1d
interp_b        = interp1d(wave_b, depth_b, kind="linear", bounds_error=False, fill_value="extrapolate")
depth_b_on_e    = interp_b(wave)
depth_frac_corr = depth_frac - depth_b_on_e
print(f"Applied TRAPPIST-1 b stellar contamination correction")
print(f"b wavelength coverage : {wave_b.min():.4f}-{wave_b.max():.4f} um")
print(f"e wavelength coverage : {wave.min():.4f}-{wave.max():.4f} um")

# Molecular / atomic feature detection
# Fompute the mean transit depth and compare it to the continuum baseline. A positive excess means more light is blocked <- looked up for signature of atmospheric absorption.
# Return the mean corrected transit depth for all data points in the specific wavelength window for everything

def mean_depth_in_window(w_arr, d_arr, w_lo, w_hi):
  mask = (w_arr >= w_lo) & (w_arr <= w_hi)
  return float(d_arr[mask].mean()) if mask.sum() > 0 else np.nan

# Find the continuum / baseline depth in the reference window
cont_depth = mean_depth_in_window(wave, depth_frac_corr, *CONT_WINDOW)
print(f"\n  Continuum reference ({CONT_WINDOW[0]:.2f}-{CONT_WINDOW[1]:.2f} um): {cont_depth:.6f} (fractional number)")
print("\n  Molecular / atomic feature analysis:")
print(f"  {'Species':<20} {'Excess':>10}   Status")
print("  " + "-" * 60)

# For loop over each window and find the fractional excess above baseline
for mol, (wlo, whi) in MOL_WINDOWS.items():
  if wave.max() < wlo:
    print(f"  {mol:<20} {'---':>10}   out of wavelength range")
    continue
  if wave.min() > whi:
    print(f"  {mol:<20} {'---':>10}   out of wavelength range (blue edge)")
    continue
  in_depth = mean_depth_in_window(wave, depth_frac_corr, wlo, whi)
  if np.isnan(in_depth) or np.isnan(cont_depth) or cont_depth == 0:
    print(f"  {mol:<20} {'---':>10}   insufficient data")
    continue
    
  # How much deeper is the transit in the specific window compared to the continuum baseline
  # Positive = absorption, negative = emission or maybe noise
  excess = (in_depth - cont_depth) / cont_depth
  if   excess > 0.005: status = "detected  (>0.5% excess)"
  elif excess > 0.001: status = "weak (>0.1% excess)"
  else:                status = "not detected"
  print(f"  {mol:<20} {excess * 100:>+9.2f}%   {status}")
  inferred_features[mol] = excess

# Summarize the strongest signal for each thing we're looking for
h2o_signal = max(
  inferred_features.get("H2O 1.4 um", 0.0),
  inferred_features.get("H2O 1.9 um", 0.0),
  inferred_features.get("H2O 2.7 um", 0.0),)
co2_signal = inferred_features.get("CO2 4.3 um", 0.0)
ch4_signal = inferred_features.get("CH4 3.3 um", 0.0)
co_signal  = inferred_features.get("CO  4.7 um", 0.0)
o2_signal  = inferred_features.get("O2  0.76 um", 0.0)

# Classify the overall atmospheric state based on signal strengths so,
# atmosphere_detected = at least one species shows >0.5% excess
# weak_atmosphere = at least one species shows >0.1% excess
# flat_spectrum = no significant features found anywhere

if any(s > 0.005 for s in [h2o_signal, co2_signal, ch4_signal, co_signal, o2_signal]):
  atmo_state = "atmosphere_detected"
elif any(v > 0.001 for v in inferred_features.values()):
  atmo_state = "weak_atmosphere"
else:
  atmo_state = "flat_spectrum"

atmo_summary = {
  "atmosphere_detected": "Atmospheric absorption features detected",
  "weak_atmosphere":     "Tentative atmospheric features (weak signal)",
  "flat_spectrum":       "Flat spectrum -- consistent with no thick atmosphere",
}[atmo_state]

print(f"\n  Inferred likely atmospheric state: {atmo_state.replace('_', ' ')}")
print(f"  Interpretation: {atmo_summary} of molecules above")

# 11.) TRAPPIST-1 e TRANSMISSION SPECTRUM PLOT =============================================================================
# Plots transit depth vs wavelength with error bars 
# Colored vertical bands highlight where different molecules/atoms absorb
# The gray band shows the continuum reference bassline 
# The red band where stellar activity can contaminate the signal
# The horizontal dashed line marks the end of background emission
print("11.)")

fig6, ax6 = plt.subplots(figsize=(13, 5))

# Plot with error bars if available and otherwise just connected points
if errs_frac is not None:
  ax6.errorbar(
    wave, depth_frac_corr, yerr=errs_frac,
    fmt="o", ms=2.5, lw=0, elinewidth=0.8, capsize=0,
    color="blue", alpha=0.85,
    label="Transit Spectrum (corrected w/ error bars)",
  )
else:
  ax6.plot(
    wave, depth_frac_corr, "o-", ms=2.5, lw=0.7,
    color="blue", alpha=0.85,
    label="Transit Spectrum (corrected)",
  )

# Pretty Colora for each molecular/atomic species' shaded window
mol_shade = {
  "H2O": ("deepskyblue", 0.34),
  "CH4": ("limegreen",   0.34),
  "CO2": ("salmon",      0.34),
  "CO":  ("gold",        0.34),
  "O2":  ("mediumpurple",0.34),
}

# Shade each species absorption window with the correct color 
# and only add one legend entry per species cause multiple windows share a color
labeled_mols = set()
for mol, (wlo, whi) in MOL_WINDOWS.items():
  if wave.max() < wlo or wave.min() > whi:
    continue
  mol_base = mol.split(" ")[0]
  col, al  = mol_shade.get(mol_base, ("plum", 0.20))
  lbl = mol_base if mol_base not in labeled_mols else None
  ax6.axvspan(wlo, whi, alpha=al, color=col, label=lbl)
  labeled_mols.add(mol_base)

# Shade the continuum reference window in gray
ax6.axvspan(*CONT_WINDOW, alpha=0.10, color="gray", label="Stellar Background")

# Shade region affected by stellar chromospheric emission/flares
ax6.axvspan(0.55, 0.75, alpha=0.15, color="red", label="Stellar Interference")

# Horizontal line marking where background emission ends to see what molecules are above or bellow that
ax6.axhline(y=-0.002222, color="black", linestyle="--", linewidth=1.0, label="Background Baseline")

# Add this after all the ax6 plotting calls, before ax6.legend(...)
y_data = depth_frac_corr
y_pad  = (np.nanmax(y_data) - np.nanmin(y_data)) * 0.3
ax6.set_ylim(np.nanmin(y_data) - y_pad, np.nanmax(y_data) + y_pad)
ax6.set_xlabel("Wavelength (microns)")
ax6.set_ylabel("Transit Depth (Corrected)")
ax6.set_title(
  "TRAPPIST-1 e Transit Spectrum\n"
  "(TRAPPIST-1 b stellar contamination subtracted)\n"
  f"Atmospheric interpretation = {atmo_summary}"
)

ax6.legend(fontsize=8, ncol=4, loc="upper right")
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 12.) SECTION 8: TRAPPIST-1 SYSTEM ORBITAL DIAGRAM =============================================================================
# Draws the 7 TRAPPIST-1 planets as circles on their orbits around its star TRAPPIST-1
# with the habitable zone shown as green ring
# Planet sizes are not to scale
# orbital distances are roughly to scale
print("12.)")
print("=" * 70)

# Known physical properties of the TRAPPIST-1 system from Agol et al. 2021 data
T1_LUM_LOG = -3.28                   # log10(L/L_sun) for TRAPPIST-1
T1_LUM_LIN = 10.0 ** T1_LUM_LOG     # linear luminosity

T1E_ORBSMA = 0.02928285  # AU 
T1E_RADE   = 0.920       # R_Earth
T1E_MASSE  = 0.772       # M_Earth
T1E_PERIOD = 6.100       # days
T1E_INSOL  = 0.662       # S_Earth (insolation relative to Earth)
T1_TEFF    = 2566.0      # K (effective temperature of TRAPPIST-1)

# Compute equilibrium temperature for TRAPPIST-1 e using Stefan-Boltzmann with bond albedo correction
T1E_TEQ = (
  278.5
  * T1_LUM_LIN ** 0.25
  / np.sqrt(T1E_ORBSMA)
  * (1.0 - BOND_ALBEDO) ** 0.25
)

# Estimate surface temperature and factor in the greenhouse effect
T1E_TSURF = T1E_TEQ * GH_FACTOR

print(f"T_eq (A=0.30) = {T1E_TEQ:.1f} K")
print(f"T_surf estimate = {T1E_TSURF:.1f} K")

# Semi-major axes and eccentricities for all 7 TRAPPIST-1 planets from Agol et al. 2021 data
TRAPPIST_PLANETS = {
  "b": {"a": 0.01154775, "e": 0.00622},
  "c": {"a": 0.01581512, "e": 0.00654},
  "d": {"a": 0.02228038, "e": 0.00837},
  "e": {"a": 0.02928285, "e": 0.00510},
  "f": {"a": 0.03853361, "e": 0.01007},
  "g": {"a": 0.04687692, "e": 0.00208},
  "h": {"a": 0.06193488, "e": 0.00567},
}

# Habitable zone boundaries for TRAPPIST-1 using Kopparapu et al. 2013 data scaling
# Much closer to the star than Earth's HZ because TRAPPIST-1 is dim :(
HZ_INNER_OPT  = 0.95 * np.sqrt(T1_LUM_LIN)
HZ_OUTER_OPT  = 1.70 * np.sqrt(T1_LUM_LIN)
HZ_INNER_CONS = 0.99 * np.sqrt(T1_LUM_LIN)
HZ_OUTER_CONS = 1.67 * np.sqrt(T1_LUM_LIN)

fig8, ax8 = plt.subplots(figsize=(10, 7))

# Make a circle angle array for drawing orbits and annuli
theta = np.linspace(0, 2 * np.pi, 500)

# Build the HZ rings using two Circle patches each
# Outer filled circle overdrawing with white inner circle creates the annulus
from matplotlib.patches import Circle
hz_opt_outer = Circle((0, 0), HZ_OUTER_OPT,  color="green", alpha=0.15, zorder=1)
hz_opt_inner = Circle((0, 0), HZ_INNER_OPT,  color="white", alpha=1.0,  zorder=2)
hz_con_outer = Circle((0, 0), HZ_OUTER_CONS, color="green", alpha=0.30, zorder=3)
hz_con_inner = Circle((0, 0), HZ_INNER_CONS, color="white", alpha=1.0,  zorder=4)

ax8.add_patch(hz_opt_outer)
ax8.add_patch(hz_opt_inner)
ax8.add_patch(hz_con_outer)
ax8.add_patch(hz_con_inner)

# Redraw the star dot on top of the white inner circles so it is not hidden
ax8.scatter([0], [0], marker="*", s=150, color="red", zorder=10)

# Draw each planet's elliptical orbit and place its symbol at a fixed angle
# Eccentric used for planet placement angle
angle_deg = {"b": 30, "c": 70, "d": 120, "e": 170, "f": 220, "g": 270, "h": 320}

for letter, params in TRAPPIST_PLANETS.items():
  a = params["a"]
  e = params["e"]
  b = a * np.sqrt(1 - e**2)   # semi-minor axis
  
  # Ellipse centered at (a*e, 0) so the star sits at the origin
  x_orbit = a * np.cos(theta) - a * e
  y_orbit = b * np.sin(theta)
  ax8.plot(x_orbit, y_orbit, color="gray", lw=0.75, ls="--", zorder=5)

  # Place each planet at its fixed dislay angle
  ang = np.radians(angle_deg[letter])
  px  = a * np.cos(ang) - a * e
  py  = b * np.sin(ang)

  if letter == "e":
    # Highlight TRAPPIST-1 e with a larger dot
    ax8.plot(px, py, "o", ms=9, color="#4f4cb0", zorder=11, markeredgecolor="#8eb057", markeredgewidth=1.5)
  else:
    # Make other TRAPPIST-1 planets smaller gray dots with appropriate planet letter label
    ax8.plot(px, py, "o", ms=6, color="gray", zorder=11, markeredgecolor="black", markeredgewidth=0.6)
    ax8.annotate(
      letter, xy=(px, py),
      xytext=(px + 0.002, py + 0.002),
      fontsize=8, zorder=12,
    )

# Custom legend for HZ shading and planet markers
legend_handles = [
  mpatches.Patch(facecolor="green", alpha=0.45, label="Conservative HZ"),
  mpatches.Patch(facecolor="green", alpha=0.15, label="Optimistic HZ"),
  plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4f4cb0", markersize=9, markeredgecolor="#597037", label="TRAPPIST-1 e"),
  plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",    markersize=6, markeredgecolor="black",   label="Other TRAPPIST-1 planets"),
  plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red",     markersize=15,                            label="TRAPPIST-1 (Host Star: M-Type Dwarf Star)"),
]
ax8.legend(handles=legend_handles, fontsize=8, loc="upper right")

# Set equal aspect ratio so orbits appear correctly proportioned
max_sma = max(p["a"] for p in TRAPPIST_PLANETS.values()) * 1.25
ax8.set_xlim(-max_sma, max_sma)
ax8.set_ylim(-max_sma, max_sma)
ax8.set_aspect("equal")
ax8.set_xlabel("Distance (AU)")
ax8.set_ylabel("Distance (AU)")
ax8.set_title("TRAPPIST-1 System Orbital Diagram")
ax8.grid(True, alpha=0.3)

# Summary of parameters for TRAPPIST-1 e and its host star
props_text = (
  f"Exoplanet TRAPPIST-1 e:\n"
  f"Radius = {T1E_RADE} R_sun\n"
  f"Mass = {T1E_MASSE} M_sun\n"
  f"Period = {T1E_PERIOD} days\n"
  f"a = {T1E_ORBSMA} AU\n"
  f"Insol = {T1E_INSOL} S_sun\n"
  f"T_eq = {T1E_TEQ:.1f} K\n"
  f"T_surf = {T1E_TSURF:.1f} K\n"
  f"\nHost star TRAPPIST-1:\n"
  f"Type: M8V Dwarf Star \n"
  f"T_eff = {T1_TEFF:.0f} K\n"
  f"log L = {T1_LUM_LOG} L_sun"
)

ax8.text(
  0.02, 0.02, props_text,
  transform=ax8.transAxes, fontsize=8,
  va="bottom", ha="left",
  bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),)
plt.tight_layout()

plt.show()