#!/bin/bash
# The whole replication. Set the paths below (or export them) and run from the repository root:
#   DATASHARE=~/data/datashare.zip GOYAL=~/data/goyal_Monthly.csv JKP=~/data/jkp_usa.parquet WRDS_USER=... scripts/reproduce.sh
# DATA is the folder for the panels, the WRDS cache and the work folders (about 30 GB). WORKERS caps the processes.
set -euo pipefail
DATA=${DATA:-$PWD/replication}
WORKERS=${WORKERS:-16}
mkdir -p "$DATA"
echo "== simulation study (Online Appendix C)"
cmk simulate --out "$DATA/simulation"
echo "== main panel (Section 7)"
[ -f "$DATA/panel_gkx/datefile.txt" ] || cmk panel gkx --datashare "$DATASHARE" --goyal "$GOYAL" --crsp-cache "$DATA/crsp" --out "$DATA/panel_gkx"
cmk all --config gkx --panel-dir "$DATA/panel_gkx" --work "$DATA/work_gkx" --workers "$WORKERS"
echo "== second panel (Online Appendix D.2)"
[ -f "$DATA/panel_jkp/datefile.txt" ] || cmk panel jkp --parquet "$JKP" --goyal "$GOYAL" --crsp-cache "$DATA/crsp" --out "$DATA/panel_jkp"
cmk all --config jkp --panel-dir "$DATA/panel_jkp" --work "$DATA/work_jkp" --workers "$WORKERS"
echo "== done: $DATA/work_gkx/output, $DATA/work_jkp/output, $DATA/simulation"
