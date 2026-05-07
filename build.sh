#!/bin/sh

set -e

docker build -f environment/Dockerfile -t ghcr.io/benjamin-heasly/geffenlab-data-summary:local .

docker run --rm \
  --user $(id -u):$(id -g) \
  --volume /home/ninjaben/codin/geffen-lab-data/raw_data/:/home/ninjaben/codin/geffen-lab-data/raw_data/ \
  --volume /home/ninjaben/codin/geffen-lab-data/processed_data/:/home/ninjaben/codin/geffen-lab-data/processed_data/ \
  --volume /home/ninjaben/codin/geffen-lab-data/analysis/:/home/ninjaben/codin/geffen-lab-data/analysis/ \
  ghcr.io/benjamin-heasly/geffenlab-data-summary:local \
  --raw-data-root /home/ninjaben/codin/geffen-lab-data/raw_data \
  --processed-data-root /home/ninjaben/codin/geffen-lab-data/processed_data \
  --analysis-root /home/ninjaben/codin/geffen-lab-data/analysis \
  --experimenter BH \
  --subject AS20-minimal3 \
  --date "03112025" \
  --event-times-pattern "tprime/*/*nidq.xd_8_3_0.txt" \
  --multiplot
