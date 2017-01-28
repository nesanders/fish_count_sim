# fish_count_sim
Generate synthetic data for a fish counting video server

## Contents

* MyRWA_fishcount_sampledata_simple.py - Script to generate dataset.  'Settings' section has configurable parameters.
* simulated_fish_videos.csv - Video metadata
* simulated_fish_crossings.csv - Individual fish crossing times
* fish_counts_minute_sim.png - view of full simulated dataset, fish volume minute-to-minute
* fish_counts_minute_sim_zoom?.png - view of full simulated dataset, zoomed in at various illustrative time points
* fish_counts_histogram.png - view of the distribution of video durations in seconds on a log scale.  Expect to see a power law distribution, with an additional spike at the minimum time (inactivity timer length)

## Notes

* Input data not distributed with this repository
