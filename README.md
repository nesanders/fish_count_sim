# fish_count_sim

Generate synthetic data for a fish counting video server

## Contents

* MyRWA_fishcount_sampledata_simple_discrete.py - Script to generate dataset.  'Settings' section has configurable parameters.

* simulated_fish_videos.csv - Video metadata
* simulated_fish_crossings.csv - Individual fish crossing times, expressed as entrance and exit times relative to the starting epoch in seconds

* fish_counts_minute_sim.png - view of full simulated dataset, fish volume minute-to-minute
* fish_counts_minute_sim_zoom?.png - view of full simulated dataset, zoomed in at various illustrative time points

* fish_counts_histogram.png - view of the distribution of fish counts in each video
* fish_duration_histogram.png - view of the distribution of video durations in seconds on a log scale.  Expect to see a power law distribution, with an additional spike at the maximum duration
* fish_crossing_times_histogram.png - view of the distribution of how long each fish stays in the frame.  This is pre-specified as a setting.
* video_counts_perday.png - The total number of videos recorded each day of the season

## Notes

* This commit contains outputs from two runs of the script, one with a minimum activity timer between video splits of 4s and one with 20s.
* Input data not distributed with this repository
