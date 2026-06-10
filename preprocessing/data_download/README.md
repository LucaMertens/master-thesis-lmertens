# data_download

Download GrandTour ROS1 bags from Kleinkram into `/datahome/lmertens/grand_tour_dataset/ros1/`.

- `download_wayfaster_data.py` — bulk download of the bags the labeling pipeline
  needs (`alphasense`, `cpt7_ie_tc`, `tf_minimal`) for all missions.
  Run with `uv run python download_wayfaster_data.py`.
- `download_data.ipynb` — interactive notebook for exploring what files are
  available per mission before downloading. Not used for bulk downloads.
