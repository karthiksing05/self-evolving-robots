# YCB object assets: attribution and licence

The files in this directory are derived from the YCB Object and Model Set:

> B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel, A. M. Dollar. "Benchmarking in
> Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols."
> arXiv:1502.03143 (2015); IEEE Robotics & Automation Magazine 22 (2015) 36-52,
> doi:10.1109/MRA.2015.2448951.

Source data: the Google 16k scans at https://ycb-benchmarks.s3.amazonaws.com/data/google/
(project site https://www.ycbbenchmarks.com/).  Masses are from Table II of the paper above.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/.  The YCB data page
(https://ycb-benchmarks.s3.amazonaws.com/index.html, checked 2026-11-01) states "License for the
data set: Creative Commons Attribution 4.0 International (CC BY 4.0)" and "License for the code:
MIT license".  ycbbenchmarks.com itself returned HTTP 500 that day.

Changes: these are processed derivatives, not the original scans, produced by
`scripts/import_ycb.py`.  Each scan was re-posed (resting on the table, narrow side along y,
bottom at z = 0), reduced to convex pieces (`part_<i>`: the convex hull for nearly convex objects,
else a CoACD decomposition; hull vertices, thinned to at most 256),
decimated to at most 3000 faces by vertex clustering (`visual_v`, `visual_f`), reduced to one
average texture colour (`rgba`), and given the mass properties of a uniform solid at the published
mass (`com`, `inertia`).  The original meshes and textures are not redistributed here.
