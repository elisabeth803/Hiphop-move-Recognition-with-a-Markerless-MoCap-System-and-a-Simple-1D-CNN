import os
from pathlib import Path
import musicalgestures as mg

video_root = Path("videoer")

for thing in video_root.iterdir():
    if thing.is_file():
        video = mg.MgVideo(
                str(thing),
                starttime=3,
                endtime=19
            )
