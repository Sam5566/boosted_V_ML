#! /bin/bash

find best_model/* -size -10M -type f -print0 | xargs -0 git add
find CNN/* -size -2M -type f -print0 | xargs -0 git add
find sample/event_base/* -size -10M -type f -print0 | xargs -0 git add
find sample/jet_base/* -size -10M -type f -print0 | xargs -0 git add
find BDT/* -size -2M -type f -print0 | xargs -0 git add
find analysis/* -size -10M -type f -print0 | xargs -0 git add
